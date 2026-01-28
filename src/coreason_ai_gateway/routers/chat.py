# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Annotated, Any, AsyncIterator

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    Request,
)
from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from coreason_ai_gateway.config import get_settings
from coreason_ai_gateway.dependencies import RedisDep, get_upstream_client, validate_request_budget
from coreason_ai_gateway.middleware.accounting import record_usage
from coreason_ai_gateway.middleware.auth import verify_gateway_token
from coreason_ai_gateway.schemas import ChatCompletionRequest
from coreason_ai_gateway.utils.logger import logger

router = APIRouter()


@router.post("/v1/chat/completions", status_code=200)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    client: Annotated[AsyncOpenAI, Depends(get_upstream_client)],
    redis_client: RedisDep,
    x_coreason_project_id: Annotated[str, Header()],
    token: Annotated[str, Depends(verify_gateway_token)],
    _budget: Annotated[None, Depends(validate_request_budget)],
    x_coreason_trace_id: Annotated[str | None, Header()] = None,
) -> Any:
    """
    Proxies chat completion requests to LLM providers.
    Enforces authentication, budgeting, and JIT secret injection via dependencies.

    Args:
        request (Request): The incoming HTTP request.
        body (ChatCompletionRequest): The parsed request body matching OpenAI schema.
        background_tasks (BackgroundTasks): FastAPI background task manager for accounting.
        client (AsyncOpenAI): Injected upstream client with JIT secret.
        redis_client (Redis): Injected Redis client.
        x_coreason_project_id (str): The project ID from the header.
        token (str): The verified gateway access token.
        _budget (None): Dependency trigger for budget validation.
        x_coreason_trace_id (str | None): Optional trace ID for distributed tracing.

    Returns:
        Any: A ChatCompletion response or a StreamingResponse (SSE).
    """
    settings = get_settings()

    context = {}
    if x_coreason_trace_id:
        context["trace_id"] = x_coreason_trace_id

    with logger.contextualize(**context):
        # Upstream Execution (with Retry)
        # Note: Client instantiation and Vault retrieval are now handled by `client` dependency.
        # Budget check is handled by `_budget` dependency.

        # We rely on FastAPI dependency teardown to close the client.
        # However, for streaming response, we need to ensure the client stays open
        # until the stream is consumed. FastAPI dependency teardown happens AFTER
        # the response is sent. So yielding the client in dependency is correct.

        try:
            # Prepare arguments
            kwargs = body.model_dump(exclude_unset=True)

            async for attempt in AsyncRetrying(
                stop=(
                    stop_after_attempt(settings.RETRY_STOP_AFTER_ATTEMPT)
                    | stop_after_delay(settings.RETRY_STOP_AFTER_DELAY)
                ),
                wait=wait_exponential(multiplier=1, min=settings.RETRY_WAIT_MIN, max=settings.RETRY_WAIT_MAX),
                retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
                reraise=True,
            ):
                with attempt:
                    response = await client.chat.completions.create(**kwargs)

        except Exception as e:
            # Exceptions are handled by global handlers or retried.
            # Client closing is handled by dependency teardown.
            raise e

        # Handle Response
        if body.stream:

            async def stream_generator() -> AsyncIterator[str]:
                # Re-apply logger context for the stream generator duration
                with logger.contextualize(**context):
                    usage = None
                    try:
                        # response is an AsyncStream
                        async for chunk in response:
                            # Capture usage if available (OpenAI stream_options)
                            if hasattr(chunk, "usage") and chunk.usage:
                                usage = chunk.usage

                            yield f"data: {chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        # Client close handled by dependency teardown
                        if usage:
                            await record_usage(x_coreason_project_id, usage, redis_client, trace_id=x_coreason_trace_id)

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # response is ChatCompletion
            # Client close handled by dependency teardown
            background_tasks.add_task(
                record_usage,
                x_coreason_project_id,
                response.usage,
                redis_client,
                trace_id=x_coreason_trace_id,
            )
            return response
