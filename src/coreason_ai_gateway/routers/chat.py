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
    HTTPException,
    Request,
    status,
)
from fastapi.responses import StreamingResponse

from coreason_ai_gateway.dependencies import RedisDep, get_service, get_upstream_api_key, validate_request_budget
from coreason_ai_gateway.middleware.accounting import record_usage
from coreason_ai_gateway.schemas import ChatCompletionRequest
from coreason_ai_gateway.service import ServiceAsync
from coreason_ai_gateway.utils.logger import logger

"""
Chat completion router.
Handles the main /v1/chat/completions endpoint.
"""


router = APIRouter()


@router.post("/v1/chat/completions", status_code=200)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    service: Annotated[ServiceAsync, Depends(get_service)],
    api_key: Annotated[str, Depends(get_upstream_api_key)],
    redis_client: RedisDep,
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
        service (ServiceAsync): Injected core service.
        api_key (str): Injected upstream API Key.
        redis_client (Redis): Injected Redis client.
        _budget (None): Dependency trigger for budget validation.
        x_coreason_trace_id (str | None): Optional trace ID for distributed tracing.

    Returns:
        Any: A ChatCompletion response or a StreamingResponse (SSE).
    """
    if not hasattr(request.state, "user_context"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User Context Missing",
        )
    user_context = request.state.user_context

    context = {}
    if x_coreason_trace_id:
        context["trace_id"] = x_coreason_trace_id

    # Add identity to logger context
    context["user_id"] = user_context.sub

    with logger.contextualize(**context):
        # Upstream Execution (via ServiceAsync)
        # Note: Client instantiation, Retry, and Provider interaction are handled by ServiceAsync.
        # Budget check is handled by `_budget` dependency.

        try:
            response = await service.chat_completions(body, api_key, user_context)

        except Exception as e:
            # Exceptions are handled by global handlers or retried in service.
            raise e

        # Handle Response
        if body.stream:

            async def stream_generator() -> AsyncIterator[str]:
                # Re-apply logger context for the stream generator duration
                with logger.contextualize(**context):
                    usage = None
                    try:
                        # response is an AsyncStream (AsyncIterator[ChatCompletionChunk])
                        async for chunk in response:
                            # Capture usage if available (OpenAI stream_options)
                            if hasattr(chunk, "usage") and chunk.usage:
                                usage = chunk.usage

                            yield f"data: {chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        if usage:
                            await record_usage(user_context, usage, redis_client, trace_id=x_coreason_trace_id)

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # response is ChatCompletion
            background_tasks.add_task(
                record_usage,
                user_context,
                response.usage,  # type: ignore
                redis_client,
                trace_id=x_coreason_trace_id,
            )
            return response
