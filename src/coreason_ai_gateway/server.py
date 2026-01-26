# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from coreason_vault import CoreasonVaultConfig, VaultManagerAsync
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from redis import asyncio as redis
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from .config import get_settings
from .dependencies import RedisDep, VaultDep
from .exception_handlers import register_exception_handlers
from .middleware.accounting import record_usage
from .middleware.auth import verify_gateway_token
from .middleware.budget import check_budget, estimate_tokens
from .routing import resolve_provider_path
from .schemas import ChatCompletionRequest
from .utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting up Coreason AI Gateway...")

    # 1. Setup Redis
    try:
        app.state.redis = redis.from_url(str(settings.REDIS_URL), encoding="utf-8", decode_responses=True)
        logger.info("Redis client initialized.")
    except Exception as e:
        logger.exception("Failed to initialize Redis client")
        raise e

    # 2. Setup Vault
    try:
        vault_config = CoreasonVaultConfig(VAULT_ADDR=str(settings.VAULT_ADDR))
        app.state.vault = VaultManagerAsync(config=vault_config)
        await app.state.vault.authenticate(
            role_id=settings.VAULT_ROLE_ID,
            secret_id=settings.VAULT_SECRET_ID.get_secret_value(),
        )
        logger.info("Vault client authenticated.")
    except Exception as e:
        logger.exception("Failed to initialize Vault client")
        # Ensure redis is closed if vault fails
        if hasattr(app.state, "redis"):
            await app.state.redis.close()
        raise e

    yield

    # 3. Teardown
    logger.info("Shutting down Coreason AI Gateway...")
    if hasattr(app.state, "redis"):
        try:
            await app.state.redis.close()
            logger.info("Redis connection closed.")
        except Exception:
            logger.exception("Failed to close Redis connection")

    if hasattr(app.state, "vault"):
        try:
            await app.state.vault.close()
            logger.info("Vault connection closed.")
        except Exception:
            logger.exception("Failed to close Vault connection")


app = FastAPI(title="Coreason AI Gateway", lifespan=lifespan)
register_exception_handlers(app)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint to verify service status.
    """
    return {"status": "ok"}


@app.post("/v1/chat/completions", status_code=200)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    redis_client: RedisDep,  # type: ignore[type-arg]
    vault_client: VaultDep,
    token: Annotated[str, Depends(verify_gateway_token)],
    x_coreason_project_id: Annotated[str, Header()],
    x_coreason_trace_id: Annotated[str | None, Header()] = None,
) -> Any:
    """
    Proxies chat completion requests to LLM providers.
    Enforces authentication, budgeting, and JIT secret injection.
    """
    settings = get_settings()

    # 1. Budget Check (Fail Fast)
    estimated_tokens = estimate_tokens(body.messages)
    await check_budget(x_coreason_project_id, estimated_tokens, redis_client)

    # 2. Secret Retrieval (JIT)
    provider_path = resolve_provider_path(body.model)
    try:
        # According to TRD: secret/infrastructure/{provider}
        secret_path = f"secret/{provider_path}"
        secret_data = await vault_client.get_secret(secret_path)
    except Exception as e:
        logger.exception(f"Vault secret retrieval failed for {provider_path}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Security subsystem unavailable",
        ) from e

    if not secret_data or "api_key" not in secret_data:
        logger.error(f"Invalid secret structure for {provider_path}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Security subsystem unavailable",
        )

    api_key = secret_data["api_key"]

    # 3. Client Instantiation & Upstream Execution
    client = AsyncOpenAI(api_key=api_key)

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
        # Exceptions are now handled by global exception handlers.
        # However, we must ensure client is closed.
        await client.close()
        # If it's not one of our handled types, we might want to log it or let it bubble up.
        # But if we let it bubble up, FastAPI catches it as 500 if unhandled.
        # Our exception handlers cover OpenAI errors. Generic Exception is not covered.
        # So we should probably re-raise or handle as 500 explicitly if we want custom logging.
        # But we also have retry logic which catches exceptions.
        # Wait, tenacity re-raises.
        raise e

    # 4. Handle Response
    if body.stream:

        async def stream_generator() -> AsyncIterator[str]:
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
                await client.close()
                if usage:
                    await record_usage(x_coreason_project_id, usage, redis_client)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        # response is ChatCompletion
        await client.close()
        background_tasks.add_task(record_usage, x_coreason_project_id, response.usage, redis_client)
        return response
