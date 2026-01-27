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

from coreason_vault import VaultManagerAsync
from fastapi import Depends, Header, HTTPException, Request, status
from openai import AsyncOpenAI
from redis.asyncio import Redis

from coreason_ai_gateway.middleware.budget import check_budget, estimate_tokens
from coreason_ai_gateway.routing import resolve_provider_path
from coreason_ai_gateway.schemas import ChatCompletionRequest
from coreason_ai_gateway.utils.logger import logger


def get_redis_client(request: Request) -> Redis:  # type: ignore[type-arg]
    """
    Dependency to retrieve the Redis client from app state.
    """
    if not hasattr(request.app.state, "redis"):
        raise RuntimeError("Redis client is not initialized in app state")
    return request.app.state.redis  # type: ignore[no-any-return]


def get_vault_client(request: Request) -> VaultManagerAsync:
    """
    Dependency to retrieve the Vault client from app state.
    """
    if not hasattr(request.app.state, "vault"):
        raise RuntimeError("Vault client is not initialized in app state")
    return request.app.state.vault


# Type aliases for use in endpoints
# Redis[Any] causes runtime TypeError in some envs, so we use bare Redis and suppress mypy

RedisDep = Annotated[Redis[Any], Depends(get_redis_client)]
VaultDep = Annotated[VaultManagerAsync, Depends(get_vault_client)]


async def validate_request_budget(
    body: ChatCompletionRequest,
    redis_client: RedisDep,
    x_coreason_project_id: Annotated[str, Header()],
) -> None:
    """
    Dependency that enforces budget limits for the incoming request.
    Calculates estimated cost and rejects if budget is insufficient.
    """
    estimated_tokens = estimate_tokens(body.messages)
    await check_budget(x_coreason_project_id, estimated_tokens, redis_client)


async def get_upstream_client(
    body: ChatCompletionRequest,
    vault_client: VaultDep,
) -> AsyncIterator[AsyncOpenAI]:
    """
    Dependency that provides an authenticated AsyncOpenAI client.
    Handles Just-In-Time secret retrieval and client lifecycle.
    """
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

    # Client Instantiation
    # Disable internal retries to let tenacity handle resilience policies
    client = AsyncOpenAI(api_key=api_key, max_retries=0)
    try:
        yield client
    finally:
        await client.close()
