# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import TYPE_CHECKING, Annotated, Any

from coreason_vault import VaultManagerAsync
from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis

from coreason_ai_gateway.middleware.budget import check_budget, estimate_tokens
from coreason_ai_gateway.routing import resolve_provider_path
from coreason_ai_gateway.schemas import ChatCompletionRequest
from coreason_ai_gateway.service import ServiceAsync
from coreason_ai_gateway.utils.logger import logger

"""
FastAPI dependencies for dependency injection.
Handles database connections, authentication, and service clients.
"""


def get_redis_client(request: Request) -> Redis:  # type: ignore[type-arg]
    """
    Dependency to retrieve the Redis client from app state.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        Redis: The initialized Redis client from app.state.

    Raises:
        RuntimeError: If the Redis client is not initialized in app state.
    """
    if not hasattr(request.app.state, "redis"):
        raise RuntimeError("Redis client is not initialized in app state")
    return request.app.state.redis  # type: ignore[no-any-return]


def get_vault_client(request: Request) -> VaultManagerAsync:
    """
    Dependency to retrieve the Vault client from app state.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        VaultManagerAsync: The initialized Vault client from app.state.

    Raises:
        RuntimeError: If the Vault client is not initialized in app state.
    """
    if not hasattr(request.app.state, "vault"):
        raise RuntimeError("Vault client is not initialized in app state")
    return request.app.state.vault


def get_service(request: Request) -> ServiceAsync:
    """
    Dependency to retrieve the ServiceAsync from app state.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        ServiceAsync: The initialized Service from app.state.

    Raises:
        RuntimeError: If the Service is not initialized in app state.
    """
    if not hasattr(request.app.state, "service"):
        raise RuntimeError("Service is not initialized in app state")
    return request.app.state.service  # type: ignore[no-any-return]


# Type aliases for use in endpoints
if TYPE_CHECKING:
    RedisType = Redis[Any]
else:
    RedisType = Redis

# We cannot easily suppress mypy "unused ignore" because it varies by environment/version.
# However, the conditional typing above is robust.
# But Wait, at runtime RedisType is Redis. Annotated[Redis, ...] might still trigger mypy if it thinks Redis is generic.
# Actually, if we use the alias `RedisType`, mypy sees it as `Redis[Any]`.
# Runtime sees it as `Redis`.
# This avoids the TypeError and satisfies mypy without ignores.
RedisDep = Annotated[RedisType, Depends(get_redis_client)]
VaultDep = Annotated[VaultManagerAsync, Depends(get_vault_client)]


async def validate_request_budget(
    request: Request,
    body: ChatCompletionRequest,
    redis_client: RedisDep,
) -> None:
    """
    Dependency that enforces budget limits for the incoming request.
    Calculates estimated cost and rejects if budget is insufficient.

    Args:
        request (Request): The incoming request (for accessing user context).
        body (ChatCompletionRequest): The parsed request body.
        redis_client (RedisDep): The injected Redis client.

    Returns:
        None

    Raises:
        HTTPException: 402 Payment Required if budget is insufficient.
        HTTPException: 500 if UserContext is missing.
    """
    if not hasattr(request.state, "user_context"):
        # Should be caught by middleware, but defensive check
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User Context Missing",
        )

    context = request.state.user_context
    estimated_tokens = estimate_tokens(body.messages)
    await check_budget(context, estimated_tokens, redis_client)


async def get_upstream_api_key(
    body: ChatCompletionRequest,
    vault_client: VaultDep,
) -> str:
    """
    Dependency that retrieves the API Key for the upstream provider.
    Handles Just-In-Time secret retrieval.

    Args:
        body (ChatCompletionRequest): The parsed request body.
        vault_client (VaultDep): The injected Vault client.

    Returns:
        str: The API key.

    Raises:
        HTTPException: 503 Service Unavailable if secret retrieval fails or structure is invalid.
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

    return str(secret_data["api_key"])
