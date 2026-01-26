# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Annotated

from coreason_vault import VaultManagerAsync
from fastapi import Depends, Request
from redis.asyncio import Redis


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
RedisDep = Annotated[Redis, Depends(get_redis_client)]
VaultDep = Annotated[VaultManagerAsync, Depends(get_vault_client)]
