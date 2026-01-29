# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import os
from unittest.mock import AsyncMock, patch

import pytest
from coreason_ai_gateway.server import lifespan
from fastapi import FastAPI

# Set environment variables for config (required for get_settings call inside lifespan)
os.environ["VAULT_ADDR"] = "http://vault:8200"
os.environ["VAULT_ROLE_ID"] = "dummy-role-id"
os.environ["VAULT_SECRET_ID"] = "dummy-secret-id"
os.environ["REDIS_URL"] = "redis://redis:6379"
os.environ["GATEWAY_ACCESS_TOKEN"] = "dummy-token"


@pytest.mark.anyio
async def test_lifespan_vault_init_failure() -> None:
    """
    Verify that if VaultManagerAsync initialization fails (e.g. constructor error),
    the previously opened Redis connection is still closed to prevent leaks.
    """
    app = FastAPI()
    with (
        patch("coreason_ai_gateway.server.redis.from_url") as mock_redis,
        patch("coreason_ai_gateway.server.VaultManagerAsync") as mock_vault_cls,
        patch("coreason_ai_gateway.server.CoreasonVaultConfig"),
    ):
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance

        # Simulate VaultManagerAsync constructor failure
        mock_vault_cls.side_effect = TypeError("Init failed")

        with pytest.raises(TypeError, match="Init failed"):
            async with lifespan(app):
                pass

        # Verify Redis was closed
        redis_instance.close.assert_awaited_once()


@pytest.mark.anyio
async def test_lifespan_config_failure() -> None:
    """
    Verify that if CoreasonVaultConfig initialization fails,
    the previously opened Redis connection is closed.
    """
    app = FastAPI()
    with (
        patch("coreason_ai_gateway.server.redis.from_url") as mock_redis,
        patch("coreason_ai_gateway.server.CoreasonVaultConfig") as mock_config,
    ):
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance

        # Simulate Config failure
        mock_config.side_effect = ValueError("Bad Config")

        with pytest.raises(ValueError, match="Bad Config"):
            async with lifespan(app):
                pass

        # Verify Redis was closed
        redis_instance.close.assert_awaited_once()


@pytest.mark.anyio
async def test_lifespan_teardown_exceptions() -> None:
    """
    Verify that exceptions during teardown (closing Service/Vault/Redis) are caught and logged,
    ensuring cleanup continues or at least doesn't crash the shutdown process.
    """
    app = FastAPI()
    with (
        patch("coreason_ai_gateway.server.redis.from_url") as mock_redis,
        patch("coreason_ai_gateway.server.VaultManagerAsync") as mock_vault_cls,
        patch("coreason_ai_gateway.server.CoreasonVaultConfig"),
        patch("coreason_ai_gateway.service.ServiceAsync") as mock_service_cls,
    ):
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance

        vault_instance = AsyncMock()
        mock_vault_cls.return_value = vault_instance

        service_instance = AsyncMock()
        mock_service_cls.return_value = service_instance

        # Configure exceptions during close
        redis_instance.close.side_effect = Exception("Redis Close Error")
        vault_instance.auth.close.side_effect = Exception("Vault Close Error")
        service_instance.__aexit__.side_effect = Exception("Service Close Error")

        # Should not raise exception
        async with lifespan(app):
            pass

        # Verify close was attempted
        redis_instance.close.assert_awaited()
        vault_instance.auth.close.assert_awaited()
        service_instance.__aexit__.assert_awaited()
