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
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_ai_gateway.server import app
from fastapi.testclient import TestClient

# Set environment variables for config
os.environ["VAULT_ADDR"] = "http://vault:8200"
os.environ["VAULT_ROLE_ID"] = "dummy-role-id"
os.environ["VAULT_SECRET_ID"] = "dummy-secret-id"
os.environ["REDIS_URL"] = "redis://redis:6379"
os.environ["GATEWAY_ACCESS_TOKEN"] = "dummy-token"


@pytest.fixture
def mock_redis_patch() -> Generator[MagicMock, None, None]:
    with patch("coreason_ai_gateway.server.redis.from_url") as mock:
        redis_instance = AsyncMock()
        mock.return_value = redis_instance
        yield mock


@pytest.fixture
def mock_vault_patch() -> Generator[MagicMock, None, None]:
    with patch("coreason_ai_gateway.server.VaultManagerAsync") as mock:
        vault_instance = AsyncMock()
        mock.return_value = vault_instance
        yield mock


@pytest.fixture
def mock_vault_config() -> Generator[MagicMock, None, None]:
    with patch("coreason_ai_gateway.server.CoreasonVaultConfig") as mock:
        yield mock


def test_health_check(mock_redis_patch: MagicMock, mock_vault_patch: MagicMock, mock_vault_config: MagicMock) -> None:
    # TestClient triggers lifespan
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_lifespan(mock_redis_patch: MagicMock, mock_vault_patch: MagicMock, mock_vault_config: MagicMock) -> None:
    # Trigger lifespan
    async with app.router.lifespan_context(app):
        # Assert Redis initialized
        assert app.state.redis is mock_redis_patch.return_value

        # Assert Vault initialized
        # Config should be created with role_id/secret_id
        # Note: Pydantic Settings/AnyUrl might add a trailing slash to the URL if path is empty.
        # We check mostly for correct structure.
        mock_vault_config.assert_called_once()
        call_kwargs = mock_vault_config.call_args.kwargs
        assert str(call_kwargs["VAULT_ADDR"]).rstrip("/") == "http://vault:8200"
        assert call_kwargs["VAULT_ROLE_ID"] == "dummy-role-id"
        assert call_kwargs["VAULT_SECRET_ID"] == "dummy-secret-id"

        assert app.state.vault is mock_vault_patch.return_value
        # No explicit auth call should be made
        mock_vault_patch.return_value.auth.authenticate_approle.assert_not_called()

    # Assert teardown
    mock_redis_patch.return_value.close.assert_awaited_once()
    mock_vault_patch.return_value.auth.close.assert_awaited_once()


@pytest.mark.anyio
async def test_lifespan_redis_failure(mock_redis_patch: MagicMock) -> None:
    # Mock redis raising exception
    mock_redis_patch.side_effect = Exception("Redis connection failed")

    with pytest.raises(Exception, match="Redis connection failed"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.anyio
async def test_lifespan_vault_failure(
    mock_redis_patch: MagicMock, mock_vault_patch: MagicMock, mock_vault_config: MagicMock
) -> None:
    # Mock vault init raising exception (e.g. config issue or internal setup)
    mock_vault_patch.side_effect = Exception("Vault init failed")

    with pytest.raises(Exception, match="Vault init failed"):
        async with app.router.lifespan_context(app):
            pass

    # Ensure Redis closed even if Vault fails
    mock_redis_patch.return_value.close.assert_awaited_once()


@pytest.mark.anyio
async def test_teardown_resilience(
    mock_redis_patch: MagicMock, mock_vault_patch: MagicMock, mock_vault_config: MagicMock
) -> None:
    # Mock Redis close failure
    mock_redis_patch.return_value.close.side_effect = Exception("Redis close error")
    mock_vault_patch.return_value.auth.close.side_effect = Exception("Vault close error")

    # The exception should be logged but not raised out of lifespan teardown (or at least Vault should close)
    # Since it's in a generator finally block (effectively), if we swallow exception, it's fine.
    # The updated code catches Exception, logs it, and continues.

    async with app.router.lifespan_context(app):
        pass

    # Verify Redis close was called (and failed)
    mock_redis_patch.return_value.close.assert_awaited_once()

    # Verify Vault close was ALSO called despite Redis failure
    mock_vault_patch.return_value.auth.close.assert_awaited_once()


@pytest.mark.anyio
async def test_full_lifecycle_integration(
    mock_redis_patch: MagicMock, mock_vault_patch: MagicMock, mock_vault_config: MagicMock
) -> None:
    """
    Complex redundant test:
    Manually enter lifespan context, check state, simulate a health request manually
    (or via a client inside the context), and exit.
    This redundantly validates the whole flow without relying on TestClient's auto-lifespan only.
    """
    async with app.router.lifespan_context(app):
        # 1. Check State
        assert app.state.redis is mock_redis_patch.return_value
        assert app.state.vault is mock_vault_patch.return_value

        # 2. Simulate Request Logic (Direct call to endpoint to verify it works with state)
        # Note: health_check doesn't use state, but we can verify state is present on app
        from coreason_ai_gateway.server import health_check

        res = await health_check()
        assert res == {"status": "ok"}

        # 3. Simulate Logic that might use Redis (just verifying the mock is accessible)
        await app.state.redis.ping()
        mock_redis_patch.return_value.ping.assert_awaited_once()

    # 4. Teardown
    mock_redis_patch.return_value.close.assert_awaited_once()
    mock_vault_patch.return_value.auth.close.assert_awaited_once()


# Test main.py
def test_main() -> None:
    from coreason_ai_gateway.main import main

    with patch("uvicorn.run") as mock_run:
        main()
        mock_run.assert_called_once()
        # Verify call arguments
        args, kwargs = mock_run.call_args
        assert args[0] == app
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 8000


def test_main_execution_failure() -> None:
    """Verify that main propagates exceptions from uvicorn.run."""
    from coreason_ai_gateway.main import main

    with patch("uvicorn.run") as mock_run:
        # Simulate uvicorn crashing (e.g. port in use)
        mock_run.side_effect = OSError("Port already in use")

        with pytest.raises(OSError, match="Port already in use"):
            main()
