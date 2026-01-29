# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_ai_gateway.server import app
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
    monkeypatch.setenv("VAULT_ROLE_ID", "dummy-role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "dummy-secret-id")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    monkeypatch.setenv("GATEWAY_ACCESS_TOKEN", "valid-token")


@pytest.fixture
def mock_dependencies() -> Generator[dict[str, Any], None, None]:
    with (
        patch("coreason_ai_gateway.server.redis.from_url") as mock_redis,
        patch("coreason_ai_gateway.server.VaultManagerAsync") as mock_vault,
        patch("coreason_ai_gateway.server.CoreasonVaultConfig") as mock_vault_config,
        patch("coreason_ai_gateway.service.AsyncOpenAI") as mock_openai,
    ):
        # Redis setup
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance
        redis_instance.get.return_value = "1000"  # Sufficient budget by default

        # Mock Pipeline
        # Use MagicMock for the pipeline object itself, but configure async methods explicitly.
        pipeline_mock = MagicMock()
        pipeline_mock.__aenter__ = AsyncMock(return_value=pipeline_mock)
        pipeline_mock.__aexit__ = AsyncMock(return_value=None)
        pipeline_mock.execute = AsyncMock()
        pipeline_mock.decrby = MagicMock()
        pipeline_mock.incrby = MagicMock()
        execute_mock = pipeline_mock.execute

        # Configure pipeline() to return the pipeline mock
        redis_instance.pipeline = MagicMock(return_value=pipeline_mock)

        # Vault setup
        vault_instance = AsyncMock()
        mock_vault.return_value = vault_instance
        # Default mock structure (nested auth)
        vault_instance.auth = AsyncMock()
        # authenticate_approle is handled internally by constructor logic now, but we keep mock for safety
        vault_instance.auth.authenticate_approle = AsyncMock()
        vault_instance.auth.close = AsyncMock()
        vault_instance.get_secret.return_value = {"api_key": "sk-test"}

        # OpenAI setup
        openai_client = AsyncMock()
        mock_openai.return_value = openai_client

        yield {
            "redis": redis_instance,
            "vault": vault_instance,
            "vault_config": mock_vault_config,
            "openai": mock_openai,
            "client": openai_client,
            "pipeline": pipeline_mock,
            "execute": execute_mock,
        }


@pytest.fixture
def client(mock_dependencies: dict[str, Any]) -> Generator[TestClient, None, None]:
    """
    Client fixture that ensures dependencies are mocked.
    """
    with TestClient(app) as c:
        yield c
