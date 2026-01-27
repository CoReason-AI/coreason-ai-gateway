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
from fastapi.testclient import TestClient
from openai import AuthenticationError, BadRequestError

from coreason_ai_gateway.server import app


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
        patch("coreason_ai_gateway.server.CoreasonVaultConfig"),
        patch("coreason_ai_gateway.dependencies.AsyncOpenAI") as mock_openai,
    ):
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance
        redis_instance.get.return_value = "1000"

        redis_instance.pipeline = MagicMock()
        pipeline_mock = MagicMock()
        redis_instance.pipeline.return_value = pipeline_mock

        async def aenter(*args: Any, **kwargs: Any) -> MagicMock:
            return pipeline_mock

        pipeline_mock.__aenter__ = AsyncMock(side_effect=aenter)
        pipeline_mock.__aexit__ = AsyncMock()
        pipeline_mock.execute = AsyncMock()

        vault_instance = AsyncMock()
        mock_vault.return_value = vault_instance
        vault_instance.authenticate = AsyncMock()
        vault_instance.get_secret.return_value = {"api_key": "sk-test"}

        openai_client = AsyncMock()
        mock_openai.return_value = openai_client

        yield {"redis": redis_instance, "vault": vault_instance, "openai": mock_openai, "client": openai_client}


@pytest.fixture
def client(mock_dependencies: dict[str, Any]) -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


def test_upstream_bad_request(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Simulate Upstream 400 (e.g. Context Limit Exceeded)
    mock_dependencies["client"].chat.completions.create.side_effect = BadRequestError(
        message="Context length exceeded", response=MagicMock(), body={}
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 400
    assert "Upstream provider rejected request" in response.json()["detail"]


def test_upstream_authentication_error(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Simulate Upstream 401 (Gateway used bad key)
    mock_dependencies["client"].chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key", response=MagicMock(), body={}
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    # Should return 502, not 401
    assert response.status_code == 502
    assert "Upstream authentication failed" in response.json()["detail"]


def test_empty_secret_key(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Vault returns structure but key is missing
    mock_dependencies["vault"].get_secret.return_value = {}  # Empty dict

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 503
    assert "Security subsystem unavailable" in response.json()["detail"]


def test_invalid_json_body(client: TestClient) -> None:
    # Malformed JSON (Request validation)
    response = client.post(
        "/v1/chat/completions",
        content="{ invalid json }",
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 422


def test_invalid_pydantic_schema(client: TestClient) -> None:
    # Valid JSON but missing required field 'messages'
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4"},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 422
