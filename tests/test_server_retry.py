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
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from coreason_ai_gateway.server import app


@pytest.fixture(autouse=True)
def setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
    monkeypatch.setenv("VAULT_ROLE_ID", "dummy-role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "dummy-secret-id")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    monkeypatch.setenv("GATEWAY_ACCESS_TOKEN", "valid-token")
    monkeypatch.setenv("RETRY_STOP_AFTER_ATTEMPT", "5")
    monkeypatch.setenv("RETRY_STOP_AFTER_DELAY", "2")  # Short delay for testing


@pytest.fixture
def mock_dependencies() -> Generator[dict[str, Any], None, None]:
    with (
        patch("coreason_ai_gateway.server.redis.from_url") as mock_redis,
        patch("coreason_ai_gateway.server.VaultManagerAsync") as mock_vault,
        patch("coreason_ai_gateway.server.CoreasonVaultConfig"),
        patch("coreason_ai_gateway.dependencies.AsyncOpenAI") as mock_openai,
    ):
        # Redis setup
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

        # Vault setup
        vault_instance = AsyncMock()
        mock_vault.return_value = vault_instance
        vault_instance.authenticate = AsyncMock()
        vault_instance.get_secret.return_value = {"api_key": "sk-test"}

        # OpenAI setup
        openai_client = AsyncMock()
        mock_openai.return_value = openai_client

        yield {"redis": redis_instance, "vault": vault_instance, "openai": mock_openai, "client": openai_client}


def test_retry_stops_after_delay(mock_dependencies: dict[str, Any]) -> None:
    mock_dependencies["client"].chat.completions.create.side_effect = RateLimitError(
        message="Rate limit", response=MagicMock(), body={}
    )

    with TestClient(app) as client:
        import time

        start = time.time()
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        end = time.time()

        assert response.status_code == 429
        duration = end - start
        assert duration >= 2.0


def test_retry_stops_after_attempts_if_faster(
    mock_dependencies: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETRY_STOP_AFTER_ATTEMPT", "2")
    monkeypatch.setenv("RETRY_STOP_AFTER_DELAY", "100")

    mock_dependencies["client"].chat.completions.create.side_effect = RateLimitError(
        message="Rate limit", response=MagicMock(), body={}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        assert response.status_code == 429
        assert mock_dependencies["client"].chat.completions.create.call_count == 2


def test_no_retry_on_non_retriable_error(mock_dependencies: dict[str, Any]) -> None:
    # AuthenticationError should not trigger retry
    mock_dependencies["client"].chat.completions.create.side_effect = AuthenticationError(
        message="Auth failed", response=MagicMock(), body={}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        # Should be 502 per exception handlers (AuthenticationError -> upstream_authentication_handler -> 502)
        assert response.status_code == 502
        assert mock_dependencies["client"].chat.completions.create.call_count == 1


def test_no_retry_on_bad_request(mock_dependencies: dict[str, Any]) -> None:
    # BadRequestError should not trigger retry
    mock_dependencies["client"].chat.completions.create.side_effect = BadRequestError(
        message="Bad Request", response=MagicMock(), body={}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        # Should be 400 per exception handlers
        assert response.status_code == 400
        assert mock_dependencies["client"].chat.completions.create.call_count == 1


def test_mixed_failures_eventually_succeed(mock_dependencies: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # Allow enough time for retries (default fixture has 2s delay, min wait is 2s)
    monkeypatch.setenv("RETRY_STOP_AFTER_DELAY", "20")

    # First call: RateLimit (should retry)
    # Second call: InternalServerError (should retry)
    # Third call: Success
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.model_dump.return_value = {"id": "123", "choices": []}

    mock_dependencies["client"].chat.completions.create.side_effect = [
        RateLimitError(message="Rate limit", response=MagicMock(), body={}),
        InternalServerError(message="Server Error", response=MagicMock(), body={}),
        mock_response,
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        assert response.status_code == 200
        assert mock_dependencies["client"].chat.completions.create.call_count == 3


def test_mixed_failures_exceed_attempts(mock_dependencies: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # Set attempts to 2
    monkeypatch.setenv("RETRY_STOP_AFTER_ATTEMPT", "2")

    # Fail twice with retriable errors
    mock_dependencies["client"].chat.completions.create.side_effect = [
        APIConnectionError(message="Conn Error", request=MagicMock()),
        InternalServerError(message="Server Error", response=MagicMock(), body={}),
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        # Should fail with 502 (mapped from InternalServerError)
        assert response.status_code == 502
        assert mock_dependencies["client"].chat.completions.create.call_count == 2
