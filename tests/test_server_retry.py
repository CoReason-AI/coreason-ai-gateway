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
from openai import RateLimitError

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
        patch("coreason_ai_gateway.server.AsyncOpenAI") as mock_openai,
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
    # We set RETRY_STOP_AFTER_DELAY to 2 seconds.
    # We will mock the client.chat.completions.create to raise RateLimitError always.
    # And we will verify it stops.
    # To properly test the delay without waiting, we might need to mock time,
    # but tenacity uses time.monotonic.
    # A simpler integration-style test with short delay is acceptable here since it's "2 seconds".

    # However, to be faster and more deterministic, we can mock time.
    # But patching time.monotonic in tenacity is tricky across modules.
    # Since we set delay to 2s, we can just run it. It's a bit slow for a unit test but robust.

    mock_dependencies["client"].chat.completions.create.side_effect = RateLimitError(
        message="Rate limit", response=MagicMock(), body={}
    )

    with TestClient(app) as client:
        # Start timer
        import time

        start = time.time()
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        end = time.time()

        # It should return 429 (handled by exception handler) after retries are exhausted.
        # But wait, does it return 429?
        # If AsyncRetrying re-raises, it bubbles up.
        # Then exception handlers catch RateLimitError and return 429.
        assert response.status_code == 429
        duration = end - start
        # Should be at least 2 seconds (the delay config).
        # And it should not have tried too many times if the wait is exponential.
        assert duration >= 2.0


def test_retry_stops_after_attempts_if_faster(
    mock_dependencies: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Set delay to be long, but attempts to be short.
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
        # Should have called exactly 2 times (initial + 1 retry? Or 2 attempts total?)
        # stop_after_attempt(2) means 2 attempts total.
        assert mock_dependencies["client"].chat.completions.create.call_count == 2
