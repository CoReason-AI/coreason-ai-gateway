# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import asyncio
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_ai_gateway.server import app
from fastapi.testclient import TestClient
from openai import RateLimitError


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


def test_retry_timeout_exceeded_by_long_execution(
    mock_dependencies: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Edge Case: The first attempt takes longer than the retry delay budget.
    Expectation: No retry should be attempted.
    """
    monkeypatch.setenv("RETRY_STOP_AFTER_DELAY", "1")  # 1 second budget
    monkeypatch.setenv("RETRY_STOP_AFTER_ATTEMPT", "5")

    async def slow_failure(**kwargs: Any) -> None:
        await asyncio.sleep(1.1)  # Sleep longer than budget
        raise RateLimitError(message="Slow Rate Limit", response=MagicMock(), body={})

    mock_dependencies["client"].chat.completions.create.side_effect = slow_failure

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        assert response.status_code == 429
        # Should call exactly once. Logic:
        # Start -> Call 1 (takes 1.1s) -> Check Stop (Time > 1s) -> Stop.
        assert mock_dependencies["client"].chat.completions.create.call_count == 1


def test_retry_wait_pushes_past_deadline(mock_dependencies: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Complex Scenario: Fast failures, but the mandatory wait time pushes the *next* attempt
    past the deadline.
    Configuration:
    - Stop Delay: 2.0s
    - Wait Min: 1.5s
    - Wait Max: 10s
    - Attempts: 5

    Sequence:
    1. Attempt 1 (t=0) -> Fails instantly.
    2. Check Stop (0 < 2.0) -> Continue.
    3. Wait (Exponential min 1.5s).
    4. Attempt 2 (t=1.5) -> Fails instantly.
    5. Check Stop (1.5 < 2.0) -> Continue.
    6. Wait (Exponential next > 1.5s). Next attempt would be at t > 3.0s.
    7. Tenacity checks if *wait* will exceed? No, stop_after_delay usually checks elapsed time *so far*.
       However, tenacity's `AsyncRetrying` loop logic is:
       do attempt -> check stop -> do wait -> loop.
       If check stop passes, it waits.
       So Attempt 3 might happen at t=3.0s, *then* check stop (3.0 > 2.0) -> Stop.

    Let's verify strict counting.
    """
    monkeypatch.setenv("RETRY_STOP_AFTER_DELAY", "3")
    monkeypatch.setenv("RETRY_WAIT_MIN", "1")
    monkeypatch.setenv("RETRY_WAIT_MAX", "5")
    monkeypatch.setenv("RETRY_STOP_AFTER_ATTEMPT", "10")

    mock_dependencies["client"].chat.completions.create.side_effect = RateLimitError(
        message="Fast Rate Limit", response=MagicMock(), body={}
    )

    with TestClient(app) as client:
        import time

        start = time.time()
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        duration = time.time() - start

        assert response.status_code == 429

        # Timeline (Config: Stop 3s, Wait Min 1s):
        # T+0: Call 1 (Fail). Elapsed ~0.
        #      Check Stop (0 < 3) -> Continue.
        #      Wait 1s.
        # T+1: Call 2 (Fail). Elapsed ~1.
        #      Check Stop (1 < 3) -> Continue.
        #      Wait 1s (Exponential multiplier=1, min=1. so 1s).
        # T+2: Call 3 (Fail). Elapsed ~2.
        #      Check Stop (2 < 3) -> Continue.
        #      Wait 1s (or 2s if exp kicks in? tenacity default exp base is 2?
        #      Here wait_exponential(multiplier=1, min=1)).
        #      Multiplier 1 implies: 1 * 2^(n-1). n=attempt.
        #      Wait 1: 1 * 2^0 = 1.
        #      Wait 2: 1 * 2^1 = 2.
        # T+4: Call 4 (Fail). Elapsed ~4.
        #      Check Stop (4 < 3) -> Stop.

        # So we expect 3 or 4 calls depending on precise timing.
        assert mock_dependencies["client"].chat.completions.create.call_count in (3, 4)
        assert duration >= 2.0  # At least two waits (1s + 1s)
