# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import json
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionChunk

from coreason_ai_gateway.server import app
from coreason_ai_gateway.utils.logger import logger


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
        patch("coreason_ai_gateway.server.AsyncOpenAI") as mock_openai,
    ):
        # Redis setup
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance
        redis_instance.get.return_value = "1000"

        # Mock Pipeline
        pipeline_mock = MagicMock()
        # redis.pipeline() is synchronous, so we must replace the AsyncMock's
        # default async method with a sync Mock that returns the pipeline object.
        redis_instance.pipeline = MagicMock(return_value=pipeline_mock)

        async def aenter(*args: Any, **kwargs: Any) -> MagicMock:
            return pipeline_mock

        pipeline_mock.__aenter__ = AsyncMock(side_effect=aenter)
        # Must return None to allow exceptions to propagate
        pipeline_mock.__aexit__ = AsyncMock(return_value=None)
        pipeline_mock.execute = AsyncMock()

        # Vault setup
        vault_instance = AsyncMock()
        mock_vault.return_value = vault_instance
        vault_instance.authenticate = AsyncMock()
        vault_instance.get_secret.return_value = {"api_key": "sk-test"}

        # OpenAI setup
        openai_client = AsyncMock()
        mock_openai.return_value = openai_client

        yield {
            "redis": redis_instance,
            "vault": vault_instance,
            "openai": mock_openai,
            "client": openai_client,
            "pipeline": pipeline_mock,
        }


@pytest.fixture
def log_capture() -> Generator[list[str], None, None]:
    logs: list[str] = []
    # Capture raw logs in serialized format
    handler_id = logger.add(lambda msg: logs.append(msg), format="{message}", level="INFO", serialize=True)
    yield logs
    logger.remove(handler_id)


def test_missing_trace_id(mock_dependencies: dict[str, Any], log_capture: list[str]) -> None:
    """Verify that requests without Trace ID header do not crash and logs don't have the key."""
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.model_dump.return_value = {"id": "123", "choices": []}
    mock_dependencies["client"].chat.completions.create.return_value = mock_response

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        assert response.status_code == 200

    # Ensure no log has 'trace_id' in extra
    for log_msg in log_capture:
        log_entry = json.loads(log_msg)
        extra = log_entry.get("record", {}).get("extra", {})
        assert "trace_id" not in extra, f"Found trace_id unexpectedly in: {log_entry}"


@pytest.mark.anyio
async def test_background_task_error_logging(mock_dependencies: dict[str, Any], log_capture: list[str]) -> None:
    """Verify that exceptions in background tasks (accounting) still carry the Trace ID."""
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.model_dump.return_value = {"id": "123", "choices": []}
    mock_dependencies["client"].chat.completions.create.return_value = mock_response

    # Force Redis failure during accounting
    mock_dependencies["pipeline"].execute.side_effect = Exception("Redis Connection Failed")

    trace_id = "trace-error-123"

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={
                "Authorization": "Bearer valid-token",
                "X-Coreason-Project-ID": "proj-1",
                "X-Coreason-Trace-ID": trace_id,
            },
        )

    # Look for the exception log
    found_error_log = False
    for log_msg in log_capture:
        log_entry = json.loads(log_msg)
        record = log_entry.get("record", {})
        extra = record.get("extra", {})
        message = record.get("message", "")

        if "Failed to record usage" in message and record.get("exception"):
            # This is the target log
            if extra.get("trace_id") == trace_id:
                found_error_log = True
                break

    assert found_error_log, "Exception log for background task did not contain correct Trace ID"


@pytest.mark.anyio
async def test_streaming_context_preservation(mock_dependencies: dict[str, Any], log_capture: list[str]) -> None:
    """Verify that logic inside the stream generator (simulated by logging) has the Trace ID."""
    trace_id = "trace-stream-456"

    # Define a generator that logs explicitly to verify context
    async def logging_generator(**kwargs: Any) -> AsyncGenerator[ChatCompletionChunk, None]:
        # Emulate a log that would happen deep in the stack
        logger.info("Inside stream generator")

        chunk = MagicMock(spec=ChatCompletionChunk)
        chunk.model_dump_json.return_value = '{"id": "1", "choices": []}'
        chunk.usage = None
        yield chunk

    mock_dependencies["client"].chat.completions.create.side_effect = logging_generator

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            headers={
                "Authorization": "Bearer valid-token",
                "X-Coreason-Project-ID": "proj-1",
                "X-Coreason-Trace-ID": trace_id,
            },
        )
        assert response.status_code == 200
        # Consume stream
        for _ in response.iter_lines():
            pass

    # Verify the specific log message has the trace ID
    found_stream_log = False
    for log_msg in log_capture:
        log_entry = json.loads(log_msg)
        record = log_entry.get("record", {})
        if record.get("message") == "Inside stream generator":
            extra = record.get("extra", {})
            if extra.get("trace_id") == trace_id:
                found_stream_log = True
                break

    assert found_stream_log, "Log inside stream generator did not contain Trace ID. Context was lost."
