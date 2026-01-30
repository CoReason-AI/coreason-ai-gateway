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
from unittest.mock import MagicMock

import pytest
from coreason_identity.models import UserContext
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionChunk

from coreason_ai_gateway.middleware.accounting import record_usage
from coreason_ai_gateway.server import app
from coreason_ai_gateway.utils.logger import logger


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
    mock_dependencies["execute"].side_effect = Exception("Redis Connection Failed")

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


@pytest.mark.anyio
async def test_accounting_pipeline_integrity(mock_dependencies: dict[str, Any]) -> None:
    """
    Verify that record_usage correctly queues commands on the pipeline
    and executes them. This ensures mocks are wired correctly and logic is sound.
    """
    project_id = "proj-integrity"
    usage = MagicMock()
    usage.total_tokens = 42

    pipeline_mock = mock_dependencies["pipeline"]
    redis_client = mock_dependencies["redis"]

    # Manually invoke record_usage to verify pipeline interaction
    context = UserContext(sub=project_id, email="test@example.com")
    await record_usage(context, usage, redis_client, trace_id="trace-integrity")

    # Verify pipeline was created
    redis_client.pipeline.assert_called_once()

    # Verify commands
    pipeline_mock.decrby.assert_called_once_with(f"budget:{project_id}:remaining", 42)
    pipeline_mock.incrby.assert_called_once_with(f"usage:{project_id}:total", 42)

    # Verify execute
    pipeline_mock.execute.assert_awaited_once()
