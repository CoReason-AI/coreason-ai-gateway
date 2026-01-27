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
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import CompletionUsage
from redis.exceptions import ConnectionError, RedisError

from coreason_ai_gateway.middleware.accounting import record_usage


@pytest.fixture
def mock_redis() -> MagicMock:
    mock_redis = MagicMock()
    # Mock the pipeline context manager
    mock_pipeline = AsyncMock()
    # decrby and incrby on a pipeline are synchronous (chainable)
    mock_pipeline.decrby = MagicMock()
    mock_pipeline.incrby = MagicMock()

    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.__aenter__.return_value = mock_pipeline
    mock_pipeline.__aexit__.return_value = None
    return mock_redis


@pytest.mark.anyio
async def test_record_usage_success(mock_redis: MagicMock) -> None:
    usage = CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    await record_usage("proj-123", usage, mock_redis)

    mock_redis.pipeline.assert_called_once()
    mock_pipeline = mock_redis.pipeline.return_value.__aenter__.return_value
    mock_pipeline.decrby.assert_called_once_with("budget:proj-123:remaining", 30)
    mock_pipeline.incrby.assert_called_once_with("usage:proj-123:total", 30)
    mock_pipeline.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_record_usage_no_usage(mock_redis: MagicMock) -> None:
    await record_usage("proj-123", None, mock_redis)
    mock_redis.pipeline.assert_not_called()


@pytest.mark.anyio
async def test_record_usage_zero_tokens(mock_redis: MagicMock) -> None:
    usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    await record_usage("proj-123", usage, mock_redis)
    mock_redis.pipeline.assert_not_called()


@pytest.mark.anyio
async def test_record_usage_redis_failure(mock_redis: MagicMock) -> None:
    mock_pipeline = mock_redis.pipeline.return_value.__aenter__.return_value
    mock_pipeline.execute.side_effect = Exception("Redis down")

    usage = CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    # Should not raise exception (it logs it)
    await record_usage("proj-123", usage, mock_redis)

    mock_pipeline.execute.assert_awaited_once()


# --- Edge Cases & Complex Scenarios ---


@pytest.mark.anyio
async def test_record_usage_negative_tokens(mock_redis: MagicMock) -> None:
    """Test that negative token counts (invalid state) are ignored."""
    usage = CompletionUsage(completion_tokens=-5, prompt_tokens=0, total_tokens=-5)

    await record_usage("proj-123", usage, mock_redis)

    mock_redis.pipeline.assert_not_called()


@pytest.mark.anyio
async def test_record_usage_large_tokens(mock_redis: MagicMock) -> None:
    """Test handling of very large integer values."""
    large_val = 2**60  # Large integer
    usage = CompletionUsage(completion_tokens=large_val, prompt_tokens=0, total_tokens=large_val)

    await record_usage("proj-123", usage, mock_redis)

    mock_pipeline = mock_redis.pipeline.return_value.__aenter__.return_value
    mock_pipeline.decrby.assert_called_once_with("budget:proj-123:remaining", large_val)
    mock_pipeline.incrby.assert_called_once_with("usage:proj-123:total", large_val)


@pytest.mark.anyio
async def test_record_usage_complex_project_id(mock_redis: MagicMock) -> None:
    """Test proper key construction with complex project IDs."""
    complex_id = "group:subgroup/user@example.com"
    usage = CompletionUsage(completion_tokens=10, prompt_tokens=10, total_tokens=20)

    await record_usage(complex_id, usage, mock_redis)

    mock_pipeline = mock_redis.pipeline.return_value.__aenter__.return_value
    mock_pipeline.decrby.assert_called_with(f"budget:{complex_id}:remaining", 20)
    mock_pipeline.incrby.assert_called_with(f"usage:{complex_id}:total", 20)


@pytest.mark.anyio
async def test_record_usage_concurrency(mock_redis: MagicMock) -> None:
    """Test concurrent execution of multiple usage records."""
    usage = CompletionUsage(completion_tokens=5, prompt_tokens=5, total_tokens=10)

    # Run 5 concurrent calls
    tasks = [record_usage(f"proj-{i}", usage, mock_redis) for i in range(5)]
    await asyncio.gather(*tasks)

    assert mock_redis.pipeline.call_count == 5
    # Verify each pipeline was executed
    # Note: Since we reuse the same mock object for all calls, we just verify count
    mock_pipeline = mock_redis.pipeline.return_value.__aenter__.return_value
    assert mock_pipeline.execute.await_count == 5


@pytest.mark.anyio
async def test_record_usage_pipeline_creation_error(mock_redis: MagicMock) -> None:
    """Test handling when redis.pipeline() raises a synchronous error."""
    mock_redis.pipeline.side_effect = RedisError("Pipeline creation failed")
    usage = CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    # Should catch and log
    await record_usage("proj-fail", usage, mock_redis)

    mock_redis.pipeline.assert_called_once()


@pytest.mark.anyio
async def test_record_usage_execute_connection_error(mock_redis: MagicMock) -> None:
    """Test handling when pipe.execute() raises a specific Redis ConnectionError."""
    mock_pipeline = mock_redis.pipeline.return_value.__aenter__.return_value
    mock_pipeline.execute.side_effect = ConnectionError("Connection lost")

    usage = CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    # Should catch and log
    await record_usage("proj-conn-fail", usage, mock_redis)

    mock_pipeline.execute.assert_awaited_once()
