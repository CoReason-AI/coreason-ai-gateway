# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import CompletionUsage

from coreason_ai_gateway.middleware.accounting import record_usage


@pytest.mark.anyio
async def test_record_usage_success() -> None:
    mock_redis = MagicMock()  # Changed to MagicMock so pipeline() is not awaited
    # Mock the pipeline context manager
    mock_pipeline = AsyncMock()
    # decrby and incrby on a pipeline are synchronous (chainable)
    mock_pipeline.decrby = MagicMock()
    mock_pipeline.incrby = MagicMock()

    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.__aenter__.return_value = mock_pipeline
    mock_pipeline.__aexit__.return_value = None

    usage = CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    await record_usage("proj-123", usage, mock_redis)

    mock_redis.pipeline.assert_called_once()
    mock_pipeline.decrby.assert_called_once_with("budget:proj-123:remaining", 30)
    mock_pipeline.incrby.assert_called_once_with("usage:proj-123:total", 30)
    mock_pipeline.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_record_usage_no_usage() -> None:
    mock_redis = MagicMock()

    await record_usage("proj-123", None, mock_redis)

    mock_redis.pipeline.assert_not_called()


@pytest.mark.anyio
async def test_record_usage_zero_tokens() -> None:
    mock_redis = MagicMock()
    usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    await record_usage("proj-123", usage, mock_redis)

    mock_redis.pipeline.assert_not_called()


@pytest.mark.anyio
async def test_record_usage_redis_failure() -> None:
    mock_redis = MagicMock()
    mock_pipeline = AsyncMock()
    # decrby and incrby on a pipeline are synchronous (chainable)
    mock_pipeline.decrby = MagicMock()
    mock_pipeline.incrby = MagicMock()

    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.__aenter__.return_value = mock_pipeline
    mock_pipeline.__aexit__.return_value = None

    mock_pipeline.execute.side_effect = Exception("Redis down")

    usage = CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    # Should not raise exception (it logs it)
    await record_usage("proj-123", usage, mock_redis)

    mock_pipeline.execute.assert_awaited_once()
