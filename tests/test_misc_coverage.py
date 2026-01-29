# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from openai.types import CompletionUsage
from coreason_identity.models import UserContext

from coreason_ai_gateway.middleware.accounting import record_usage
from coreason_ai_gateway.middleware.auth import verify_gateway_token
from coreason_ai_gateway.middleware.budget import check_budget, estimate_tokens
from coreason_ai_gateway.routing import resolve_provider_path


def test_routing_coverage() -> None:
    assert resolve_provider_path("claude-3-opus") == "infrastructure/anthropic"

    with pytest.raises(HTTPException) as exc:
        resolve_provider_path("unknown-model")
    assert exc.value.status_code == 400


@pytest.mark.anyio
async def test_budget_coverage() -> None:
    # estimate_tokens fallback
    with patch("json.dumps", side_effect=ValueError):
        assert estimate_tokens([{"role": "user", "content": "test"}]) > 0

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    context = UserContext(sub="proj1", email="test@example.com")

    # Missing budget key
    with pytest.raises(HTTPException) as exc:
        await check_budget(context, 100, mock_redis)
    assert exc.value.status_code == 402

    # Corrupted budget
    mock_redis.get.return_value = "not-int"
    with pytest.raises(HTTPException) as exc:
        await check_budget(context, 100, mock_redis)
    assert exc.value.status_code == 402


@pytest.mark.anyio
async def test_auth_coverage() -> None:
    # Missing header is handled by FastAPI Depends if not optional,
    # but verify_gateway_token takes Annotated[str | None, Header()] = None

    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token(None)
    assert exc.value.status_code == 401

    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token("Basic token")
    assert exc.value.status_code == 401

    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token("Bearer ")
    assert exc.value.status_code == 401


@pytest.mark.anyio
async def test_accounting_coverage() -> None:
    mock_redis = AsyncMock()
    # Mock pipeline
    pipeline = MagicMock()

    async def aenter(*args: Any, **kwargs: Any) -> MagicMock:
        return pipeline

    pipeline.__aenter__ = AsyncMock(side_effect=aenter)
    pipeline.__aexit__ = AsyncMock()
    pipeline.execute = AsyncMock()
    # redis.pipeline() is synchronous, so we need to ensure the mock method is synchronous
    mock_redis.pipeline = MagicMock(return_value=pipeline)

    # Exception handling
    pipeline.execute.side_effect = Exception("Redis fail")

    # Should not raise, just log exception
    usage = CompletionUsage(completion_tokens=10, prompt_tokens=5, total_tokens=15)
    context = UserContext(sub="proj1", email="test@example.com")

    await record_usage(context, usage, mock_redis)

    # Total tokens <= 0
    mock_redis.pipeline.reset_mock()
    usage_zero = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    await record_usage(context, usage_zero, mock_redis)
    assert not mock_redis.pipeline.called
