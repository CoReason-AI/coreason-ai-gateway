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
from unittest.mock import AsyncMock, patch

import pytest
from coreason_ai_gateway.middleware.budget import check_budget, estimate_tokens
from coreason_identity.models import UserContext
from fastapi import HTTPException

# --- estimate_tokens Tests ---


def test_estimate_tokens_simple() -> None:
    messages = [{"role": "user", "content": "hello"}]
    # json: [{"role": "user", "content": "hello"}] -> length ~37
    # 37 // 4 = 9
    assert estimate_tokens(messages) > 0


def test_estimate_tokens_empty() -> None:
    messages: list[dict[str, Any]] = []
    # json: [] -> length 2
    # 2 // 4 = 0
    assert estimate_tokens(messages) == 0


def test_estimate_tokens_fallback() -> None:
    # Simulate a non-serializable object by mocking json.dumps
    with patch("coreason_ai_gateway.middleware.budget.json.dumps", side_effect=ValueError):
        messages = [{"role": "user", "content": "test"}]
        # Should rely on str(messages)
        assert estimate_tokens(messages) > 0


# --- check_budget Tests ---


@pytest.mark.anyio
async def test_check_budget_sufficient() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = "1000"
    context = UserContext(sub="proj-123", email="test@example.com")

    # Should not raise
    await check_budget(context, 100, mock_redis)
    mock_redis.get.assert_awaited_once_with("budget:proj-123:remaining")


@pytest.mark.anyio
async def test_check_budget_exact() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = "100"
    context = UserContext(sub="proj-123", email="test@example.com")

    # Should not raise (assuming remaining >= cost is allowed)
    await check_budget(context, 100, mock_redis)


@pytest.mark.anyio
async def test_check_budget_insufficient() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = "50"
    context = UserContext(sub="proj-123", email="test@example.com")

    with pytest.raises(HTTPException) as exc:
        await check_budget(context, 100, mock_redis)

    assert exc.value.status_code == 402
    assert exc.value.detail == "Budget exceeded for User ID proj-123"


@pytest.mark.anyio
async def test_check_budget_missing_key() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    context = UserContext(sub="proj-123", email="test@example.com")

    with pytest.raises(HTTPException) as exc:
        await check_budget(context, 100, mock_redis)

    assert exc.value.status_code == 402
    assert exc.value.detail == "Budget exceeded for User ID proj-123"


@pytest.mark.anyio
async def test_check_budget_corrupted_value() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = "not-a-number"
    context = UserContext(sub="proj-123", email="test@example.com")

    with pytest.raises(HTTPException) as exc:
        await check_budget(context, 100, mock_redis)

    assert exc.value.status_code == 402
    assert exc.value.detail == "Budget exceeded for User ID proj-123"
