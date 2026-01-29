# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException, status
from redis.asyncio import Redis

"""
Budget middleware for enforcing financial limits.
Checks estimated cost against Redis budget before processing.
"""


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimates the number of tokens in the messages using a fast heuristic.
    Rule: len(json.dumps(messages)) // 4.

    Args:
        messages (list[dict[str, Any]]): The list of message dictionaries.

    Returns:
        int: The estimated token count.
    """
    try:
        content = json.dumps(messages)
        return len(content) // 4
    except (TypeError, ValueError):
        # Fallback for non-serializable objects (though Pydantic ensures structure)
        return len(str(messages)) // 4


async def check_budget(project_id: str, estimated_cost: int, redis_client: Redis[Any]) -> None:
    """
    Checks if the project has sufficient budget.
    Raises HTTPException(402) if budget is insufficient or missing.

    Args:
        project_id (str): The Project ID from headers.
        estimated_cost (int): The estimated token cost.
        redis_client (Redis[Any]): The Async Redis client.

    Raises:
        HTTPException: 402 Payment Required if budget < cost.
    """
    key = f"budget:{project_id}:remaining"
    remaining = await redis_client.get(key)

    if remaining is None:
        # Fail Secure: No budget key means 0 budget.
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Budget exceeded for Project ID {project_id}",
        )

    try:
        remaining_int = int(remaining)
    except (ValueError, TypeError):
        # Corrupted data acts as 0 budget
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Budget exceeded for Project ID {project_id}",
        ) from None

    if remaining_int < estimated_cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Budget exceeded for Project ID {project_id}",
        )
