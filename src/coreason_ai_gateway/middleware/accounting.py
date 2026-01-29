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

from typing import Any

from openai.types import CompletionUsage
from redis.asyncio import Redis
from coreason_identity.models import UserContext

from coreason_ai_gateway.utils.logger import logger

"""
Accounting middleware for tracking token usage.
Updates Redis counters asynchronously.
"""


async def record_usage(
    context: UserContext,
    usage: CompletionUsage | None,
    redis_client: Redis[Any],
    trace_id: str | None = None,
) -> None:
    """
    Records the token usage in Redis asynchronously.
    Updates both the remaining budget and the total usage counter.

    Args:
        context (UserContext): The User Context containing identity.
        usage (CompletionUsage | None): The usage statistics from the OpenAI response.
        redis_client (Redis[Any]): The Async Redis client.
        trace_id (str | None): Optional trace ID for distributed tracing logs.

    Returns:
        None
    """
    user_id = context.sub
    ctx = {}
    if trace_id:
        ctx["trace_id"] = trace_id

    with logger.contextualize(**ctx):
        if not usage:
            logger.warning(f"No usage data provided for User ID {user_id}")
            return

        total_tokens = usage.total_tokens
        if total_tokens <= 0:
            return

        logger.info(f"Recording usage for User ID {user_id}: {total_tokens} tokens")

        try:
            async with redis_client.pipeline() as pipe:
                pipe.decrby(f"budget:{user_id}:remaining", total_tokens)
                pipe.incrby(f"usage:{user_id}:total", total_tokens)
                await pipe.execute()
        except Exception:
            logger.exception(f"Failed to record usage for User ID {user_id}")
