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

from coreason_ai_gateway.utils.logger import logger


async def record_usage(
    project_id: str,
    usage: CompletionUsage | None,
    redis_client: Redis[Any],
    trace_id: str | None = None,
) -> None:
    """
    Records the token usage in Redis asynchronously.
    Updates both the remaining budget and the total usage counter.

    Args:
        project_id (str): The Project ID associated with the request.
        usage (CompletionUsage | None): The usage statistics from the OpenAI response.
        redis_client (Redis[Any]): The Async Redis client.
        trace_id (str | None): Optional trace ID for distributed tracing logs.

    Returns:
        None
    """
    context = {}
    if trace_id:
        context["trace_id"] = trace_id

    with logger.contextualize(**context):
        if not usage:
            logger.warning(f"No usage data provided for Project ID {project_id}")
            return

        total_tokens = usage.total_tokens
        if total_tokens <= 0:
            return

        logger.info(f"Recording usage for Project ID {project_id}: {total_tokens} tokens")

        try:
            async with redis_client.pipeline() as pipe:
                pipe.decrby(f"budget:{project_id}:remaining", total_tokens)
                pipe.incrby(f"usage:{project_id}:total", total_tokens)
                await pipe.execute()
        except Exception:
            logger.exception(f"Failed to record usage for Project ID {project_id}")
