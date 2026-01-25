# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from contextlib import asynccontextmanager
from typing import AsyncIterator

from coreason_vault import CoreasonVaultConfig, VaultManagerAsync
from fastapi import FastAPI
from redis import asyncio as redis

from .config import get_settings
from .utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting up Coreason AI Gateway...")

    # 1. Setup Redis
    try:
        app.state.redis = redis.from_url(str(settings.REDIS_URL), encoding="utf-8", decode_responses=True)
        logger.info("Redis client initialized.")
    except Exception as e:
        logger.exception("Failed to initialize Redis client")
        raise e

    # 2. Setup Vault
    try:
        vault_config = CoreasonVaultConfig(VAULT_ADDR=str(settings.VAULT_ADDR))
        app.state.vault = VaultManagerAsync(config=vault_config)
        await app.state.vault.authenticate(
            role_id=settings.VAULT_ROLE_ID,
            secret_id=settings.VAULT_SECRET_ID.get_secret_value(),
        )
        logger.info("Vault client authenticated.")
    except Exception as e:
        logger.exception("Failed to initialize Vault client")
        # Ensure redis is closed if vault fails
        if hasattr(app.state, "redis"):
            await app.state.redis.close()
        raise e

    yield

    # 3. Teardown
    logger.info("Shutting down Coreason AI Gateway...")
    if hasattr(app.state, "redis"):
        await app.state.redis.close()
        logger.info("Redis connection closed.")

    if hasattr(app.state, "vault"):
        await app.state.vault.close()
        logger.info("Vault connection closed.")


app = FastAPI(title="Coreason AI Gateway", lifespan=lifespan)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint to verify service status.
    """
    return {"status": "ok"}
