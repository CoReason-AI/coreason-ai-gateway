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
from .exception_handlers import register_exception_handlers
from .middleware.auth import AuthMiddleware
from .routers.chat import router as chat_router
from .utils.logger import logger

"""
Main server application module.
Initializes FastAPI app, manages lifecycle (Redis/Vault), and includes routers.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manages the application lifecycle, including connection pools.

    Initializes Redis and Vault clients on startup and ensures they are
    properly closed on shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded to the application during its runtime.

    Raises:
        Exception: If initialization of Redis or Vault fails.
    """
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
        await app.state.vault.auth.authenticate_approle(
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

    # 3. Setup Service
    from coreason_ai_gateway.service import ServiceAsync

    app.state.service = ServiceAsync()
    logger.info("Service initialized.")

    yield

    # 4. Teardown
    logger.info("Shutting down Coreason AI Gateway...")
    if hasattr(app.state, "service"):
        try:
            await app.state.service.__aexit__(None, None, None)
            logger.info("Service closed.")
        except Exception:
            logger.exception("Failed to close Service")

    if hasattr(app.state, "redis"):
        try:
            await app.state.redis.close()
            logger.info("Redis connection closed.")
        except Exception:
            logger.exception("Failed to close Redis connection")

    if hasattr(app.state, "vault"):
        try:
            await app.state.vault.auth.close()
            logger.info("Vault connection closed.")
        except Exception:
            logger.exception("Failed to close Vault connection")


app = FastAPI(title="Coreason AI Gateway", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
register_exception_handlers(app)

# Include Routers
app.include_router(chat_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint to verify service status.

    Returns:
        dict[str, str]: A dictionary containing the service status (e.g., {"status": "ok"}).
    """
    return {"status": "ok"}
