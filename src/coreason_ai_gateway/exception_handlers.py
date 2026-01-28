# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from coreason_ai_gateway.utils.logger import logger

"""
Exception handlers for the FastAPI application.
Maps upstream provider errors to appropriate HTTP responses.
"""


async def upstream_bad_request_handler(request: Request, exc: BadRequestError) -> JSONResponse:
    """
    Handles 400 Bad Request from upstream providers (e.g. Context Length Exceeded).

    Args:
        request (Request): The incoming HTTP request.
        exc (BadRequestError): The exception raised by the OpenAI client.

    Returns:
        JSONResponse: A 400 response with error details.
    """
    logger.warning(f"Upstream Bad Request: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Upstream provider rejected request: {exc.message}"},
    )


async def upstream_authentication_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    """
    Handles 401 Unauthorized from upstream (Gateway misconfiguration).
    Returns 502 Bad Gateway because the client cannot fix this.

    Args:
        request (Request): The incoming HTTP request.
        exc (AuthenticationError): The exception raised by the OpenAI client.

    Returns:
        JSONResponse: A 502 response indicating upstream authentication failure.
    """
    logger.error(f"Upstream Authentication Failed: {exc}")
    return JSONResponse(
        status_code=502,
        content={"detail": "Upstream authentication failed"},
    )


async def upstream_rate_limit_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """
    Handles 429 Rate Limit from upstream.

    Args:
        request (Request): The incoming HTTP request.
        exc (RateLimitError): The exception raised by the OpenAI client.

    Returns:
        JSONResponse: A 429 response indicating rate limit exceeded.
    """
    logger.warning(f"Upstream Rate Limit: {exc}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Upstream provider rate limit exceeded"},
    )


async def upstream_connection_error_handler(request: Request, exc: APIConnectionError) -> JSONResponse:
    """
    Handles network issues with upstream.

    Args:
        request (Request): The incoming HTTP request.
        exc (APIConnectionError): The exception raised by the OpenAI client.

    Returns:
        JSONResponse: A 502 response indicating upstream connection error.
    """
    logger.error(f"Upstream Connection Error: {exc}")
    return JSONResponse(
        status_code=502,
        content={"detail": f"Upstream provider error: {exc.message}"},
    )


async def upstream_internal_server_error_handler(request: Request, exc: InternalServerError) -> JSONResponse:
    """
    Handles 500 from upstream.

    Args:
        request (Request): The incoming HTTP request.
        exc (InternalServerError): The exception raised by the OpenAI client.

    Returns:
        JSONResponse: A 502 response indicating upstream server error.
    """
    logger.error(f"Upstream Internal Server Error: {exc}")
    return JSONResponse(
        status_code=502,
        content={"detail": f"Upstream provider error: {exc.message}"},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all exception handlers with the FastAPI app.

    Args:
        app (FastAPI): The FastAPI application instance.

    Returns:
        None
    """
    app.add_exception_handler(BadRequestError, upstream_bad_request_handler)
    app.add_exception_handler(AuthenticationError, upstream_authentication_handler)
    app.add_exception_handler(RateLimitError, upstream_rate_limit_handler)
    app.add_exception_handler(APIConnectionError, upstream_connection_error_handler)
    app.add_exception_handler(InternalServerError, upstream_internal_server_error_handler)
