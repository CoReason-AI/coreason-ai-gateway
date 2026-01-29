# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import hashlib
import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from coreason_identity.models import UserContext

from coreason_ai_gateway.config import get_settings

"""
Authentication middleware.
Verifies the Gateway Access Token against the configured secret.
"""


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces authentication via Gateway Access Token.
    Instantiates a UserContext upon success and attaches it to request.state.
    """

    async def dispatch(self, request: Request, call_next):
        # 1. Skip Auth for Health Check
        if request.url.path == "/health":
            return await call_next(request)

        # 2. Extract Authorization Header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing Authorization Header"},
            )

        # 3. Parse Scheme and Token
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid Authorization Scheme"},
            )

        # 4. Verify Token
        settings = get_settings()
        # Constant-time comparison
        if not secrets.compare_digest(token, settings.GATEWAY_ACCESS_TOKEN.get_secret_value()):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid Gateway Access Token"},
            )

        # 5. Create UserContext
        try:
            # Extract project ID if present (optional context)
            project_id = request.headers.get("x-coreason-project-id")

            # Hash the token to avoid leaking secrets in logs/context
            token_hash = hashlib.sha256(token.encode()).hexdigest()

            # Combine hash with project_id for granular budgeting if needed
            # sub becomes the effective identity for budgeting/logging
            sub = token_hash
            if project_id:
                sub = f"{token_hash}:{project_id}"

            # Create canonical UserContext
            # Mapping: sub -> hashed identity, email -> dummy, project_context -> header
            context = UserContext(
                sub=sub,
                email="gateway@coreason.ai",
                project_context=project_id,
                permissions=["gateway"],
            )
            request.state.user_context = context

        except Exception:
            # Fail safe if context creation fails
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Authentication Context Error"},
            )

        return await call_next(request)


async def verify_gateway_token(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """
    DEPRECATED: Use AuthMiddleware instead.
    Verifies the Gateway Access Token from the Authorization header.
    Expects: Authorization: Bearer <GATEWAY_ACCESS_TOKEN>

    Args:
        authorization (str | None): The value of the Authorization header.

    Returns:
        str: The token extracted from the header if valid.

    Raises:
        HTTPException: 401 Unauthorized if the token is missing or invalid.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Gateway Access Token",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Gateway Access Token",
        )

    settings = get_settings()
    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(token, settings.GATEWAY_ACCESS_TOKEN.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Gateway Access Token",
        )

    return token
