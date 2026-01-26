# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

from coreason_ai_gateway.config import get_settings


async def verify_gateway_token(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """
    Verifies the Gateway Access Token from the Authorization header.
    Expects: Authorization: Bearer <GATEWAY_ACCESS_TOKEN>
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
