# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from unittest.mock import MagicMock

import pytest
from coreason_ai_gateway.exception_handlers import (
    upstream_connection_error_handler,
    upstream_internal_server_error_handler,
)
from fastapi import Request
from openai import APIConnectionError, InternalServerError


@pytest.mark.anyio
async def test_exception_handlers_coverage() -> None:
    request = MagicMock(spec=Request)

    # APIConnectionError
    exc_conn = APIConnectionError(message="Conn error", request=MagicMock())
    response = await upstream_connection_error_handler(request, exc_conn)
    assert response.status_code == 502
    assert "Conn error" in str(response.body)

    # InternalServerError
    exc_server = InternalServerError(message="Server error", response=MagicMock(), body={})
    response = await upstream_internal_server_error_handler(request, exc_server)
    assert response.status_code == 502
    assert "Server error" in str(response.body)
