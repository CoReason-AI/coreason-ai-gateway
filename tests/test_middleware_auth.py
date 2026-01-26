# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from coreason_ai_gateway.middleware.auth import verify_gateway_token


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> str:
    mock_token = "secret-token-123"

    class MockSettings:
        GATEWAY_ACCESS_TOKEN = SecretStr(mock_token)

    def mock_get_settings() -> MockSettings:
        return MockSettings()

    monkeypatch.setattr("coreason_ai_gateway.middleware.auth.get_settings", mock_get_settings)
    return mock_token


@pytest.mark.anyio
async def test_verify_gateway_token_valid(mock_settings: str) -> None:
    token = mock_settings
    header = f"Bearer {token}"
    result = await verify_gateway_token(header)
    assert result == token


@pytest.mark.anyio
async def test_verify_gateway_token_invalid_token(mock_settings: str) -> None:
    header = "Bearer wrong-token"
    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token(header)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid Gateway Access Token"


@pytest.mark.anyio
async def test_verify_gateway_token_missing_header() -> None:
    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token(None)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid Gateway Access Token"


@pytest.mark.anyio
async def test_verify_gateway_token_wrong_scheme(mock_settings: str) -> None:
    token = mock_settings
    header = f"Basic {token}"
    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token(header)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid Gateway Access Token"
