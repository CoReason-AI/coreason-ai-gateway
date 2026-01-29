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
from unittest.mock import patch

import pytest
from coreason_ai_gateway.middleware.auth import AuthMiddleware, verify_gateway_token
from fastapi import HTTPException, Request, Response
from pydantic import SecretStr
from starlette.types import Receive, Scope, Send


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


@pytest.mark.anyio
async def test_verify_gateway_token_case_insensitive_scheme(mock_settings: str) -> None:
    token = mock_settings
    header = f"bearer {token}"
    result = await verify_gateway_token(header)
    assert result == token


@pytest.mark.anyio
async def test_verify_gateway_token_empty_payload() -> None:
    header = "Bearer "
    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token(header)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid Gateway Access Token"


@pytest.mark.anyio
async def test_verify_gateway_token_malformed_whitespace(mock_settings: str) -> None:
    token = mock_settings
    # Double space should result in the token being " <token>" which fails comparison
    header = f"Bearer  {token}"
    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token(header)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid Gateway Access Token"


@pytest.mark.anyio
async def test_verify_gateway_token_no_space() -> None:
    header = "Bearer"
    with pytest.raises(HTTPException) as exc:
        await verify_gateway_token(header)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid Gateway Access Token"


# --- AuthMiddleware Tests ---


async def simple_app(scope: Scope, receive: Receive, send: Send) -> None:
    response = Response("OK")
    await response(scope, receive, send)


@pytest.mark.anyio
async def test_auth_middleware_valid(mock_settings: str) -> None:
    token = mock_settings
    middleware = AuthMiddleware(simple_app)

    scope = {
        "type": "http",
        "headers": [
            (b"authorization", f"Bearer {token}".encode()),
            (b"x-coreason-project-id", b"proj-123"),
        ],
        "path": "/v1/chat/completions",
    }
    request = Request(scope)

    async def call_next(req: Request) -> Response:
        assert hasattr(req.state, "user_context")
        # Verify sub is hashed and contains project_id
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expected_sub = f"{token_hash}:proj-123"
        assert req.state.user_context.sub == expected_sub
        assert req.state.user_context.project_context == "proj-123"
        return Response("OK")

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 200


@pytest.mark.anyio
async def test_auth_middleware_valid_no_project(mock_settings: str) -> None:
    token = mock_settings
    middleware = AuthMiddleware(simple_app)

    scope = {
        "type": "http",
        "headers": [
            (b"authorization", f"Bearer {token}".encode()),
        ],
        "path": "/v1/chat/completions",
    }
    request = Request(scope)

    async def call_next(req: Request) -> Response:
        assert hasattr(req.state, "user_context")
        # Verify sub is hash only
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        assert req.state.user_context.sub == token_hash
        assert req.state.user_context.project_context is None
        return Response("OK")

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 200


@pytest.mark.anyio
async def test_auth_middleware_health_bypass(mock_settings: str) -> None:
    middleware = AuthMiddleware(simple_app)

    scope = {
        "type": "http",
        "headers": [],
        "path": "/health",
    }
    request = Request(scope)

    async def call_next(req: Request) -> Response:
        # No context expected
        assert not hasattr(req.state, "user_context")
        return Response("OK")

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 200


@pytest.mark.anyio
async def test_auth_middleware_missing_header(mock_settings: str) -> None:
    middleware = AuthMiddleware(simple_app)

    scope = {
        "type": "http",
        "headers": [],
        "path": "/protected",
    }
    request = Request(scope)

    async def call_next(req: Request) -> Response:
        return Response("Should Not Reach Here")

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 401


@pytest.mark.anyio
async def test_auth_middleware_invalid_token(mock_settings: str) -> None:
    middleware = AuthMiddleware(simple_app)

    scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer invalid")],
        "path": "/protected",
    }
    request = Request(scope)

    async def call_next(req: Request) -> Response:
        return Response("Should Not Reach Here")

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 401


@pytest.mark.anyio
async def test_auth_middleware_wrong_scheme(mock_settings: str) -> None:
    middleware = AuthMiddleware(simple_app)
    scope = {"type": "http", "headers": [(b"authorization", b"Basic token")], "path": "/protected"}
    request = Request(scope)

    async def call_next(req: Request) -> Response:
        return Response("Should Not Reach")

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 401


@pytest.mark.anyio
async def test_auth_middleware_context_exception(mock_settings: str) -> None:
    middleware = AuthMiddleware(simple_app)
    token = mock_settings
    scope = {"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())], "path": "/protected"}
    request = Request(scope)

    with patch("coreason_ai_gateway.middleware.auth.UserContext", side_effect=Exception("Boom")):

        async def call_next(req: Request) -> Response:
            return Response("Should Not Reach")

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 500
