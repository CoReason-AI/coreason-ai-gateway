# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from openai import AuthenticationError, BadRequestError


def test_upstream_bad_request(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Simulate Upstream 400 (e.g. Context Limit Exceeded)
    mock_dependencies["client"].chat.completions.create.side_effect = BadRequestError(
        message="Context length exceeded", response=MagicMock(), body={}
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 400
    assert "Upstream provider rejected request" in response.json()["detail"]


def test_upstream_authentication_error(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Simulate Upstream 401 (Gateway used bad key)
    mock_dependencies["client"].chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key", response=MagicMock(), body={}
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    # Should return 502, not 401
    assert response.status_code == 502
    assert "Upstream authentication failed" in response.json()["detail"]


def test_empty_secret_key(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Vault returns structure but key is missing
    mock_dependencies["vault"].get_secret.return_value = {}  # Empty dict

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 503
    assert "Security subsystem unavailable" in response.json()["detail"]


def test_invalid_json_body(client: TestClient) -> None:
    # Malformed JSON (Request validation)
    response = client.post(
        "/v1/chat/completions",
        content="{ invalid json }",
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 422


def test_invalid_pydantic_schema(client: TestClient) -> None:
    # Valid JSON but missing required field 'messages'
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4"},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 422
