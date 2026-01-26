# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from coreason_ai_gateway.server import app


@pytest.fixture
def mock_external_deps(monkeypatch: pytest.MonkeyPatch) -> Generator[dict[str, Any], None, None]:
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
    monkeypatch.setenv("VAULT_ROLE_ID", "dummy-role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "dummy-secret-id")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    monkeypatch.setenv("GATEWAY_ACCESS_TOKEN", "valid-token")

    with (
        patch("coreason_ai_gateway.server.redis.from_url") as mock_redis,
        patch("coreason_ai_gateway.server.VaultManagerAsync") as mock_vault,
        patch("coreason_ai_gateway.server.CoreasonVaultConfig"),
        # We DO NOT patch AsyncOpenAI here because we want to test the real client against mocked HTTP
    ):
        # Redis
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance
        redis_instance.get.return_value = "1000"

        # Pipeline (sync compatible)
        pipeline_mock = MagicMock()
        redis_instance.pipeline = MagicMock(return_value=pipeline_mock)

        async def aenter(*args: Any, **kwargs: Any) -> MagicMock:
            return pipeline_mock

        pipeline_mock.__aenter__ = AsyncMock(side_effect=aenter)
        pipeline_mock.__aexit__ = AsyncMock()
        pipeline_mock.execute = AsyncMock()

        # Vault
        vault_instance = AsyncMock()
        mock_vault.return_value = vault_instance
        vault_instance.authenticate = AsyncMock()
        # Default secret for happy path
        vault_instance.get_secret.return_value = {"api_key": "sk-dummy-key"}

        yield {
            "redis": redis_instance,
            "vault": vault_instance,
            "pipeline": pipeline_mock,
        }


@respx.mock  # type: ignore[misc]
@pytest.mark.anyio
async def test_integration_happy_path(mock_external_deps: dict[str, Any]) -> None:
    # Mock OpenAI
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4-0613",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello there!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
            },
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-integration"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello there!"

        # Verify request to OpenAI
        assert route.called
        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer sk-dummy-key"
        # Verify body forwarding
        import json

        body = json.loads(request.content)
        assert body["model"] == "gpt-4"
        assert body["messages"][0]["content"] == "Hello!"

        # Verify Redis Accounting (Background Task)
        assert mock_external_deps["pipeline"].execute.called


@respx.mock  # type: ignore[misc]
@pytest.mark.anyio
async def test_integration_upstream_500_retry(mock_external_deps: dict[str, Any]) -> None:
    # Simulates: 500, 500, 200 (Success on 3rd attempt)
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            Response(500, json={"error": "server_error"}),
            Response(502, json={"error": "bad_gateway"}),
            Response(
                200,
                json={
                    "id": "chatcmpl-retry",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Finally works!"}}],
                    "usage": {"total_tokens": 10},
                },
            ),
        ]
    )

    # We might need to speed up retries to avoid slow tests, but mocked time is better.
    # Tenacity uses time.sleep, but we are in async? Tenacity supports async.
    # We rely on defaults in config (min=2s).
    # We should override config for test speed.
    with patch("coreason_ai_gateway.server.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.RETRY_WAIT_MIN = 0.01
        settings.RETRY_WAIT_MAX = 0.05
        settings.RETRY_STOP_AFTER_ATTEMPT = 5
        settings.VAULT_ADDR = "http://vault:8200"
        settings.VAULT_ROLE_ID = "dummy-role-id"
        settings.VAULT_SECRET_ID.get_secret_value.return_value = "dummy-secret-id"
        settings.REDIS_URL = "redis://redis:6379"
        settings.GATEWAY_ACCESS_TOKEN.get_secret_value.return_value = "valid-token"

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "retry me"}]},
                headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-retry"},
            )

            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "Finally works!"
            assert route.call_count == 3


@respx.mock  # type: ignore[misc]
@pytest.mark.anyio
async def test_integration_upstream_failure_exhausted(mock_external_deps: dict[str, Any]) -> None:
    # Simulates: 500 forever
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(500, json={"error": "server_error"})
    )

    with patch("coreason_ai_gateway.server.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.RETRY_WAIT_MIN = 0.01
        settings.RETRY_WAIT_MAX = 0.05
        settings.RETRY_STOP_AFTER_ATTEMPT = 2
        settings.VAULT_ADDR = "http://vault:8200"
        settings.VAULT_ROLE_ID = "dummy-role-id"
        settings.VAULT_SECRET_ID.get_secret_value.return_value = "dummy-secret-id"
        settings.REDIS_URL = "redis://redis:6379"
        settings.GATEWAY_ACCESS_TOKEN.get_secret_value.return_value = "valid-token"

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "fail me"}]},
                headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-fail"},
            )

            # Gateway maps APIConnectionError/InternalServerError to 502
            # Or if it's 500 from upstream, we return 502
            assert response.status_code == 502
            assert route.call_count == 2


@respx.mock  # type: ignore[misc]
@pytest.mark.anyio
async def test_integration_streaming(mock_external_deps: dict[str, Any]) -> None:
    # SSE Stream
    stream_content = [
        'data: {"id":"1","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        'data: {"id":"1","choices":[{"delta":{"content":"lo"}}]}\n\n',
        'data: {"usage":{"total_tokens": 5}}\n\n',  # Usage reporting
        "data: [DONE]\n\n",
    ]

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content="".join(stream_content),
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "stream"}], "stream": True},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-stream"},
        )

        assert response.status_code == 200
        # Check output
        content = response.text
        # Gateway re-serializes, so check for essential content
        assert "Hel" in content
        assert "lo" in content
        assert "[DONE]" in content
        assert "data: " in content

        assert route.called

        # Verify usage accounting
        # Note: Usage accounting happens AFTER stream is consumed.
        assert mock_external_deps["pipeline"].execute.called


@respx.mock
@pytest.mark.anyio
async def test_integration_upstream_connection_error(mock_external_deps: dict[str, Any]) -> None:
    # Simulate connection error (e.g., DNS failure, timeout)
    # respx.mock(side_effect=httpx.ConnectError)

    # We need to import httpx exceptions
    import httpx

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("Connection refused", request=MagicMock())
    )

    with patch("coreason_ai_gateway.server.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.RETRY_WAIT_MIN = 0.01
        settings.RETRY_WAIT_MAX = 0.05
        settings.RETRY_STOP_AFTER_ATTEMPT = 2
        settings.VAULT_ADDR = "http://vault:8200"
        settings.VAULT_ROLE_ID = "dummy-role-id"
        settings.VAULT_SECRET_ID.get_secret_value.return_value = "dummy-secret-id"
        settings.REDIS_URL = "redis://redis:6379"
        settings.GATEWAY_ACCESS_TOKEN.get_secret_value.return_value = "valid-token"

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "connect fail"}]},
                headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-connect-fail"},
            )

            # APIConnectionError -> 502
            assert response.status_code == 502
            assert "Upstream provider error" in response.json()["detail"]
            assert route.call_count == 2


@respx.mock
@pytest.mark.anyio
async def test_integration_mid_stream_error(mock_external_deps: dict[str, Any]) -> None:
    # Simulate a stream that breaks midway
    import httpx
    import openai

    # Define an iterator that yields one chunk then raises
    async def broken_stream() -> Any:
        yield b'data: {"id":"1","choices":[{"delta":{"content":"Hel"}}]}\n\n'
        # httpx treats exceptions in generators as stream errors
        raise httpx.ReadError("Network Reset")

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=broken_stream(),
        )
    )

    with TestClient(app) as client:
        # Note: TestClient/Starlette might swallow the exception during streaming if the generator exits early
        # or behaves like a truncated stream. We primarily verify that usage accounting (which happens in 'finally')
        # is NOT triggered because 'usage' variable remains None.
        try:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "stream"}]},
                headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-stream-fail"},
            )
            # Consume stream to trigger error (or partial read)
            for _ in response.iter_lines():
                pass
        except (httpx.ReadError, openai.APIConnectionError, Exception):
            pass  # Expected if it raises

        # Verify usage was NOT recorded (because stream crashed before usage chunk)
        assert not mock_external_deps["pipeline"].execute.called


@respx.mock
@pytest.mark.anyio
async def test_integration_malformed_json_response(mock_external_deps: dict[str, Any]) -> None:
    # Upstream returns 200 but garbage body
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            content="NOT JSON",
        )
    )

    with TestClient(app) as client:
        # AsyncOpenAI will raise APIResponseValidationError or similar when parsing JSON fails.
        # This is usually wrapped in APIError or handled as internal error.
        # If unhandled, it bubbles as 500.

        # Note: If tenacity catches it, it might retry.
        # But parsing errors on 200 OK might not be in the retry list (RateLimit, Connection, 500).
        # AsyncOpenAI raises APIError for bad json? No, likely APIMismatchedResponse or similar.

        # Let's verify what happens. It should probably result in a 500 (unhandled exception in handler)
        # or 502 if we catch generic APIErrors.

        # Current server.py retry list: (RateLimitError, APIConnectionError, InternalServerError)
        # Malformed JSON (on 200) is likely openai.APIStatusError or just specific parsing error.

        # Because we don't catch it in server.py explicitly (except via global handlers maybe?),
        # checking exception handlers...
        # We handle: BadRequest, Authentication, RateLimit, APIConnection, InternalServer.
        # Malformed JSON might fall through.

        with pytest.raises(Exception):
            # We expect some exception if TestClient catches it, or 500 if middleware catches it.
            # TestClient usually re-raises app exceptions.
            client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "bad json"}]},
                headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-malformed"},
            )
