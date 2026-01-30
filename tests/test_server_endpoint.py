# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from openai import APIConnectionError, RateLimitError
from openai.types.chat import ChatCompletionChunk

from coreason_ai_gateway.server import app  # noqa: E402


def test_auth_failure(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json={"model": "gpt-4", "messages": []}, headers={"Authorization": "Bearer invalid"}
    )
    assert response.status_code == 401


def test_optional_project_id(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Prepare success response
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.model_dump.return_value = {"id": "123", "choices": []}
    mock_dependencies["client"].chat.completions.create.return_value = mock_response

    # Request without Project ID header should succeed
    response = client.post(
        "/v1/chat/completions", json={"model": "gpt-4", "messages": []}, headers={"Authorization": "Bearer valid-token"}
    )
    assert response.status_code == 200


def test_budget_failure(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    mock_dependencies["redis"].get.return_value = "0"  # Zero budget

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 402


@pytest.mark.anyio
async def test_success_non_streaming(mock_dependencies: dict[str, Any]) -> None:
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.model_dump.return_value = {"id": "123", "choices": []}

    mock_dependencies["client"].chat.completions.create.return_value = mock_response

    # Using standard TestClient
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        assert response.status_code == 200
        mock_dependencies["client"].chat.completions.create.assert_awaited()
        # Verify accounting called (background task might need manual trigger or mock)
        # BackgroundTasks in FastAPI TestClient are executed synchronously usually?
        # Yes, TestClient runs background tasks.

        # Verify redis usage update
        # We can't easily verify pipeline execution details on AsyncMock without inspecting calls deep
        assert mock_dependencies["redis"].pipeline.called


def test_vault_failure(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    mock_dependencies["vault"].get_secret.side_effect = Exception("Vault down")

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 503


def test_vault_invalid_secret(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    mock_dependencies["vault"].get_secret.return_value = {"other": "value"}

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 503


def test_upstream_rate_limit(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Mock create to raise RateLimitError
    mock_dependencies["client"].chat.completions.create.side_effect = RateLimitError(
        message="Rate limit", response=MagicMock(), body={}
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 429
    # Verify retries
    assert mock_dependencies["client"].chat.completions.create.call_count >= 1


def test_upstream_api_error(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    mock_dependencies["client"].chat.completions.create.side_effect = APIConnectionError(
        message="Connection error", request=MagicMock()
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
    )
    assert response.status_code == 502


def test_upstream_generic_error(mock_dependencies: dict[str, Any], client: TestClient) -> None:
    # Generic exception bubbles up as 500 because we re-raise it and it's not handled by our custom handlers.
    # We could add a custom handler for Exception, but typically FastAPI returns 500.
    mock_dependencies["client"].chat.completions.create.side_effect = Exception("Boom")

    with pytest.raises(Exception, match="Boom"):
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )


@pytest.mark.anyio
async def test_streaming_success(mock_dependencies: dict[str, Any]) -> None:
    # Prepare async iterator for streaming response
    chunk1 = MagicMock(spec=ChatCompletionChunk)
    chunk1.model_dump_json.return_value = '{"id": "1", "choices": [{"delta": {"content": "Hello"}}]}'
    chunk1.usage = None

    chunk2 = MagicMock(spec=ChatCompletionChunk)
    chunk2.model_dump_json.return_value = '{"id": "1", "choices": [{"delta": {"content": " World"}}]}'
    chunk2.usage = MagicMock(total_tokens=5)

    async def response_generator(**kwargs: Any) -> AsyncGenerator[ChatCompletionChunk, None]:
        yield chunk1
        yield chunk2

    mock_dependencies["client"].chat.completions.create.side_effect = response_generator

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        assert response.status_code == 200
        content = response.text
        assert "data: {" in content
        assert "data: [DONE]" in content

        # Verify accounting
        assert mock_dependencies["redis"].pipeline.called


@pytest.mark.anyio
async def test_streaming_with_options(mock_dependencies: dict[str, Any]) -> None:
    # Test that stream_options are passed correctly
    async def response_generator(**kwargs: Any) -> AsyncGenerator[ChatCompletionChunk, None]:
        chunk = MagicMock(spec=ChatCompletionChunk)
        chunk.model_dump_json.return_value = '{"id": "1", "choices": []}'
        chunk.usage = None
        yield chunk

    mock_dependencies["client"].chat.completions.create.side_effect = response_generator

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )

        # Verify call arguments
        call_args = mock_dependencies["client"].chat.completions.create.call_args
        assert call_args
        assert call_args.kwargs.get("stream") is True
        assert call_args.kwargs.get("stream_options") == {"include_usage": True}


@pytest.mark.anyio
async def test_streaming_options_ignored_when_not_streaming(mock_dependencies: dict[str, Any]) -> None:
    # stream_options should be allowed even if stream=False
    # (OpenAI allows this, though it might be ignored or used for final usage)
    # The important thing is that the gateway passes it through.
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.model_dump.return_value = {"id": "123", "choices": []}
    mock_dependencies["client"].chat.completions.create.return_value = mock_response

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        assert response.status_code == 200
        call_args = mock_dependencies["client"].chat.completions.create.call_args
        assert call_args.kwargs.get("stream") is False
        assert call_args.kwargs.get("stream_options") == {"include_usage": True}


@pytest.mark.anyio
async def test_streaming_with_none_options(mock_dependencies: dict[str, Any]) -> None:
    # Explicitly sending null/None for stream_options
    async def response_generator(**kwargs: Any) -> AsyncGenerator[ChatCompletionChunk, None]:
        chunk = MagicMock(spec=ChatCompletionChunk)
        chunk.model_dump_json.return_value = '{"id": "1", "choices": []}'
        yield chunk

    mock_dependencies["client"].chat.completions.create.side_effect = response_generator

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": None,
            },
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        call_args = mock_dependencies["client"].chat.completions.create.call_args
        # None should be passed or excluded depending on how model_dump(exclude_unset=True) behaves.
        # If it was sent as None in JSON, it's set to None.
        # But wait, exclude_unset=True excludes fields that were NOT set in the constructor.
        # If we send "stream_options": None in JSON, Pydantic sees it as set to None.
        # Let's check if it is in kwargs.
        assert "stream_options" in call_args.kwargs
        assert call_args.kwargs["stream_options"] is None


@pytest.mark.anyio
async def test_streaming_include_usage_false(mock_dependencies: dict[str, Any]) -> None:
    async def response_generator(**kwargs: Any) -> AsyncGenerator[ChatCompletionChunk, None]:
        chunk = MagicMock(spec=ChatCompletionChunk)
        chunk.model_dump_json.return_value = '{"id": "1", "choices": []}'
        yield chunk

    mock_dependencies["client"].chat.completions.create.side_effect = response_generator

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": False},
            },
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        call_args = mock_dependencies["client"].chat.completions.create.call_args
        assert call_args.kwargs.get("stream_options") == {"include_usage": False}


@pytest.mark.anyio
async def test_complex_streaming_scenario(mock_dependencies: dict[str, Any]) -> None:
    # Test combination of tools, stop, and stream_options
    async def response_generator(**kwargs: Any) -> AsyncGenerator[ChatCompletionChunk, None]:
        chunk = MagicMock(spec=ChatCompletionChunk)
        chunk.model_dump_json.return_value = '{"id": "1", "choices": []}'
        yield chunk

    mock_dependencies["client"].chat.completions.create.side_effect = response_generator

    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}},
        }
    ]

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "tools": tools,
                "tool_choice": "auto",
                "stop": ["STOP"],
            },
            headers={"Authorization": "Bearer valid-token", "X-Coreason-Project-ID": "proj-1"},
        )
        call_args = mock_dependencies["client"].chat.completions.create.call_args
        assert call_args.kwargs.get("stream") is True
        assert call_args.kwargs.get("stream_options") == {"include_usage": True}
        assert call_args.kwargs.get("tools") == tools
        assert call_args.kwargs.get("tool_choice") == "auto"
        assert call_args.kwargs.get("stop") == ["STOP"]


@pytest.mark.anyio
async def test_router_missing_user_context_defensive_check(mock_dependencies: dict[str, Any]) -> None:
    """
    Test the defensive check in the router where user_context is missing from request.state.
    We need to bypass the middleware to trigger this, or mock the middleware to NOT set it.
    """
    # We can use a side_effect on the request.state.user_context access or just remove it in a dependency override?
    # Easier: Mock the request object passed to the endpoint function directly?
    # Or, since we are using TestClient, we rely on the app stack.
    # The AuthMiddleware runs before the router. If AuthMiddleware succeeds, context is set.
    # To test the router check, we need to bypass AuthMiddleware or make it fail to set context but NOT return 401?
    # That's impossible with current middleware logic (it sets context or raises 401).
    # SO, this code is unreachable in normal flow, hence "Defensive".
    # To test it, we must mock the `request` object in a direct call to the endpoint function, OR use a dependency
    # override that deletes the context? No, dependency runs after middleware.

    # We will use `patch` to simulate the endpoint function execution environment or just call the function directly?
    # Calling the function directly is the most robust way to unit test the router logic in isolation.

    from coreason_ai_gateway.routers.chat import chat_completions

    # Mock Request
    req = MagicMock(spec=Request)
    req.state = MagicMock()
    # Explicitly remove user_context
    if hasattr(req.state, "user_context"):
        del req.state.user_context

    # Call endpoint directly
    with pytest.raises(HTTPException) as exc:
        await chat_completions(
            request=req,
            body=MagicMock(),
            background_tasks=MagicMock(),
            service=MagicMock(),
            api_key="key",
            redis_client=MagicMock(),
            _budget=None,
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "User Context Missing"
