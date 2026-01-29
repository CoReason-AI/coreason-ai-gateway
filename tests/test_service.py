from typing import Any

import httpx
import pytest
from coreason_identity.models import UserContext
from openai.types.chat import ChatCompletion

from coreason_ai_gateway.schemas import ChatCompletionRequest
from coreason_ai_gateway.service import Service, ServiceAsync


@pytest.mark.anyio
async def test_service_async_chat_completions(respx_mock: Any) -> None:
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
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

    async with ServiceAsync() as svc:
        context = UserContext(sub="user-123", email="test@example.com")
        req = ChatCompletionRequest(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
        resp = await svc.chat_completions(req, api_key="sk-test", context=context)

        assert isinstance(resp, ChatCompletion)
        assert resp.choices[0].message.content == "Hello there!"
        assert resp.usage and resp.usage.total_tokens == 21


def test_service_sync_chat_completions(respx_mock: Any) -> None:
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello sync!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
            },
        )
    )

    with Service() as svc:
        context = UserContext(sub="user-123", email="test@example.com")
        req = ChatCompletionRequest(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
        resp = svc.chat_completions(req, api_key="sk-test", context=context)

        assert isinstance(resp, ChatCompletion)
        assert resp.choices[0].message.content == "Hello sync!"


@pytest.mark.anyio
async def test_service_async_streaming(respx_mock: Any) -> None:
    # Use bytes for streaming content mock
    mock_content = (
        b'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": null}\n\n'
        b'data: {"choices": [{"delta": {"content": " world"}}], "usage": null}\n\n'
        b"data: [DONE]\n\n"
    )

    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=mock_content, headers={"Content-Type": "text/event-stream"})
    )

    async with ServiceAsync() as svc:
        context = UserContext(sub="user-123", email="test@example.com")
        req = ChatCompletionRequest(model="gpt-4", messages=[{"role": "user", "content": "hi"}], stream=True)
        resp = await svc.chat_completions(req, api_key="sk-test", context=context)

        chunks = []
        # resp should be AsyncIterator
        # assert not isinstance(resp, ChatCompletion)
        # But we need to help mypy. casting is easier or just ignore.

        async for chunk in resp:
            # chunk is ChatCompletionChunk
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert "".join(chunks) == "Hello world"

        # Ensure we close the generator if it's still open?
        # The 'async for' loop exhausts the iterator, so it should be closed.
        # However, pytest-asyncio sometimes warns if 'aclose' isn't explicitly checked or if the underlying
        # response isn't closed.
        # httpx response in 'respx' mock handles this usually.
        # The warning "RuntimeWarning: coroutine method 'aclose' of 'AsyncStream._iter_events' was never awaited"
        # suggests that the OpenAI AsyncStream needs explicit closing or context management?
        # The OpenAI client usually handles this if we use `stream=True` in a context manager,
        # but here we are returning the stream object.
        # The test consumes it fully.


def test_service_sync_streaming_buffered(respx_mock: Any) -> None:
    mock_content = (
        b'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": null}\n\n'
        b'data: {"choices": [{"delta": {"content": " world"}}], "usage": null}\n\n'
        b"data: [DONE]\n\n"
    )

    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=mock_content, headers={"Content-Type": "text/event-stream"})
    )

    with Service() as svc:
        context = UserContext(sub="user-123", email="test@example.com")
        req = ChatCompletionRequest(model="gpt-4", messages=[{"role": "user", "content": "hi"}], stream=True)
        resp_iter = svc.chat_completions(req, api_key="sk-test", context=context)

        # It should be an iterator of chunks
        chunks = []
        for chunk in resp_iter:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert "".join(chunks) == "Hello world"
