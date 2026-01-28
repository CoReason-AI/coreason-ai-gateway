# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Any, AsyncIterator, Iterator, Optional, Union

import anyio
import httpx
from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, stop_after_delay, wait_exponential

from coreason_ai_gateway.config import get_settings
from coreason_ai_gateway.schemas import ChatCompletionRequest


class ServiceAsync:
    """
    Core Async Service for CoReason AI Gateway.
    Handles lifecycle management and interactions with upstream providers.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        """
        Initialize the ServiceAsync instance.

        Args:
            client (Optional[httpx.AsyncClient]): An external HTTP client.
                                                  If None, a new one is created.
        """
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

    async def __aenter__(self) -> "ServiceAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._client.aclose()
        return None

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        api_key: str,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Execute a chat completion request against the upstream provider.

        Handles retries and client configuration.

        Args:
            request (ChatCompletionRequest): The request model.
            api_key (str): The API key for the provider.

        Returns:
            Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]: The response from OpenAI.
        """
        settings = get_settings()

        # Initialize OpenAI client with the shared httpx client
        client = AsyncOpenAI(api_key=api_key, http_client=self._client, max_retries=0)

        kwargs = request.model_dump(exclude_unset=True)

        try:
            async for attempt in AsyncRetrying(
                stop=(
                    stop_after_attempt(settings.RETRY_STOP_AFTER_ATTEMPT)
                    | stop_after_delay(settings.RETRY_STOP_AFTER_DELAY)
                ),
                wait=wait_exponential(multiplier=1, min=settings.RETRY_WAIT_MIN, max=settings.RETRY_WAIT_MAX),
                retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
                reraise=True,
            ):
                with attempt:
                    response = await client.chat.completions.create(**kwargs)
                    return response
        except Exception as e:
            raise e
        raise RuntimeError("Retry loop finished without result")  # pragma: no cover


class Service:
    """
    Synchronous Facade for ServiceAsync.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._async = ServiceAsync(client)

    def __enter__(self) -> "Service":
        # Initialize async context logic if needed?
        # Since ServiceAsync.__aenter__ is lightweight (returns self), we can skip explicitly running it
        # UNLESS it does logic. It is just `return self`.
        # However, to be correct, we should run it?
        # anyio.run(self._async.__aenter__)
        # But as discussed, anyio.run creates a loop and closes it.
        # If __aenter__ did resource allocation bound to the loop, it would be lost.
        # Fortunately ServiceAsync only manages httpx.AsyncClient which is lazy.
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        anyio.run(self._async.__aexit__, exc_type, exc_val, exc_tb)

    def chat_completions(
        self,
        request: ChatCompletionRequest,
        api_key: str,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Synchronous wrapper for chat_completions.

        Note: If streaming is requested, the entire stream is buffered and returned as an iterator,
        because the async event loop closes upon return.
        """

        async def wrapper() -> Union[ChatCompletion, list[ChatCompletionChunk]]:
            response = await self._async.chat_completions(request, api_key)
            if request.stream:
                # We must consume the stream while the loop is open
                chunks = []
                # response is AsyncStream[ChatCompletionChunk]
                async for chunk in response:
                    chunks.append(chunk)
                return chunks
            return response

        result = anyio.run(wrapper)

        if isinstance(result, list):
            return iter(result)
        return result
