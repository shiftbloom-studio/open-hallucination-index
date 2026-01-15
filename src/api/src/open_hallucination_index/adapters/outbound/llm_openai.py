"""
OpenAI-Compatible LLM Adapter
=============================

Adapter for OpenAI-compatible LLM APIs (including vLLM, Ollama, etc.).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from open_hallucination_index.ports.llm_provider import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
)

if TYPE_CHECKING:
    from open_hallucination_index.infrastructure.config import LLMSettings

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Exception raised when LLM inference fails."""

    pass


class OpenAILLMAdapter(LLMProvider):
    """
    Adapter for OpenAI-compatible LLM inference APIs.

    Wraps the openai Python client for:
    - Local vLLM inference (port 8000)
    - OpenAI API
    - Azure OpenAI
    - Ollama with OpenAI compatibility
    """

    def __init__(self, settings: LLMSettings) -> None:
        """
        Initialize the adapter with configuration.

        Args:
            settings: LLM connection settings.
        """
        self._settings = settings

        # Configure connection pooling limits for optimal performance
        # - max_connections: Total concurrent connections allowed
        # - max_keepalive_connections: Connections kept alive for reuse
        connection_limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
        )

        # Configure granular timeouts
        timeout_config = httpx.Timeout(
            connect=10.0,  # Connection establishment timeout
            read=self._settings.timeout_seconds,  # Read timeout (response)
            write=30.0,  # Write timeout (request body)
            pool=5.0,  # Pool acquisition timeout
        )

        # Force IPv4 by using a custom transport with retries
        ipv4_transport = httpx.AsyncHTTPTransport(
            local_address="0.0.0.0",
            retries=2,  # Retry on connection failures
        )

        # Create optimized HTTP client with connection pooling
        http_client = httpx.AsyncClient(
            transport=ipv4_transport,
            limits=connection_limits,
            timeout=timeout_config,
            http2=True,  # Enable HTTP/2 for better multiplexing
        )

        self._client = AsyncOpenAI(
            base_url=self._settings.base_url,
            api_key=self._settings.api_key.get_secret_value(),
            timeout=self._settings.timeout_seconds,
            max_retries=self._settings.max_retries,
            http_client=http_client,
        )
        self._model = settings.model

    def _format_messages_for_mistral(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        """
        Format messages for Mistral-Instruct models.

        Mistral-Instruct models require strict user/assistant alternation.
        System messages are prepended to the first user message.
        """
        formatted: list[dict[str, str]] = []
        system_content: list[str] = []

        for msg in messages:
            if msg.role == "system":
                system_content.append(msg.content)
            elif msg.role == "user":
                content = msg.content
                if system_content:
                    # Prepend system instructions to first user message
                    prefix = "\n\n".join(system_content)
                    content = f"[INST] {prefix}\n\n{content} [/INST]"
                    system_content.clear()
                formatted.append({"role": "user", "content": content})
            else:  # assistant
                formatted.append({"role": msg.role, "content": msg.content})

        return formatted

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.

        Returns:
            LLM response with generated content.

        Raises:
            LLMProviderError: If inference fails.
        """
        try:
            # Mistral-Instruct models don't support system messages natively.
            # We merge system messages into the first user message.
            formatted_messages = self._format_messages_for_mistral(messages)

            response_format = {"type": "json_object"} if json_mode else None

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                response_format=response_format,
            )

            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
                raw_response=response.model_dump(),
            )

        except APIConnectionError as e:
            # Use warning level for connection errors - expected during startup
            # or when vLLM is temporarily unavailable. Callers handle fallback.
            logger.warning(f"LLM not available (will use fallback): {e}")
            raise LLMProviderError(f"Failed to connect to LLM: {e}") from e
        except APIStatusError as e:
            logger.error(f"LLM API error: {e.status_code} - {e.message}")
            raise LLMProviderError(f"LLM API error: {e.message}") from e
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            raise LLMProviderError(f"Unexpected error: {e}") from e

    async def complete_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens as they're generated.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Token strings as they're generated.
        """
        try:
            formatted_messages = self._format_messages_for_mistral(messages)

            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise LLMProviderError(f"Streaming failed: {e}") from e

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding vector for text using OpenAI embeddings API.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        try:
            # Use OpenAI API key for embeddings (separate from vLLM key)
            openai_key = self._settings.openai_api_key.get_secret_value()
            if not openai_key:
                raise LLMProviderError("OPENAI_API_KEY not set. Required for embeddings.")
            embedding_client = AsyncOpenAI(
                api_key=openai_key,
                timeout=self._settings.timeout_seconds,
            )
            response = await embedding_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise LLMProviderError(f"Embedding failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if the LLM API is reachable."""
        try:
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False
