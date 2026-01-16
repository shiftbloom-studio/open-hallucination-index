"""
LLMProvider Port
================

Abstract interface for language model inference.
Provides a unified contract for different LLM backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Response from an LLM inference call."""

    content: str
    model: str
    usage: dict[str, int] | None = None  # tokens: prompt, completion, total
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMProvider(ABC):
    """
    Port for LLM inference operations.

    Provides a unified interface for:
    - Chat completions
    - Streaming responses
    - Embeddings generation (if supported)

    Implementations might wrap:
    - OpenAI API
    - vLLM local inference
    - Anthropic Claude
    - Local models via Ollama/LlamaCpp
    """

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...
        # Make this a generator
        if False:
            yield ""

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.

        Default implementation raises NotImplementedError.
        Override in adapters that support embeddings.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        raise NotImplementedError("Embeddings not supported by this provider")

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is operational."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/ID of the model being used."""
        ...
