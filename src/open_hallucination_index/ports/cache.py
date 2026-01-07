"""
CacheProvider Port
==================

Abstract interface for semantic caching of verification results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_hallucination_index.domain.results import VerificationResult


class CacheProvider(ABC):
    """
    Port for caching verification results.

    Supports semantic caching where similar (not just identical)
    inputs may return cached results.

    Responsibilities:
    - Store verification results by input hash
    - Optional semantic similarity lookup
    - TTL-based expiration
    - Cache invalidation
    """

    @abstractmethod
    async def get(self, key: str) -> VerificationResult | None:
        """
        Retrieve cached result by exact key.

        Args:
            key: Cache key (typically hash of input text).

        Returns:
            Cached result or None if not found/expired.
        """
        ...

    @abstractmethod
    async def get_similar(
        self,
        text: str,
        similarity_threshold: float = 0.95,
    ) -> VerificationResult | None:
        """
        Retrieve cached result by semantic similarity.

        Args:
            text: Input text to find similar cached result for.
            similarity_threshold: Minimum similarity to consider a match.

        Returns:
            Cached result for similar input, or None.
        """
        ...

    @abstractmethod
    async def set(
        self,
        key: str,
        result: VerificationResult,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Store result in cache.

        Args:
            key: Cache key.
            result: Verification result to cache.
            ttl_seconds: Time-to-live in seconds (None = use default).
        """
        ...

    @abstractmethod
    async def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached entry.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if entry was found and removed.
        """
        ...

    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the cache is operational."""
        return True
