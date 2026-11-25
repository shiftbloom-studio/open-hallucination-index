"""
Redis cache management for benchmark execution.

Provides cache operations needed during benchmarking:
- Connection management
- Key deletion and full flush
- Cache state logging

Design Notes:
- Uses synchronous Redis client (benchmarks run in async context but cache ops are fast)
- Graceful degradation when Redis unavailable
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import redis

if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis cache manager for benchmark execution.

    Handles cache operations during benchmarking:
    - Connecting to Redis
    - Clearing OHI-specific keys
    - Full database flush between runs

    Usage:
        ```python
        cache = CacheManager(host="localhost", port=6379)
        cache.connect(console)
        cache.flush("after_warmup")
        cache.close()
        ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
    ) -> None:
        """
        Initialize cache manager.

        Args:
            host: Redis host
            port: Redis port
            password: Optional Redis password
        """
        self._host = host
        self._port = port
        self._password = password
        self._client: redis.Redis | None = None

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._client is not None

    def connect(self, console: Console | None = None) -> bool:
        """
        Attempt to connect to Redis.

        Args:
            console: Optional Rich console for status output

        Returns:
            True if connected successfully
        """
        try:
            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                password=self._password,
                decode_responses=True,
            )
            self._client.ping()

            if console:
                console.print(f"  [green]✓[/green] Redis connected ({self._host}:{self._port})")
            return True

        except Exception as e:
            if console:
                console.print(f"  [yellow]⚠[/yellow] Redis not available: {e}")
            self._client = None
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None

    def clear_ohi_keys(self) -> int:
        """
        Clear OHI-specific cache keys.

        Returns:
            Number of keys deleted
        """
        if not self._client:
            return 0

        try:
            keys = self._client.keys("ohi:*")
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"Cleared {deleted} OHI cache keys")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return 0

    def flush(self, reason: str = "") -> None:
        """
        Flush entire Redis database.

        Args:
            reason: Optional reason for logging
        """
        if not self._client:
            return

        try:
            self._client.flushdb()
            log_msg = "Redis cache flushed"
            if reason:
                log_msg += f" ({reason})"
            logger.info(log_msg)
        except Exception as e:
            logger.warning(f"Failed to flush Redis cache: {e}")

    def get_key_count(self, pattern: str = "ohi:*") -> int:
        """
        Count keys matching pattern.

        Args:
            pattern: Redis key pattern

        Returns:
            Number of matching keys
        """
        if not self._client:
            return 0

        try:
            return len(self._client.keys(pattern))
        except Exception:
            return 0
