"""
Redis Cache Adapter
===================

Adapter for Redis as a semantic cache for verification results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis

from open_hallucination_index.domain.results import ClaimVerification, VerificationResult
from open_hallucination_index.ports.cache import CacheProvider

if TYPE_CHECKING:
    from open_hallucination_index.infrastructure.config import RedisSettings

logger = logging.getLogger(__name__)

CACHE_PREFIX = "ohi:result:"
CLAIM_CACHE_PREFIX = "ohi:claim:"


class RedisCacheError(Exception):
    """Exception raised when Redis operations fail."""

    pass


class RedisCacheAdapter(CacheProvider):
    """
    Adapter for Redis as a verification result cache.

    Supports:
    - Exact key lookup (by input hash)
    - TTL-based expiration
    """

    def __init__(self, settings: RedisSettings) -> None:
        """
        Initialize the adapter with configuration.

        Args:
            settings: Redis connection settings.
        """
        self._settings = settings
        self._client: redis.Redis | None = None  # type: ignore[type-arg]
        self._default_ttl = settings.cache_ttl_seconds
        self._claim_ttl = settings.claim_cache_ttl_seconds

    async def connect(self) -> None:
        """Establish connection to Redis with retries."""
        max_retries = 10
        retry_delay = 1.0
        last_error = None

        for attempt in range(max_retries):
            try:
                password = None
                if self._settings.password:
                    password = self._settings.password.get_secret_value()

                if self._settings.socket_path:
                    # Connection via Unix Domain Socket
                    pool = redis.ConnectionPool(
                        connection_class=redis.UnixDomainSocketConnection,
                        path=self._settings.socket_path,
                        password=password,
                        db=self._settings.db,
                        max_connections=self._settings.max_connections,
                    )
                    if attempt == 0:
                        logger.info(
                            "Connecting to Redis via Unix socket: %s",
                            self._settings.socket_path,
                        )
                else:
                    # Create a connection pool with IPv4-only socket
                    pool = redis.ConnectionPool(
                        host=self._settings.host,
                        port=self._settings.port,
                        password=password,
                        db=self._settings.db,
                        max_connections=self._settings.max_connections,
                    )
                    # Force IPv4 socket family on the connection class
                    pool.connection_class = type(
                        "IPv4Connection",
                        (pool.connection_class,),
                        {"socket_type": socket.AF_INET},
                    )
                    if attempt == 0:
                        logger.info(
                            "Connecting to Redis via TCP: %s:%s",
                            self._settings.host,
                            self._settings.port,
                        )

                self._client = redis.Redis(
                    connection_pool=pool,
                    decode_responses=False,
                )

                await self._client.ping()  # type: ignore[misc]

                if self._settings.socket_path:
                    logger.info(
                        "Connected to Redis via Unix socket: %s",
                        self._settings.socket_path,
                    )
                else:
                    logger.info(
                        "Connected to Redis at %s:%s",
                        self._settings.host,
                        self._settings.port,
                    )
                return

            except (redis.ConnectionError, FileNotFoundError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "Redis connection attempt %s failed: %s. Retrying in %ss...",
                        attempt + 1,
                        e,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                continue
            except Exception as e:
                logger.error(f"Unexpected Redis error: {e}")
                raise RedisCacheError(f"Connection failed: {e}") from e

        logger.error(f"Redis connection failed after {max_retries} attempts: {last_error}")
        raise RedisCacheError(f"Connection failed after {max_retries} attempts: {last_error}")

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")

    async def health_check(self) -> bool:
        """Check if Redis is reachable."""
        if self._client is None:
            return False
        try:
            await self._client.ping()  # type: ignore[misc]
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{CACHE_PREFIX}{key}"

    async def get(self, key: str) -> VerificationResult | None:
        """
        Retrieve cached result by exact key.

        Args:
            key: Cache key (typically hash of input text).

        Returns:
            Cached result or None if not found/expired.
        """
        if self._client is None:
            logger.warning("Redis client not connected")
            return None

        try:
            full_key = self._make_key(key)
            data = await self._client.get(full_key)

            if data is None:
                return None

            json_data = json.loads(data)
            return VerificationResult.model_validate(json_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize cached result: {e}")
            await self.invalidate(key)
            return None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    async def get_similar(
        self,
        text: str,
        similarity_threshold: float = 0.95,
    ) -> VerificationResult | None:
        """
        Retrieve cached result by semantic similarity.

        Uses normalized text hash for approximate matching.

        Args:
            text: Input text to find similar cached result for.
            similarity_threshold: Minimum similarity to consider a match.

        Returns:
            Cached result for similar input, or None.
        """
        normalized = " ".join(text.lower().split())
        normalized_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return await self.get(normalized_hash)

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
        if self._client is None:
            logger.warning("Redis client not connected")
            return

        try:
            full_key = self._make_key(key)
            ttl = ttl_seconds or self._default_ttl
            json_data = result.model_dump_json()
            await self._client.set(full_key, json_data, ex=ttl)
            logger.debug(f"Cached result for key {key} with TTL {ttl}s")

        except Exception as e:
            logger.error(f"Cache set failed: {e}")

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached entry.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if entry was found and removed.
        """
        if self._client is None:
            return False

        try:
            full_key = self._make_key(key)
            deleted = await self._client.delete(full_key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Cache invalidate failed: {e}")
            return False

    async def clear(self) -> int:
        """
        Clear all cached entries (flush DB).

        Returns:
            Number of keys in DB (rough estimate).
        """
        if self._client is None:
            return 0

        try:
            # Flush entire database to ensure clean slate (request, response, claims, traces)
            await self._client.flushdb()
            logger.info("Flushed Redis database (all keys cleared)")
            return -1  # precise count unknown after flush

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0

    async def close(self) -> None:
        """Alias for disconnect."""
        await self.disconnect()


    # -------------------------------------------------------------------------
    # Claim-Level Caching
    # -------------------------------------------------------------------------

    def _make_claim_key(self, claim_hash: str) -> str:
        """Create full cache key for claim-level cache."""
        return f"{CLAIM_CACHE_PREFIX}{claim_hash}"

    async def get_claim(
        self,
        claim_hash: str,
    ) -> ClaimVerification | None:
        """
        Retrieve cached claim verification by claim hash.

        Args:
            claim_hash: Deterministic hash of the claim text.

        Returns:
            Cached ClaimVerification or None if not found/expired.
        """
        if self._client is None:
            logger.warning("Redis client not connected")
            return None

        try:
            full_key = self._make_claim_key(claim_hash)
            data = await self._client.get(full_key)

            if data is None:
                return None

            json_data = json.loads(data)
            logger.debug(f"Cache hit for claim hash: {claim_hash}")
            return ClaimVerification.model_validate(json_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize cached claim: {e}")
            return None
        except Exception as e:
            logger.error(f"Claim cache get failed: {e}")
            return None

    async def set_claim(
        self,
        claim_hash: str,
        verification: ClaimVerification,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Store claim verification result in cache.

        Args:
            claim_hash: Deterministic hash of the claim text.
            verification: ClaimVerification result to cache.
            ttl_seconds: Time-to-live in seconds (None = use default).
        """
        if self._client is None:
            logger.warning("Redis client not connected")
            return

        try:
            full_key = self._make_claim_key(claim_hash)
            ttl = ttl_seconds or self._claim_ttl
            json_data = verification.model_dump_json()
            await self._client.set(full_key, json_data, ex=ttl)
            logger.debug(f"Cached claim verification for hash: {claim_hash}")

        except Exception as e:
            logger.error(f"Claim cache set failed: {e}")

    async def get_claims_batch(
        self,
        claim_hashes: list[str],
    ) -> dict[str, ClaimVerification | None]:
        """
        Retrieve multiple cached claim verifications in a single batch.

        Uses Redis MGET for efficient batch retrieval.

        Args:
            claim_hashes: List of claim hashes to look up.

        Returns:
            Dictionary mapping claim hash to ClaimVerification (or None if not cached).
        """
        if self._client is None or not claim_hashes:
            return {claim_hash: None for claim_hash in claim_hashes}

        try:
            # Compute all keys
            keys_map = {self._make_claim_key(claim_hash): claim_hash for claim_hash in claim_hashes}
            keys = list(keys_map.keys())

            # Batch fetch with MGET
            values = await self._client.mget(keys)

            result: dict[str, ClaimVerification | None] = {}
            for key, value in zip(keys, values, strict=True):
                claim_hash = keys_map[key]
                if value is None:
                    result[claim_hash] = None
                else:
                    try:
                        json_data = json.loads(value)
                        result[claim_hash] = ClaimVerification.model_validate(json_data)
                    except Exception:
                        result[claim_hash] = None

            return result

        except Exception as e:
            logger.error(f"Batch claim cache get failed: {e}")
            return {claim_hash: None for claim_hash in claim_hashes}

    async def set_claims_batch(
        self,
        verifications: list[tuple[str, ClaimVerification]],
        ttl_seconds: int | None = None,
    ) -> int:
        """
        Store multiple claim verifications in cache using pipeline.

        Args:
            verifications: List of (claim_hash, ClaimVerification) tuples.
            ttl_seconds: Time-to-live in seconds (None = use default).

        Returns:
            Number of claims successfully cached.
        """
        if self._client is None or not verifications:
            return 0

        try:
            ttl = ttl_seconds or self._claim_ttl
            pipe = self._client.pipeline()

            for claim_hash, verification in verifications:
                full_key = self._make_claim_key(claim_hash)
                json_data = verification.model_dump_json()
                pipe.set(full_key, json_data, ex=ttl)

            await pipe.execute()
            logger.debug(f"Batch cached {len(verifications)} claim verifications")
            return len(verifications)

        except Exception as e:
            logger.error(f"Batch claim cache set failed: {e}")
            return 0

    async def invalidate_claim(self, claim_hash: str) -> bool:
        """
        Invalidate a cached claim verification.

        Args:
            claim_hash: Claim hash to invalidate.

        Returns:
            True if entry was found and removed.
        """
        if self._client is None:
            return False

        try:
            full_key = self._make_claim_key(claim_hash)
            deleted = await self._client.delete(full_key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Claim cache invalidate failed: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics including claim-level cache.

        Returns:
            Dictionary with cache stats.
        """
        if self._client is None:
            return {"connected": False}

        try:
            info = await self._client.info("stats")
            memory = await self._client.info("memory")

            result_count = 0
            async for _ in self._client.scan_iter(match=f"{CACHE_PREFIX}*"):
                result_count += 1

            claim_count = 0
            async for _ in self._client.scan_iter(match=f"{CLAIM_CACHE_PREFIX}*"):
                claim_count += 1

            return {
                "connected": True,
                "cached_results": result_count,
                "cached_claims": claim_count,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "used_memory": memory.get("used_memory_human", "unknown"),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"connected": True, "error": str(e)}
