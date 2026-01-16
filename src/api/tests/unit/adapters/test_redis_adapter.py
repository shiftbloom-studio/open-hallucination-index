"""Unit tests for Redis cache adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.redis_cache import RedisCacheAdapter


@pytest.fixture
def mock_settings():
    """Mock Redis settings."""
    settings = MagicMock()
    settings.host = "localhost"
    settings.port = 6379
    settings.db = 0
    settings.password = None
    settings.socket_path = None
    settings.max_connections = 10
    settings.cache_ttl_seconds = 3600
    settings.claim_cache_ttl_seconds = 7200
    return settings


@pytest.fixture
def mock_redis_client():
    """Mock Redis client with async methods."""
    client = MagicMock()
    # Redis methods need to be AsyncMock for await to work
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=1)
    client.ping = AsyncMock(return_value=True)
    client.close = AsyncMock()
    client.mget = AsyncMock(return_value=[])
    return client


@pytest.fixture
def redis_adapter(mock_settings, mock_redis_client):
    """Redis adapter with mocked client."""
    adapter = RedisCacheAdapter(mock_settings)
    adapter._client = mock_redis_client
    return adapter


class TestRedisCacheAdapter:
    """Test RedisCacheAdapter."""

    def test_initialization(self, redis_adapter: RedisCacheAdapter):
        """Test adapter initialization."""
        assert redis_adapter is not None
        assert redis_adapter._client is not None

    @pytest.mark.asyncio
    async def test_get_existing_key(self, redis_adapter: RedisCacheAdapter):
        """Test getting existing cached value."""
        # Valid JSON that matches VerificationResult schema
        valid_json = b'{"id": "550e8400-e29b-41d4-a716-446655440000", "input_hash": "hash", "input_length": 10, "trust_score": {"overall": 0.8, "claims_total": 1, "claims_supported": 1, "claims_refuted": 0, "claims_unverifiable": 0, "confidence": 0.9, "scoring_method": "test"}, "claim_verifications": [], "summary": "Test", "processing_time_ms": 100.0, "cached": false}'
        redis_adapter._client.get.return_value = valid_json

        result = await redis_adapter.get("test_key")

        assert result is not None
        assert result.trust_score.overall == 0.8
        redis_adapter._client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, redis_adapter: RedisCacheAdapter):
        """Test getting nonexistent key."""
        redis_adapter._client.get.return_value = None

        result = await redis_adapter.get("nonexistent_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_key(self, redis_adapter: RedisCacheAdapter):
        """Test deleting a key."""
        await redis_adapter.delete("test_key")

        redis_adapter._client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_success(self, redis_adapter: RedisCacheAdapter):
        """Test health check when Redis is reachable."""
        redis_adapter._client.ping.return_value = True

        result = await redis_adapter.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_no_client(self, mock_settings):
        """Test health check when client is not connected."""
        adapter = RedisCacheAdapter(mock_settings)
        adapter._client = None

        result = await adapter.health_check()

        assert result is False


class TestRedisCacheErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_get_error_returns_none(self, redis_adapter: RedisCacheAdapter):
        """Test that get errors return None instead of raising."""
        redis_adapter._client.get.side_effect = Exception("Connection failed")

        # The adapter catches exceptions and returns None
        result = await redis_adapter.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self, redis_adapter: RedisCacheAdapter):
        """Test that invalid JSON in cache returns None."""
        redis_adapter._client.get.return_value = b'invalid json'

        result = await redis_adapter.get("test_key")

        assert result is None
