"""Unit tests for Redis cache adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from open_hallucination_index.adapters.redis_cache import RedisCache


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    return MagicMock()


@pytest.fixture
def redis_cache(mock_redis_client):
    """Redis cache with mocked client."""
    with patch("redis.Redis", return_value=mock_redis_client):
        cache = RedisCache(
            host="localhost",
            port=6379,
            db=0,
            ttl=3600,
        )
        cache.client = mock_redis_client
        return cache


class TestRedisCache:
    """Test RedisCache adapter."""

    def test_initialization(self, redis_cache: RedisCache):
        """Test cache initialization."""
        assert redis_cache is not None
        assert redis_cache.client is not None

    @pytest.mark.asyncio
    async def test_get_existing_key(self, redis_cache: RedisCache):
        """Test getting existing cached value."""
        test_value = {"result": "cached_data"}
        redis_cache.client.get.return_value = b'{"result": "cached_data"}'
        
        result = await redis_cache.get("test_key")
        
        assert result is not None
        redis_cache.client.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, redis_cache: RedisCache):
        """Test getting nonexistent key."""
        redis_cache.client.get.return_value = None
        
        result = await redis_cache.get("nonexistent_key")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_value(self, redis_cache: RedisCache):
        """Test setting a cache value."""
        test_value = {"result": "new_data"}
        
        await redis_cache.set("test_key", test_value)
        
        redis_cache.client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, redis_cache: RedisCache):
        """Test setting value with custom TTL."""
        test_value = {"data": "test"}
        
        await redis_cache.set("test_key", test_value, ttl=7200)
        
        redis_cache.client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_key(self, redis_cache: RedisCache):
        """Test deleting a key."""
        await redis_cache.delete("test_key")
        
        redis_cache.client.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists_key(self, redis_cache: RedisCache):
        """Test checking key existence."""
        redis_cache.client.exists.return_value = 1
        
        result = await redis_cache.exists("test_key")
        
        assert result is True
        redis_cache.client.exists.assert_called_once_with("test_key")


class TestRedisCacheErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error(self, redis_cache: RedisCache):
        """Test handling of connection errors."""
        redis_cache.client.get.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            await redis_cache.get("test_key")

    @pytest.mark.asyncio
    async def test_set_error(self, redis_cache: RedisCache):
        """Test handling of set errors."""
        redis_cache.client.set.side_effect = Exception("Set failed")
        
        with pytest.raises(Exception):
            await redis_cache.set("test_key", {"data": "test"})

    def test_ping(self, redis_cache: RedisCache):
        """Test ping functionality."""
        redis_cache.client.ping.return_value = True
        
        result = redis_cache.client.ping()
        
        assert result is True
