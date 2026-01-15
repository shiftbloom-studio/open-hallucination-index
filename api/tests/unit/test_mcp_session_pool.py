"""
Tests for MCP Session Pool
===========================
"""

import time
import threading
from unittest.mock import MagicMock

import pytest

from open_hallucination_index.adapters.outbound.mcp_session_pool import (
    MCPPoolManager,
    PooledSession,
    MCPTransportType,
    SESSION_VALIDATION_TIMEOUT_SECONDS,
)


class TestPooledSession:
    """Tests for the PooledSession dataclass."""

    def test_has_expired_with_positive_ttl(self) -> None:
        """Test that has_expired returns True when TTL is exceeded."""
        # Create a session and backdate its creation time
        session = PooledSession(
            session=MagicMock(),
            transport_type=MCPTransportType.SSE,
            source_name="test",
        )
        # Backdate the session by 10 seconds
        session.created_at = time.time() - 10.0
        
        # Test with TTL of 5 seconds - should be expired
        assert session.has_expired(5.0) is True
        
        # Test with TTL of 15 seconds - should not be expired
        assert session.has_expired(15.0) is False

    def test_has_expired_with_zero_ttl(self) -> None:
        """Test that has_expired returns False when TTL is zero (disabled)."""
        session = PooledSession(
            session=MagicMock(),
            transport_type=MCPTransportType.SSE,
            source_name="test",
        )
        # Even if the session is old, TTL of 0 means no expiry
        session.created_at = time.time() - 1000.0
        
        assert session.has_expired(0.0) is False

    def test_has_expired_with_negative_ttl(self) -> None:
        """Test that has_expired returns False when TTL is negative (disabled)."""
        session = PooledSession(
            session=MagicMock(),
            transport_type=MCPTransportType.SSE,
            source_name="test",
        )
        session.created_at = time.time() - 1000.0
        
        assert session.has_expired(-1.0) is False


class TestMCPPoolManager:
    """Tests for the MCPPoolManager singleton."""

    def test_singleton_instance(self) -> None:
        """Test that get_instance returns the same instance."""
        manager1 = MCPPoolManager.get_instance()
        manager2 = MCPPoolManager.get_instance()
        
        assert manager1 is manager2

    def test_singleton_thread_safety(self) -> None:
        """Test that singleton is thread-safe."""
        # Reset the singleton for this test
        MCPPoolManager._instance = None
        
        instances = []
        
        def get_instance_threaded():
            instances.append(MCPPoolManager.get_instance())
        
        # Create multiple threads that try to get the instance
        threads = [threading.Thread(target=get_instance_threaded) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All instances should be the same object
        assert len(set(id(instance) for instance in instances)) == 1


class TestConstants:
    """Tests for module constants."""

    def test_session_validation_timeout_constant(self) -> None:
        """Test that SESSION_VALIDATION_TIMEOUT_SECONDS constant is defined."""
        assert SESSION_VALIDATION_TIMEOUT_SECONDS == 10.0
