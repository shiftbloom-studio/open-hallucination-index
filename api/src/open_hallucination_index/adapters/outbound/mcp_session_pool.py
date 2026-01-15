"""
MCP Session Pool - Simplified
=============================

Simple, fast connection pooling for MCP servers.
Creates sessions on-demand with semaphore-based concurrency control.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPTransportType(Enum):
    """Supported MCP transport types."""
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class PoolConfig:
    """Pool configuration with sensible defaults."""
    min_sessions: int = 0
    max_sessions: int = 3
    session_ttl_seconds: float = 300.0
    idle_timeout_seconds: float = 60.0
    health_check_interval_seconds: float = 30.0
    session_init_timeout_seconds: float = 30.0


class MCPSessionPool:
    """
    Simple MCP session pool.
    
    Design: Fresh session per request with semaphore limiting concurrency.
    Avoids SSE keep-alive complexity while providing clean, fast API.
    """

    def __init__(
        self,
        source_name: str,
        mcp_url: str,
        transport_type: MCPTransportType = MCPTransportType.SSE,
        config: PoolConfig | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._source_name = source_name
        self._mcp_url = mcp_url
        self._transport_type = transport_type
        self._config = config or PoolConfig()
        self._headers = headers or {}
        
        self._semaphore = asyncio.Semaphore(self._config.max_sessions)
        self._initialized = False
        self._is_healthy = True
        self._stats = {"created": 0, "acquired": 0, "errors": 0}

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    async def initialize(self) -> None:
        """Mark pool as ready."""
        self._initialized = True
        self._is_healthy = True
        logger.info(f"MCP pool for {self._source_name} initialized")

    async def shutdown(self) -> None:
        """Shutdown the pool."""
        self._initialized = False
        self._is_healthy = False
        logger.info(f"MCP pool for {self._source_name} shutdown")

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire an MCP session.
        
        Creates fresh SSE session, yields it, then closes.
        Semaphore limits concurrent connections.
        """
        if not self._initialized:
            raise RuntimeError(f"Pool for {self._source_name} not initialized")

        async with self._semaphore:
            self._stats["acquired"] += 1
            
            try:
                async with asyncio.timeout(self._config.session_init_timeout_seconds):
                    async with sse_client(self._mcp_url, headers=self._headers) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            self._stats["created"] += 1
                            self._is_healthy = True
                            yield session
                            
            except TimeoutError:
                self._stats["errors"] += 1
                self._is_healthy = False
                raise RuntimeError(f"Timeout connecting to {self._source_name}") from None
            except Exception as e:
                self._stats["errors"] += 1
                logger.warning(f"MCP error for {self._source_name}: {e}")
                raise

    def get_stats(self) -> dict[str, Any]:
        """Return pool statistics."""
        return {
            "source": self._source_name,
            "healthy": self._is_healthy,
            **self._stats,
        }
