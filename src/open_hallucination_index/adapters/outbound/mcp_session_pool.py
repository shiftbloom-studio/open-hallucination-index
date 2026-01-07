"""
MCP Session Pool Manager
========================

Implements persistent SSE/HTTP connections for MCP servers (Wikipedia, Context7).
Manages connection pooling, health monitoring, and automatic reconnection.

Key features:
- Session reuse across requests (avoids SSE handshake overhead)
- Configurable pool size per MCP source
- Automatic health checks and reconnection
- Graceful degradation on connection failures
- Thread-safe async operations with semaphores
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable
from uuid import uuid4

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

if TYPE_CHECKING:
    from open_hallucination_index.infrastructure.config import MCPSettings

logger = logging.getLogger(__name__)


class MCPTransportType(Enum):
    """Supported MCP transport types."""
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class PooledSession:
    """A pooled MCP session with metadata."""
    session: ClientSession
    transport_type: MCPTransportType
    source_name: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    request_count: int = 0
    is_healthy: bool = True
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Lifecycle management via worker task
    _worker_task: asyncio.Task | None = None
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _init_event: asyncio.Event = field(default_factory=asyncio.Event)
    _worker_error: Exception | None = None

    def mark_used(self) -> None:
        """Update usage tracking."""
        self.last_used_at = time.time()
        self.request_count += 1

    @property
    def age_seconds(self) -> float:
        """Return session age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Return seconds since last use."""
        return time.time() - self.last_used_at


@dataclass
class PoolConfig:
    """Configuration for session pool."""
    min_sessions: int = 1
    max_sessions: int = 5
    session_ttl_seconds: float = 300.0  # 5 minutes
    idle_timeout_seconds: float = 60.0  # 1 minute
    health_check_interval_seconds: float = 30.0
    reconnect_delay_seconds: float = 1.0
    max_reconnect_attempts: int = 3


class MCPSessionPool:
    """
    Pool manager for MCP sessions.
    
    Maintains persistent SSE/HTTP connections to MCP servers,
    reusing sessions across requests for improved performance.
    """

    def __init__(
        self,
        source_name: str,
        mcp_url: str,
        transport_type: MCPTransportType,
        config: PoolConfig | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize session pool for an MCP source.
        
        Args:
            source_name: Human-readable name (e.g., "Wikipedia", "Context7").
            mcp_url: Full MCP endpoint URL.
            transport_type: SSE or Streamable HTTP transport.
            config: Pool configuration.
            headers: Optional HTTP headers (e.g., for auth).
        """
        self._source_name = source_name
        self._mcp_url = mcp_url
        self._transport_type = transport_type
        self._config = config or PoolConfig()
        self._headers = headers or {}
        
        # Pool state
        self._sessions: list[PooledSession] = []
        self._available: asyncio.Queue[PooledSession] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._shutting_down = False
        
        # Health monitoring
        self._health_check_task: asyncio.Task | None = None
        self._is_healthy = False
        self._last_error: str | None = None
        
        # Metrics
        self._total_requests = 0
        self._total_reuses = 0
        self._total_creates = 0
        self._total_errors = 0

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy and len(self._sessions) > 0

    @property
    def pool_size(self) -> int:
        return len(self._sessions)

    @property
    def available_sessions(self) -> int:
        return self._available.qsize()

    async def initialize(self) -> None:
        """
        Initialize the pool with minimum sessions.
        
        Creates initial connections and starts health monitoring.
        """
        if self._initialized:
            return
            
        logger.info(f"Initializing MCP session pool for {self._source_name}")
        
        async with self._lock:
            # Create minimum number of sessions
            for _ in range(self._config.min_sessions):
                try:
                    session = await self._create_session()
                    if session:
                        self._sessions.append(session)
                        await self._available.put(session)
                        self._is_healthy = True
                except Exception as e:
                    logger.warning(f"Failed to create initial session for {self._source_name}: {e}")
                    self._last_error = str(e)
            
            self._initialized = True
        
        # Start background health check
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(
            f"MCP pool for {self._source_name} initialized with "
            f"{len(self._sessions)} sessions"
        )

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the pool.
        
        Closes all sessions and stops health monitoring.
        """
        logger.info(f"Shutting down MCP session pool for {self._source_name}")
        self._shutting_down = True
        
        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        async with self._lock:
            for pooled in self._sessions:
                await self._close_session(pooled)
            self._sessions.clear()
            
            # Clear the queue
            while not self._available.empty():
                try:
                    self._available.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        self._initialized = False
        self._is_healthy = False
        logger.info(f"MCP pool for {self._source_name} shutdown complete")

    async def _create_session(self) -> PooledSession | None:
        """
        Create a new pooled session via a worker task.
        
        This ensures the context manager lifecycle (AnyIO scopes)
        stays within a single task context.
        """
        pooled = PooledSession(
            session=None,  # Will be set by worker
            transport_type=self._transport_type,
            source_name=self._source_name,
        )
        
        # Start worker task
        pooled._worker_task = asyncio.create_task(
            self._session_worker(pooled),
            name=f"MCP-Worker-{self._source_name}-{uuid4().hex[:8]}"
        )
        
        try:
            # Wait for session to be initialized or worker to fail
            async with asyncio.timeout(30.0):
                await pooled._init_event.wait()
            
            if pooled._worker_error:
                raise pooled._worker_error
                
            if pooled.session is None:
                raise RuntimeError("Worker initialized but session is None")
            
            self._total_creates += 1
            logger.debug(f"Created new session worker for {self._source_name}")
            return pooled
            
        except Exception as e:
            logger.error(f"Failed to create session worker for {self._source_name}: {e}")
            # Ensure worker is stopped
            pooled._stop_event.set()
            if pooled._worker_task:
                pooled._worker_task.cancel()
            
            self._total_errors += 1
            self._last_error = str(e)
            return None

    async def _session_worker(self, pooled: PooledSession) -> None:
        """
        Worker task that owns the MCP session context managers.
        """
        try:
            if self._transport_type == MCPTransportType.SSE:
                async with sse_client(self._mcp_url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        pooled.session = session
                        pooled._init_event.set()
                        # Keep alive until signaled to stop
                        await pooled._stop_event.wait()
            else:
                async with streamablehttp_client(self._mcp_url, headers=self._headers) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        pooled.session = session
                        pooled._init_event.set()
                        # Keep alive until signaled to stop
                        await pooled._stop_event.wait()
                        
        except asyncio.CancelledError:
            logger.debug(f"Session worker for {self._source_name} cancelled")
        except Exception as e:
            logger.warning(f"Session worker for {self._source_name} error: {e}")
            pooled._worker_error = e
            if not pooled._init_event.is_set():
                pooled._init_event.set()
        finally:
            pooled.session = None
            pooled.is_healthy = False
            logger.debug(f"Session worker for {self._source_name} stopped")

    async def _close_session(self, pooled: PooledSession) -> None:
        """Close a pooled session by signaling its worker task."""
        if not pooled:
            return
            
        try:
            # Signal stop
            pooled._stop_event.set()
            
            # Wait for worker to finish gracefully
            if pooled._worker_task and not pooled._worker_task.done():
                try:
                    async with asyncio.timeout(2.0):
                        await pooled._worker_task
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pooled._worker_task.cancel()
            
            logger.debug(f"Signaled closure for {self._source_name} session")
        except Exception as e:
            logger.warning(f"Error signaling session closure for {self._source_name}: {e}")

    async def _validate_session(self, pooled: PooledSession) -> bool:
        """
        Validate that a session is still healthy.
        
        Args:
            pooled: Session to validate.
            
        Returns:
            True if session is valid, False otherwise.
        """
        try:
            # Quick health check - list tools
            # We use a slightly longer timeout and don't immediately fail on first transient error
            async with asyncio.timeout(10.0):
                await pooled.session.list_tools()
            
            return True
            
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"Session validation failed for {self._source_name}: {e}")
            return False

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[ClientSession, None]:
        """
        Acquire a session from the pool.
        
        Context manager that returns a session to the pool when done.
        Creates new sessions if pool is exhausted (up to max_sessions).
        
        Yields:
            Active MCP ClientSession.
            
        Raises:
            RuntimeError: If pool is not initialized or no sessions available.
        """
        if not self._initialized:
            raise RuntimeError(f"Session pool for {self._source_name} not initialized")
        
        if self._shutting_down:
            raise RuntimeError(f"Session pool for {self._source_name} is shutting down")
        
        self._total_requests += 1
        pooled: PooledSession | None = None
        created_new = False
        
        try:
            # Try to get an available session
            try:
                pooled = self._available.get_nowait()
                self._total_reuses += 1
            except asyncio.QueueEmpty:
                # No available sessions, try to create one
                async with self._lock:
                    if len(self._sessions) < self._config.max_sessions:
                        pooled = await self._create_session()
                        if pooled:
                            self._sessions.append(pooled)
                            created_new = True
            
            # If still no session, wait for one to become available
            if pooled is None:
                try:
                    async with asyncio.timeout(10.0):
                        pooled = await self._available.get()
                        self._total_reuses += 1
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Timeout acquiring session for {self._source_name}")
            
            # Validate the session
            if not await self._validate_session(pooled):
                # Session is stale, close and create new
                await self._close_session(pooled)
                async with self._lock:
                    if pooled in self._sessions:
                        self._sessions.remove(pooled)
                
                pooled = await self._create_session()
                if pooled:
                    async with self._lock:
                        self._sessions.append(pooled)
                    created_new = True
                else:
                    raise RuntimeError(f"Failed to create session for {self._source_name}")
            
            pooled.mark_used()
            pooled.is_healthy = True
            
            yield pooled.session
            
        except Exception as e:
            self._total_errors += 1
            self._last_error = str(e)
            
            # Mark session as unhealthy
            if pooled:
                pooled.is_healthy = False
            
            raise
            
        finally:
            # Return session to pool if still healthy
            if pooled and pooled.is_healthy and not self._shutting_down:
                try:
                    await self._available.put(pooled)
                except Exception:
                    pass
            elif pooled:
                # Close unhealthy session
                await self._close_session(pooled)
                async with self._lock:
                    if pooled in self._sessions:
                        self._sessions.remove(pooled)

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)
                
                if self._shutting_down:
                    break
                
                # Check pool health
                await self._perform_health_check()
                
                # Clean up idle sessions beyond minimum
                await self._cleanup_idle_sessions()
                
                # Ensure minimum sessions exist
                await self._ensure_minimum_sessions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check error for {self._source_name}: {e}")

    async def _perform_health_check(self) -> None:
        """Perform health check on available sessions."""
        sessions_to_check: list[PooledSession] = []
        
        # Drain queue temporarily
        while not self._available.empty():
            try:
                sessions_to_check.append(self._available.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        healthy_count = 0
        for pooled in sessions_to_check:
            if await self._validate_session(pooled):
                healthy_count += 1
                await self._available.put(pooled)
            else:
                # Close unhealthy session
                await self._close_session(pooled)
                async with self._lock:
                    if pooled in self._sessions:
                        self._sessions.remove(pooled)
        
        self._is_healthy = healthy_count > 0
        
        if not self._is_healthy:
            logger.warning(f"No healthy sessions for {self._source_name}")

    async def _cleanup_idle_sessions(self) -> None:
        """Remove sessions that have been idle too long."""
        async with self._lock:
            current_count = len(self._sessions)
            if current_count <= self._config.min_sessions:
                return
            
            # Find idle sessions
            now = time.time()
            sessions_to_remove = []
            
            for pooled in self._sessions:
                if (
                    pooled.idle_seconds > self._config.idle_timeout_seconds
                    and current_count - len(sessions_to_remove) > self._config.min_sessions
                ):
                    sessions_to_remove.append(pooled)
            
            for pooled in sessions_to_remove:
                await self._close_session(pooled)
                self._sessions.remove(pooled)
                logger.debug(f"Removed idle session for {self._source_name}")

    async def _ensure_minimum_sessions(self) -> None:
        """Ensure minimum number of sessions exist with retry logic."""
        async with self._lock:
            while len(self._sessions) < self._config.min_sessions:
                if self._shutting_down:
                    break
                    
                pooled = await self._create_session()
                if pooled:
                    self._sessions.append(pooled)
                    await self._available.put(pooled)
                else:
                    # If creation fails, wait a bit before retrying in the NEXT loop iteration
                    logger.warning(f"Failed to replenish session pool for {self._source_name}, will retry in next health loop")
                    break

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "source_name": self._source_name,
            "is_healthy": self._is_healthy,
            "pool_size": len(self._sessions),
            "available_sessions": self._available.qsize(),
            "total_requests": self._total_requests,
            "total_reuses": self._total_reuses,
            "total_creates": self._total_creates,
            "total_errors": self._total_errors,
            "reuse_rate": (
                self._total_reuses / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
            "last_error": self._last_error,
        }


class MCPPoolManager:
    """
    Global manager for all MCP session pools.
    
    Provides a unified interface for managing pools for
    multiple MCP sources (Wikipedia, Context7, etc.).
    """
    
    _instance: MCPPoolManager | None = None
    
    def __init__(self) -> None:
        self._pools: dict[str, MCPSessionPool] = {}
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> MCPPoolManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = MCPPoolManager()
        return cls._instance
    
    async def initialize_from_settings(self, settings: MCPSettings) -> None:
        """
        Initialize pools from MCP settings.
        
        Args:
            settings: MCP configuration.
        """
        if self._initialized:
            return
        
        # Wikipedia pool (SSE transport)
        if settings.wikipedia_enabled:
            wikipedia_url = settings.wikipedia_url.rstrip("/")
            if not wikipedia_url.endswith("/sse"):
                wikipedia_url = f"{wikipedia_url}/sse"
            
            self._pools["wikipedia"] = MCPSessionPool(
                source_name="Wikipedia",
                mcp_url=wikipedia_url,
                transport_type=MCPTransportType.SSE,
                config=PoolConfig(
                    min_sessions=1,
                    max_sessions=3,
                    session_ttl_seconds=300.0,
                    idle_timeout_seconds=60.0,
                ),
            )
        
        # Context7 pool (Streamable HTTP transport)
        if settings.context7_enabled:
            context7_url = settings.context7_url.rstrip("/")
            if not context7_url.endswith("/mcp"):
                context7_url = f"{context7_url}/mcp"
            
            headers = {}
            if settings.context7_api_key:
                headers["Authorization"] = f"Bearer {settings.context7_api_key}"
            
            self._pools["context7"] = MCPSessionPool(
                source_name="Context7",
                mcp_url=context7_url,
                transport_type=MCPTransportType.STREAMABLE_HTTP,
                config=PoolConfig(
                    min_sessions=1,
                    max_sessions=3,
                    session_ttl_seconds=300.0,
                    idle_timeout_seconds=60.0,
                ),
                headers=headers,
            )
        
        # Initialize all pools
        for name, pool in self._pools.items():
            try:
                await pool.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize {name} pool: {e}")
        
        self._initialized = True
        logger.info(f"MCP Pool Manager initialized with {len(self._pools)} pools")
    
    async def shutdown(self) -> None:
        """Shutdown all pools."""
        for pool in self._pools.values():
            await pool.shutdown()
        self._pools.clear()
        self._initialized = False
        logger.info("MCP Pool Manager shutdown complete")
    
    def get_pool(self, source_name: str) -> MCPSessionPool | None:
        """Get a specific pool by source name."""
        return self._pools.get(source_name.lower())
    
    @asynccontextmanager
    async def acquire_session(
        self,
        source_name: str,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Acquire a session for a specific MCP source.
        
        Args:
            source_name: Source name (e.g., "wikipedia", "context7").
            
        Yields:
            Active MCP ClientSession.
        """
        pool = self._pools.get(source_name.lower())
        if pool is None:
            raise ValueError(f"No pool configured for {source_name}")
        
        async with pool.acquire() as session:
            yield session
    
    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all pools."""
        return {
            name: pool.get_stats()
            for name, pool in self._pools.items()
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if any pool is healthy."""
        return any(pool.is_healthy for pool in self._pools.values())


# Convenience function for getting the global pool manager
def get_mcp_pool_manager() -> MCPPoolManager:
    """Get the global MCP pool manager instance."""
    return MCPPoolManager.get_instance()
