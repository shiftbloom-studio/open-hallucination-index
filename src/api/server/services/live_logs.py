"""
Live Logging Service
====================

Service for capturing and broadcasting API activity in real-time.
Supports multiple SSE clients for the admin dashboard.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LogType(str, Enum):
    """Types of log entries."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    HEALTH = "health"
    AUTH = "auth"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """A single log entry."""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    level: LogLevel = LogLevel.INFO
    log_type: LogType = LogType.REQUEST
    method: str = ""
    path: str = ""
    status_code: int | None = None
    duration_ms: float | None = None
    user_id: str | None = None
    key_prefix: str | None = None
    message: str = ""
    details: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "type": self.log_type.value,
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
            "user_id": self.user_id,
            "key_prefix": self.key_prefix,
            "message": self.message,
            "details": self.details,
        }

    def format_message(self) -> str:
        """Format as a human-readable log line."""
        parts = []

        if self.method and self.path:
            parts.append(f"{self.method} {self.path}")

        if self.status_code:
            parts.append(f"- {self.status_code}")

        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.0f}ms)")

        if self.key_prefix:
            parts.append(f"[{self.key_prefix}...]")

        if self.message and not (self.method and self.path):
            parts.append(self.message)

        return " ".join(parts) if parts else self.message


class LiveLogService:
    """
    Service for managing live log broadcasting.

    Maintains a buffer of recent logs and broadcasts new entries
    to all connected SSE clients.
    """

    _instance: LiveLogService | None = None

    def __new__(cls) -> LiveLogService:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._log_buffer: deque[LogEntry] = deque(maxlen=500)  # Keep last 500 logs
        self._subscribers: set[asyncio.Queue[LogEntry]] = set()
        self._lock = asyncio.Lock()

        # Stats
        self._total_requests = 0
        self._total_errors = 0
        self._start_time = datetime.now(UTC)

        logger.info("LiveLogService initialized")

    async def add_log(self, entry: LogEntry) -> None:
        """Add a new log entry and broadcast to subscribers."""
        async with self._lock:
            self._log_buffer.append(entry)

            # Update stats
            if entry.log_type == LogType.REQUEST:
                self._total_requests += 1
            if entry.level == LogLevel.ERROR:
                self._total_errors += 1

            # Broadcast to all subscribers
            dead_subscribers = []
            for queue in self._subscribers:
                try:
                    queue.put_nowait(entry)
                except asyncio.QueueFull:
                    # Queue is full, subscriber is too slow
                    dead_subscribers.append(queue)

            # Remove dead subscribers
            for queue in dead_subscribers:
                self._subscribers.discard(queue)

    async def subscribe(self) -> AsyncGenerator[LogEntry]:
        """Subscribe to live log updates."""
        queue: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=100)

        async with self._lock:
            self._subscribers.add(queue)

        try:
            # First, send recent logs from buffer
            for entry in list(self._log_buffer)[-50:]:  # Last 50 entries
                yield entry

            # Then stream new logs
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield entry
                except TimeoutError:
                    # Send heartbeat
                    yield LogEntry(
                        level=LogLevel.DEBUG,
                        log_type=LogType.SYSTEM,
                        message="heartbeat",
                    )
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    def get_recent_logs(self, limit: int = 100) -> list[LogEntry]:
        """Get recent logs from buffer."""
        return list(self._log_buffer)[-limit:]

    def get_stats(self) -> dict:
        """Get logging statistics."""
        uptime = (datetime.now(UTC) - self._start_time).total_seconds()
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "buffer_size": len(self._log_buffer),
            "active_subscribers": len(self._subscribers),
            "uptime_seconds": uptime,
            "requests_per_minute": (self._total_requests / uptime * 60) if uptime > 0 else 0,
        }

    # Convenience methods for logging different event types

    async def log_request(
        self,
        method: str,
        path: str,
        user_id: str | None = None,
        key_prefix: str | None = None,
    ) -> str:
        """Log an incoming request. Returns request ID for correlation."""
        entry = LogEntry(
            level=LogLevel.INFO,
            log_type=LogType.REQUEST,
            method=method,
            path=path,
            user_id=user_id,
            key_prefix=key_prefix,
            message=f"Request started: {method} {path}",
        )
        await self.add_log(entry)
        return entry.id

    async def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: str | None = None,
        key_prefix: str | None = None,
    ) -> None:
        """Log a response."""
        level = LogLevel.INFO
        if status_code >= 500:
            level = LogLevel.ERROR
        elif status_code >= 400:
            level = LogLevel.WARNING

        entry = LogEntry(
            level=level,
            log_type=LogType.RESPONSE,
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            key_prefix=key_prefix,
        )
        await self.add_log(entry)

    async def log_error(
        self,
        message: str,
        path: str = "",
        details: dict | None = None,
    ) -> None:
        """Log an error."""
        entry = LogEntry(
            level=LogLevel.ERROR,
            log_type=LogType.ERROR,
            path=path,
            message=message,
            details=details,
        )
        await self.add_log(entry)

    async def log_health_check(self, status: str, duration_ms: float) -> None:
        """Log a health check."""
        entry = LogEntry(
            level=LogLevel.DEBUG,
            log_type=LogType.HEALTH,
            path="/health",
            status_code=200 if status == "healthy" else 503,
            duration_ms=duration_ms,
            message=f"Health check: {status}",
        )
        await self.add_log(entry)

    async def log_auth(
        self,
        success: bool,
        key_prefix: str | None = None,
        reason: str = "",
    ) -> None:
        """Log an authentication attempt."""
        entry = LogEntry(
            level=LogLevel.INFO if success else LogLevel.WARNING,
            log_type=LogType.AUTH,
            key_prefix=key_prefix,
            message=f"Auth {'success' if success else 'failed'}: {reason}" if reason else f"Auth {'success' if success else 'failed'}",
        )
        await self.add_log(entry)

    async def log_system(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log a system message."""
        entry = LogEntry(
            level=level,
            log_type=LogType.SYSTEM,
            message=message,
        )
        await self.add_log(entry)


# Global singleton instance
live_log_service = LiveLogService()
