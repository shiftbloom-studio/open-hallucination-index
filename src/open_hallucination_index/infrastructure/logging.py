"""
Logging utilities for API runtime.
"""

from __future__ import annotations

import logging
import time


class HealthLiveAccessFilter(logging.Filter):
    """Throttle /health/live access log entries to reduce log noise."""

    def __init__(self, min_interval_seconds: float = 120.0) -> None:
        super().__init__()
        self._min_interval_seconds = min_interval_seconds
        self._last_logged: float | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "/health/live" not in message:
            return True

        now = time.monotonic()
        if self._last_logged is None or (now - self._last_logged) >= self._min_interval_seconds:
            self._last_logged = now
            return True

        return False
