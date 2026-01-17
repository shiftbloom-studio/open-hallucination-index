"""
Async task executor with UI-aware waiting.

Provides utilities for running concurrent async tasks while
keeping the UI responsive through periodic display updates.

Key Features:
- Uses `asyncio.wait()` with timeout for interruptible waiting
- Forces UI refresh during long-running operations
- Proper semaphore-based concurrency control
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from benchmark.runner._display import LiveBenchmarkDisplay

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def execute_with_display_updates(
    tasks: list[Awaitable[T]],
    display: LiveBenchmarkDisplay,
    poll_interval: float = 2.0,  # Increased to reduce jitter
) -> list[T]:
    """
    Execute async tasks while keeping the display updated.

    Uses `asyncio.wait()` with timeout to periodically yield control
    back to the display, preventing UI freezes during long operations.

    Args:
        tasks: List of awaitable tasks to execute
        display: LiveBenchmarkDisplay instance to update
        poll_interval: Seconds between forced display updates

    Returns:
        List of results in completion order (not submission order)

    Example:
        ```python
        async def process(item):
            return await api.verify(item)

        tasks = [process(item) for item in items]
        results = await execute_with_display_updates(tasks, display)
        ```
    """
    results: list[T] = []
    pending: set[asyncio.Task[T]] = {
        asyncio.ensure_future(t) if not isinstance(t, asyncio.Task) else t for t in tasks
    }

    while pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=poll_interval,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Force display refresh even if no tasks completed
        display.force_refresh()

        # Collect results from completed tasks
        for future in done:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Task failed: {e}")
                raise

    return results


async def execute_with_callback(
    tasks: list[Awaitable[T]],
    display: LiveBenchmarkDisplay,
    on_complete: Callable[[T], None],
    poll_interval: float = 0.5,
) -> list[T]:
    """
    Execute async tasks with per-result callback and display updates.

    Similar to `execute_with_display_updates` but calls a callback
    for each completed result, allowing incremental processing.

    Args:
        tasks: List of awaitable tasks
        display: LiveBenchmarkDisplay instance
        on_complete: Callback invoked for each completed result
        poll_interval: Seconds between forced display updates

    Returns:
        List of all results in completion order
    """
    results: list[T] = []
    pending: set[asyncio.Task[T]] = {
        asyncio.ensure_future(t) if not isinstance(t, asyncio.Task) else t for t in tasks
    }

    while pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=poll_interval,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Force display refresh
        display.force_refresh()

        # Process completed tasks
        for future in done:
            try:
                result = future.result()
                results.append(result)
                on_complete(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                raise

    return results


def create_semaphore_wrapper(
    semaphore: asyncio.Semaphore,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Create a decorator that wraps async functions with semaphore acquisition.

    Args:
        semaphore: Semaphore for concurrency control

    Returns:
        Decorator function

    Example:
        ```python
        sem = asyncio.Semaphore(5)
        wrap = create_semaphore_wrapper(sem)

        @wrap
        async def limited_call(x):
            return await api.call(x)
        ```
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator
