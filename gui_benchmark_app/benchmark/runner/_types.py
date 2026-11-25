"""
Type definitions and data structures for the benchmark runner.

This module contains pure data classes with no external dependencies
beyond the standard library, enabling clean separation of concerns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColorScheme:
    """
    Consistent color scheme for console output.

    Matches ConsoleReporter styling for visual consistency.
    """

    good: str = "green"
    warn: str = "yellow"
    bad: str = "red"
    dim: str = "dim"
    cyan: str = "cyan"
    accent: str = "bright_magenta"


# Global color scheme instance
COLORS = ColorScheme()


@dataclass
class LiveStats:
    """
    Real-time benchmark statistics tracked during execution.

    This dataclass serves as the single source of truth for all
    progress tracking, shared between the display and executor.

    Attributes:
        total_evaluators: Number of evaluators to benchmark
        completed_evaluators: Number of evaluators completed
        current_evaluator: Name of currently running evaluator
        current_metric: Name of current benchmark metric
        current_total: Total items in current task
        current_completed: Completed items in current task
        current_correct: Correct predictions in current task
        current_errors: Errors in current task
        start_time: Benchmark start timestamp (perf_counter)
        current_latencies: Latencies for current task
        correct: Total correct predictions across all tasks
        errors: Total errors across all tasks
        total_processed: Total items processed across all tasks
        evaluator_results: Per-evaluator metrics for display
    """

    # Overall progress
    total_evaluators: int = 0
    completed_evaluators: int = 0
    current_evaluator: str = ""
    current_metric: str = ""

    # Current task progress
    current_total: int = 0
    current_completed: int = 0
    current_correct: int = 0
    current_errors: int = 0

    # Timing
    start_time: float = field(default_factory=time.perf_counter)
    current_latencies: list[float] = field(default_factory=list)

    # Accumulated results
    correct: int = 0
    errors: int = 0
    total_processed: int = 0

    # Per-evaluator results for display
    evaluator_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def reset_task(self) -> None:
        """Reset task-level counters for a new task."""
        self.current_total = 0
        self.current_completed = 0
        self.current_correct = 0
        self.current_errors = 0
        self.current_latencies = []


@dataclass
class TaskResult:
    """
    Result from a single benchmark task execution.

    Generic container for any benchmark result that needs
    latency tracking and error handling.
    """

    latency_ms: float
    is_correct: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkContext:
    """
    Execution context passed to benchmark methods.

    Encapsulates all dependencies needed by benchmark execution,
    enabling clean function signatures and testability.
    """

    max_latency_ms: float
    concurrency: int
    warmup_requests: int = 3
