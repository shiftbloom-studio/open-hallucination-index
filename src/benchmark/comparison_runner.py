"""
Comparison Benchmark Runner
============================

This module provides backward compatibility imports.
The actual implementation has been refactored into the `benchmark.runner` package.

For new code, prefer importing directly from `benchmark.runner`:

    from benchmark.runner import (
        ComparisonBenchmarkRunner,
        run_comparison_benchmark,
        LiveStats,
        LiveBenchmarkDisplay,
    )

This file re-exports all public symbols for backward compatibility.
"""

from __future__ import annotations

# Re-export everything from the new package
from benchmark.runner import (
    ComparisonBenchmarkRunner,
    LiveBenchmarkDisplay,
    LiveStats,
    run_comparison_benchmark,
)

# Also export internal types for any code that relied on them
from benchmark.runner._types import COLORS, BenchmarkContext, ColorScheme, TaskResult

__all__ = [
    # Primary exports
    "ComparisonBenchmarkRunner",
    "run_comparison_benchmark",
    "LiveStats",
    "LiveBenchmarkDisplay",
    # Type exports
    "COLORS",
    "ColorScheme",
    "BenchmarkContext",
    "TaskResult",
]
