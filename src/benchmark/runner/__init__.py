"""
Comparison Benchmark Runner
============================

Modular benchmark execution framework for evaluating hallucination detection systems.

This package provides:
- Real-time live progress display with Rich
- Concurrent async task execution with proper UI updates
- Multiple benchmark modes (standard, strategy comparison, cache testing)
- Pluggable evaluator architecture
- Buffered logging to prevent display corruption

Optimized for:
- Docker exec environments
- Git Bash within VS Code
- Standard terminal emulators

Public API:
    ComparisonBenchmarkRunner: Main orchestrator class
    run_comparison_benchmark: Convenience async function
    LiveStats: Real-time statistics dataclass
    LiveBenchmarkDisplay: Rich-based live display
    create_optimized_console: Console factory for Docker/Git Bash
    install_buffered_logging: Install log handler for display integration
    uninstall_buffered_logging: Cleanup log handler

Example:
    ```python
    from benchmark.runner import run_comparison_benchmark

    report = await run_comparison_benchmark()
    print(f"Winner: {report.get_ranking('f1_score')[0]}")
    ```
"""

from benchmark.runner.runner import (
    ComparisonBenchmarkRunner,
    run_comparison_benchmark,
)
from benchmark.runner._types import LiveStats
from benchmark.runner._display import (
    LiveBenchmarkDisplay,
    create_optimized_console,
    install_buffered_logging,
    uninstall_buffered_logging,
)

__all__ = [
    "ComparisonBenchmarkRunner",
    "run_comparison_benchmark",
    "LiveStats",
    "LiveBenchmarkDisplay",
    "create_optimized_console",
    "install_buffered_logging",
    "uninstall_buffered_logging",
]
