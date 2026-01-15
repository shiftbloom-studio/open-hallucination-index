#!/usr/bin/env python3
"""
OHI Benchmark Suite
===================

Research-grade benchmark for evaluating hallucination detection performance
of the Open Hallucination Index (OHI) API against VectorRAG and GraphRAG systems.

This is the main entry point for backward compatibility. The implementation
has been refactored into the `benchmark` package for better maintainability.

Usage:
    python benchmark_suite.py [OPTIONS]
    python -m benchmark [OPTIONS]

For detailed options, run:
    python benchmark_suite.py --help

Examples:
    python benchmark_suite.py
    python benchmark_suite.py --strategies vector_semantic,mcp_enhanced
    python benchmark_suite.py --threshold 0.6 --concurrency 5

Author: OHI Team
Version: 2.0.0
"""

from __future__ import annotations

import sys

# Explicit public API for re-exports
__all__ = [
    # Core
    "OHIBenchmarkRunner",
    "BenchmarkConfig",
    "get_config",
    # Models
    "BenchmarkCase",
    "ResultMetric",
    "StrategyReport",
    "BenchmarkReport",
    "VerificationStrategy",
    "DifficultyLevel",
    "ConfidenceInterval",
    "StatisticalComparison",
    # Metrics
    "ConfusionMatrix",
    "CalibrationMetrics",
    "LatencyStats",
    "ROCAnalysis",
    "PRCurveAnalysis",
    # Statistical
    "mcnemar_test",
    "bootstrap_ci",
    "delong_test",
    "wilson_ci",
    # Reporters
    "BaseReporter",
    "ConsoleReporter",
    "MarkdownReporter",
    "JSONReporter",
    "CSVReporter",
    # Entry
    "main",
]

# Re-export core components for backward compatibility
# These are intentionally re-exported for public API
from benchmark import (
    OHIBenchmarkRunner,
    BenchmarkConfig,
    get_config,
    # Models
    BenchmarkCase,
    ResultMetric,
    StrategyReport,
    BenchmarkReport,
    VerificationStrategy,
    DifficultyLevel,
    ConfidenceInterval,
    StatisticalComparison,
    # Metrics
    ConfusionMatrix,
    CalibrationMetrics,
    LatencyStats,
    ROCAnalysis,
    PRCurveAnalysis,
    # Statistical functions
    mcnemar_test,
    bootstrap_ci,
    delong_test,
    wilson_ci,
    # Reporters
    BaseReporter,
    ConsoleReporter,
    MarkdownReporter,
    JSONReporter,
    CSVReporter,
)


def main() -> int:
    """
    Main entry point for backward compatibility.

    Delegates to the benchmark module's CLI.
    """
    from benchmark.__main__ import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
