#!/usr/bin/env python3
"""
OHI Benchmark Suite (Legacy Entry)
=================================

Research-grade benchmark for evaluating hallucination detection performance
of the Open Hallucination Index (OHI) API against VectorRAG and GraphRAG systems.

This module is a legacy entry point that re-exports the public API and delegates
CLI execution to the `benchmark` package.

Usage:
    python -m benchmark [OPTIONS]
    python -m benchmark.legacy_entry [OPTIONS]

For detailed options, run:
    python -m benchmark --help

Examples:
    python -m benchmark
    python -m benchmark --strategies vector_semantic,mcp_enhanced
    python -m benchmark --threshold 0.6 --concurrency 5

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
from benchmark import (  # isort: skip
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
