"""
OHI Benchmark Suite - Research-Grade Hallucination Detection Evaluation
========================================================================

A comprehensive benchmarking framework for evaluating the Open Hallucination Index
API against VectorRAG, GraphRAG, and hybrid retrieval strategies.

Features:
---------
* Multi-strategy comparison with statistical significance testing
* Multi-evaluator comparison (OHI vs GPT-4 vs VectorRAG)
* Stratified analysis by domain, difficulty, and claim complexity
* Bootstrap confidence intervals for all metrics
* DeLong test for AUC comparison between strategies
* McNemar's test for paired classifier comparison
* Calibration analysis (Brier Score, ECE, MCE)
* ROC/PR curve analysis with optimal threshold detection
* Comprehensive multi-format reporting (HTML, JSON, Markdown, CSV)
* Publication-ready comparison charts (radar, heatmap, grouped bars)

Usage:
------
    # Standard benchmark
    python -m benchmark --strategies vector_semantic,mcp_enhanced

    # Comparison benchmark (OHI vs GPT-4 vs VectorRAG)
    python -m benchmark.comparison_benchmark

    # Programmatically
    from benchmark import OHIBenchmarkRunner
    async with OHIBenchmarkRunner(strategies=["vector_semantic", "mcp_enhanced"]) as runner:
        report = await runner.run_benchmark()

    # Comparison benchmark
    from benchmark.comparison_runner import run_comparison_benchmark
    report = await run_comparison_benchmark()

Author: OHI Research Team
License: MIT
Version: 2.0.0
"""

from benchmark.config import BenchmarkConfig, get_config
from benchmark.models import (
    BenchmarkCase,
    BenchmarkReport,
    ConfidenceInterval,
    DifficultyLevel,
    ResultMetric,
    StatisticalComparison,
    StrategyReport,
    VerificationStrategy,
)
from benchmark.metrics import (
    CalibrationMetrics,
    ConfusionMatrix,
    LatencyStats,
    PRCurveAnalysis,
    ROCAnalysis,
)
from benchmark.analysis.statistical import (
    bootstrap_ci,
    delong_test,
    mcnemar_test,
    wilson_ci,
)
from benchmark.reporters.base import BaseReporter
from benchmark.reporters.console import ConsoleReporter
from benchmark.reporters.markdown import MarkdownReporter
from benchmark.reporters.json_reporter import JSONReporter
from benchmark.reporters.csv_reporter import CSVReporter
from benchmark.runner import OHIBenchmarkRunner

# Comparison benchmark components
from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.comparison_metrics import (
    ComparisonReport,
    EvaluatorMetrics,
    FActScoreMetrics,
    HallucinationMetrics as ComparisonHallucinationMetrics,
    LatencyMetrics as ComparisonLatencyMetrics,
    TruthfulQAMetrics,
)
from benchmark.comparison_runner import ComparisonBenchmarkRunner, run_comparison_benchmark

__version__ = "2.0.0"
__all__ = [
    # Core runner
    "OHIBenchmarkRunner",
    # Comparison runner
    "ComparisonBenchmarkRunner",
    "run_comparison_benchmark",
    "ComparisonBenchmarkConfig",
    "ComparisonReport",
    "EvaluatorMetrics",
    # Configuration
    "BenchmarkConfig",
    "get_config",
    # Models
    "BenchmarkCase",
    "BenchmarkReport",
    "ConfidenceInterval",
    "DifficultyLevel",
    "ResultMetric",
    "StatisticalComparison",
    "StrategyReport",
    "VerificationStrategy",
    # Metrics
    "CalibrationMetrics",
    "ConfusionMatrix",
    "LatencyStats",
    "PRCurveAnalysis",
    "ROCAnalysis",
    # Comparison Metrics
    "ComparisonHallucinationMetrics",
    "ComparisonLatencyMetrics",
    "FActScoreMetrics",
    "TruthfulQAMetrics",
    # Statistical
    "bootstrap_ci",
    "delong_test",
    "mcnemar_test",
    "wilson_ci",
    # Reporters
    "BaseReporter",
    "ConsoleReporter",
    "CSVReporter",
    "JSONReporter",
    "MarkdownReporter",
]
