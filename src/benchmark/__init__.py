"""
OHI Benchmark Suite - Multi-Evaluator Comparison Framework
============================================================

A research-grade benchmarking framework for comparing hallucination detection
systems across multiple dimensions:

Evaluators:
-----------
* OHI (Open Hallucination Index) - Hybrid verification with knowledge graph + vector + MCP
* GPT-4 (OpenAI) - Direct LLM verification baseline
* VectorRAG - Vector similarity baseline (Qdrant by default; optional fair mode via Wikipedia API)

Metrics:
--------
* Hallucination Detection - Accuracy, precision, recall, F1, MCC
* TruthfulQA - Adversarial question answering accuracy
* FActScore - Atomic fact precision (claim decomposition + verification)
* Latency - Response time analysis (P50, P90, P99)

Features:
---------
* Multi-evaluator comparison with statistical significance testing
* Stratified analysis by domain, difficulty, and claim complexity
* Redis cache testing (with/without cache, cache cleared before run)
* Comprehensive multi-format reporting (HTML, JSON, Markdown, CSV)
* Publication-ready comparison charts (radar, heatmap, grouped bars, combined dashboard)

Usage:
------
    # Run comparison benchmark via CLI
    python -m benchmark.comparison_benchmark --evaluators ohi,gpt4,vector_rag

    # Quick test
    python src/benchmark/test_evaluators_quick.py

    # Detailed comparison
    python src/benchmark/test_evaluators_detailed.py

    # Programmatic usage
    from benchmark import ComparisonBenchmarkRunner, ComparisonBenchmarkConfig

    async with ComparisonBenchmarkRunner() as runner:
        report = await runner.run_comparison()

Author: OHI Research Team
License: MIT
Version: 3.0.0
"""

from benchmark.comparison_config import (
    ComparisonBenchmarkConfig,
    EvaluatorType,
    MetricType,
    OpenAIConfig,
    VectorRAGConfig,
)
from benchmark.comparison_metrics import (
    ComparisonReport,
    EvaluatorMetrics,
    FActScoreMetrics,
    HallucinationMetrics,
    LatencyMetrics,
    TruthfulQAMetrics,
)
from benchmark.comparison_runner import (
    ComparisonBenchmarkRunner,
    run_comparison_benchmark,
)

# Reporters
from benchmark.reporters.base import BaseReporter
from benchmark.reporters.charts import ChartsReporter
from benchmark.reporters.console import ConsoleReporter
from benchmark.reporters.csv_reporter import CSVReporter
from benchmark.reporters.json_reporter import JSONReporter
from benchmark.reporters.markdown import MarkdownReporter

# Evaluators
from benchmark.evaluators import (
    BaseEvaluator,
    EvaluatorResult,
    FActScoreResult,
    FairVectorRAGEvaluator,
    GPT4Evaluator,
    OHIEvaluator,
    VectorRAGEvaluator,
    VerificationVerdict,
    get_evaluator,
)

__version__ = "3.0.0"
__all__ = [
    # Core Runner
    "ComparisonBenchmarkRunner",
    "run_comparison_benchmark",
    # Configuration
    "ComparisonBenchmarkConfig",
    "EvaluatorType",
    "MetricType",
    "OpenAIConfig",
    "VectorRAGConfig",
    # Metrics
    "ComparisonReport",
    "EvaluatorMetrics",
    "FActScoreMetrics",
    "HallucinationMetrics",
    "LatencyMetrics",
    "TruthfulQAMetrics",
    # Evaluators
    "BaseEvaluator",
    "EvaluatorResult",
    "FActScoreResult",
    "FairVectorRAGEvaluator",
    "GPT4Evaluator",
    "OHIEvaluator",
    "VectorRAGEvaluator",
    "VerificationVerdict",
    "get_evaluator",
    # Reporters
    "BaseReporter",
    "ChartsReporter",
    "ConsoleReporter",
    "CSVReporter",
    "JSONReporter",
    "MarkdownReporter",
]
