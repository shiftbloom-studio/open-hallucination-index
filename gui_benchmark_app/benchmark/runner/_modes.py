"""
Benchmark execution modes.

Provides different comparison modes:
- Standard: Compare all evaluators head-to-head
- Strategy: Compare OHI verification strategies
- Cache: Compare cold vs warm cache performance

Each mode orchestrates the benchmark execution loop
with appropriate setup and teardown.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.rule import Rule

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.comparison_benchmark import EvaluatorMetrics
from benchmark.runner._benchmarks import (
    run_factscore_benchmark,
    run_hallucination_benchmark,
    run_truthfulqa_benchmark,
    warmup_evaluator,
)
from benchmark.runner._cache import CacheManager
from benchmark.runner._display import LiveBenchmarkDisplay
from benchmark.runner._types import LiveStats

if TYPE_CHECKING:
    from rich.console import Console

    from benchmark.comparison_benchmark import ComparisonReport
    from benchmark.evaluators import BaseEvaluator

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluator Benchmarking
# =============================================================================


async def benchmark_single_evaluator(
    evaluator: BaseEvaluator,
    display: LiveBenchmarkDisplay,
    stats: LiveStats,
    config: ComparisonBenchmarkConfig,
    cache: CacheManager,
    max_latency_ms: float,
) -> EvaluatorMetrics:
    """
    Run all benchmarks for a single evaluator.
    
    Args:
        evaluator: Evaluator to benchmark
        display: Live display for progress
        stats: Shared statistics
        config: Benchmark configuration
        cache: Cache manager for flush operations
        max_latency_ms: Maximum latency threshold
        
    Returns:
        EvaluatorMetrics with all results
    """
    metrics = EvaluatorMetrics(evaluator_name=evaluator.name)
    concurrency = _effective_concurrency(evaluator, config.concurrency)
    
    # Warmup
    include_factscore = "factscore" in config.metrics
    if hasattr(display, "log_warning"):
        display.log_warning(
            f"Warmup started for {evaluator.name} ({config.warmup_requests} requests)"
        )
    await warmup_evaluator(evaluator, config.warmup_requests, include_factscore)
    if hasattr(display, "log_warning"):
        display.log_warning(f"Warmup finished for {evaluator.name}")
    cache.flush(f"after:warmup:{evaluator.name}")
    
    # 1. Hallucination Detection
    if "hallucination" in config.metrics:
        start = time.perf_counter()
        halluc_metrics, latencies = await run_hallucination_benchmark(
            evaluator=evaluator,
            display=display,
            stats=stats,
            dataset_path=config.hallucination_dataset,
            extended_dataset_path=config.extended_dataset,
            max_samples=config.hallucination_max_samples,
            concurrency=concurrency,
            max_latency_ms=max_latency_ms,
        )
        metrics.hallucination = halluc_metrics
        metrics.latency.latencies_ms.extend(latencies)
        cache.flush(f"after:hallucination:{evaluator.name}")
        logger.info(f"{evaluator.name} hallucination took {time.perf_counter() - start:.2f}s")
    
    # 2. TruthfulQA
    if "truthfulqa" in config.metrics:
        start = time.perf_counter()
        tqa_metrics, latencies = await run_truthfulqa_benchmark(
            evaluator=evaluator,
            display=display,
            stats=stats,
            max_samples=config.truthfulqa.max_samples,
            categories=config.truthfulqa.categories,
            concurrency=concurrency,
            max_latency_ms=max_latency_ms,
        )
        metrics.truthfulqa = tqa_metrics
        metrics.latency.latencies_ms.extend(latencies)
        cache.flush(f"after:truthfulqa:{evaluator.name}")
        logger.info(f"{evaluator.name} truthfulqa took {time.perf_counter() - start:.2f}s")
    
    # 3. FActScore
    if "factscore" in config.metrics:
        start = time.perf_counter()
        fac_metrics, latencies = await run_factscore_benchmark(
            evaluator=evaluator,
            display=display,
            stats=stats,
            max_samples=config.factscore.max_samples,
            concurrency=concurrency,
            max_latency_ms=max_latency_ms,
        )
        metrics.factscore = fac_metrics
        metrics.latency.latencies_ms.extend(latencies)
        cache.flush(f"after:factscore:{evaluator.name}")
        logger.info(f"{evaluator.name} factscore took {time.perf_counter() - start:.2f}s")
    
    # Remove first latency (warmup artifact)
    if metrics.latency.latencies_ms:
        metrics.latency.latencies_ms.pop(0)
    
    return metrics


def _effective_concurrency(evaluator: BaseEvaluator, configured: int) -> int:
    """Determine evaluator-specific concurrency cap."""
    max_concurrency = getattr(evaluator, "max_concurrency", None)
    if isinstance(max_concurrency, int) and max_concurrency > 0:
        return max(1, min(configured, max_concurrency))
    return max(1, configured)


# =============================================================================
# Standard Comparison Mode
# =============================================================================


async def run_standard_comparison(
    evaluators: dict[str, BaseEvaluator],
    report: ComparisonReport,
    config: ComparisonBenchmarkConfig,
    cache: CacheManager,
    console: Console,
    max_latency_ms: float,
) -> None:
    """
    Run standard evaluator comparison with live display.
    
    Args:
        evaluators: Dict of evaluator name to instance
        report: Report to populate with results
        config: Benchmark configuration
        cache: Cache manager
        console: Rich console for output
        max_latency_ms: Maximum latency threshold
    """
    stats = LiveStats(
        total_evaluators=len(evaluators),
        start_time=time.perf_counter(),
    )
    
    with LiveBenchmarkDisplay(console, stats) as display:
        for evaluator in evaluators.values():
            display.set_evaluator(evaluator.name)
            
            metrics = await benchmark_single_evaluator(
                evaluator=evaluator,
                display=display,
                stats=stats,
                config=config,
                cache=cache,
                max_latency_ms=max_latency_ms,
            )
            report.add_evaluator(metrics)
            
            display.complete_evaluator(
                evaluator.name,
                {
                    "accuracy": metrics.hallucination.accuracy,
                    "f1": metrics.hallucination.f1_score,
                    "p50": metrics.latency.p50,
                    "p95": metrics.latency.p95,
                },
            )


# =============================================================================
# Strategy Comparison Mode
# =============================================================================


async def run_strategy_comparison(
    evaluators: dict[str, BaseEvaluator],
    report: ComparisonReport,
    config: ComparisonBenchmarkConfig,
    cache: CacheManager,
    console: Console,
    max_latency_ms: float,
) -> None:
    """
    Run OHI strategy comparison mode.
    
    Tests each verification strategy separately to find the optimal one.
    
    Args:
        evaluators: Dict of evaluator name to instance
        report: Report to populate with results
        config: Benchmark configuration
        cache: Cache manager
        console: Rich console for output
        max_latency_ms: Maximum latency threshold
    """
    console.print(Panel(
        f"[bold yellow]OHI Strategy Comparison Mode[/bold yellow]\n"
        f"Testing {len(config.ohi_strategies)} strategies",
        border_style="yellow",
    ))
    
    non_ohi_count = sum(1 for name in evaluators if name != "ohi")
    total_evals = non_ohi_count + len(config.ohi_strategies)
    
    stats = LiveStats(
        total_evaluators=total_evals,
        start_time=time.perf_counter(),
    )
    
    with LiveBenchmarkDisplay(console, stats) as display:
        # Run non-OHI evaluators first
        for name, evaluator in evaluators.items():
            if name != "ohi":
                display.set_evaluator(evaluator.name)
                metrics = await benchmark_single_evaluator(
                    evaluator=evaluator,
                    display=display,
                    stats=stats,
                    config=config,
                    cache=cache,
                    max_latency_ms=max_latency_ms,
                )
                report.add_evaluator(metrics)
                display.complete_evaluator(
                    evaluator.name,
                    {
                        "accuracy": metrics.hallucination.accuracy,
                        "f1": metrics.hallucination.f1_score,
                        "p50": metrics.latency.p50,
                        "p95": metrics.latency.p95,
                    },
                )
        
        # Run each OHI strategy
        if "ohi" in evaluators:
            from benchmark.evaluators import OHIEvaluator
            
            for strategy in config.ohi_strategies:
                display.set_evaluator(f"OHI ({strategy})")
                
                strategy_config = ComparisonBenchmarkConfig.from_env()
                strategy_config.ohi_strategy = strategy
                strategy_evaluator = OHIEvaluator(strategy_config)
                
                try:
                    if await strategy_evaluator.health_check():
                        metrics = await benchmark_single_evaluator(
                            evaluator=strategy_evaluator,
                            display=display,
                            stats=stats,
                            config=config,
                            cache=cache,
                            max_latency_ms=max_latency_ms,
                        )
                        metrics.evaluator_name = f"OHI ({strategy})"
                        report.add_evaluator(metrics)
                        display.complete_evaluator(
                            f"OHI ({strategy})",
                            {
                                "accuracy": metrics.hallucination.accuracy,
                                "f1": metrics.hallucination.f1_score,
                                "p50": metrics.latency.p50,
                                "p95": metrics.latency.p95,
                            },
                        )
                finally:
                    await strategy_evaluator.close()


# =============================================================================
# Cache Comparison Mode
# =============================================================================


async def run_cache_comparison(
    evaluators: dict[str, BaseEvaluator],
    report: ComparisonReport,
    config: ComparisonBenchmarkConfig,
    cache: CacheManager,
    console: Console,
    max_latency_ms: float,
) -> None:
    """
    Run cache comparison mode with cold vs warm testing.
    
    Tests each evaluator twice:
    1. Cold cache: Cache cleared before run
    2. Warm cache: Cache populated from previous run
    
    Args:
        evaluators: Dict of evaluator name to instance
        report: Report to populate with results
        config: Benchmark configuration
        cache: Cache manager
        console: Rich console for output
        max_latency_ms: Maximum latency threshold
    """
    console.print(Panel(
        "[bold yellow]Cache Testing Mode[/bold yellow]\n"
        "Testing with cold vs warm cache",
        border_style="yellow",
    ))
    
    stats = LiveStats(
        total_evaluators=len(evaluators) * 2,
        start_time=time.perf_counter(),
    )
    
    with LiveBenchmarkDisplay(console, stats) as display:
        for evaluator in evaluators.values():
            # Cold cache test
            display.set_evaluator(f"{evaluator.name} (Cold)")
            deleted = cache.clear_ohi_keys()
            logger.info(f"Cleared {deleted} cache keys")
            
            metrics_cold = await benchmark_single_evaluator(
                evaluator=evaluator,
                display=display,
                stats=stats,
                config=config,
                cache=cache,
                max_latency_ms=max_latency_ms,
            )
            metrics_cold.evaluator_name = f"{evaluator.name} (Cold)"
            report.add_evaluator(metrics_cold)
            display.complete_evaluator(
                f"{evaluator.name} (Cold)",
                {
                    "accuracy": metrics_cold.hallucination.accuracy,
                    "f1": metrics_cold.hallucination.f1_score,
                    "p50": metrics_cold.latency.p50,
                    "p95": metrics_cold.latency.p95,
                },
            )
            
            # Warm cache test
            display.set_evaluator(f"{evaluator.name} (Warm)")
            
            metrics_warm = await benchmark_single_evaluator(
                evaluator=evaluator,
                display=display,
                stats=stats,
                config=config,
                cache=cache,
                max_latency_ms=max_latency_ms,
            )
            metrics_warm.evaluator_name = f"{evaluator.name} (Warm)"
            report.add_evaluator(metrics_warm)
            display.complete_evaluator(
                f"{evaluator.name} (Warm)",
                {
                    "accuracy": metrics_warm.hallucination.accuracy,
                    "f1": metrics_warm.hallucination.f1_score,
                    "p50": metrics_warm.latency.p50,
                    "p95": metrics_warm.latency.p95,
                },
            )
            
            # Log speedup
            if metrics_cold.latency.p50 > 0 and metrics_warm.latency.p50 > 0:
                speedup = metrics_cold.latency.p50 / metrics_warm.latency.p50
                logger.info(
                    f"Cache speedup for {evaluator.name}: {speedup:.2f}x "
                    f"(P50: {metrics_cold.latency.p50:.0f}ms → {metrics_warm.latency.p50:.0f}ms)"
                )
    
    # Print summary
    console.print(Rule("📊 Cache Impact Summary", style="cyan"))
    for name in evaluators:
        cold_name = f"{name} (Cold)"
        warm_name = f"{name} (Warm)"
        if cold_name in report.evaluators and warm_name in report.evaluators:
            cold = report.evaluators[cold_name]
            warm = report.evaluators[warm_name]
            if cold.latency.p50 > 0 and warm.latency.p50 > 0:
                speedup = cold.latency.p50 / warm.latency.p50
                console.print(
                    f"  [green]✓[/green] {name}: {speedup:.2f}x speedup "
                    f"(P50: {cold.latency.p50:.0f}ms → {warm.latency.p50:.0f}ms)"
                )
