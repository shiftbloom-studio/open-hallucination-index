"""
Comparison Benchmark Runner
============================

Main orchestration for multi-evaluator benchmark comparison.

Runs three evaluators (OHI, GPT-4, VectorRAG) across four metrics:
- Hallucination Detection
- TruthfulQA
- FActScore
- Latency

Additional modes:
- OHI Strategy Comparison: Compare all verification strategies
- Cache Testing: Compare performance with/without Redis cache

Generates comprehensive comparison reports and visualizations.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import redis
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.comparison_metrics import (
    ComparisonReport,
    EvaluatorMetrics,
    FActScoreMetrics,
    HallucinationMetrics,
    LatencyMetrics,
    TruthfulQAMetrics,
)
from benchmark.datasets import HallucinationLoader, TruthfulQALoader
from benchmark.evaluators import (
    BaseEvaluator,
    EvaluatorResult,
    FActScoreResult,
    get_evaluator,
)

logger = logging.getLogger("OHI-Comparison-Benchmark")


class ComparisonBenchmarkRunner:
    """
    Orchestrates multi-evaluator benchmark comparison.
    
    Runs OHI, GPT-4, and VectorRAG across multiple metrics
    and generates comprehensive comparison reports.
    
    Supports special modes:
    - OHI Strategy Comparison: Test all verification strategies
    - Cache Testing: Compare with/without Redis cache
    """
    
    def __init__(
        self,
        config: ComparisonBenchmarkConfig | None = None,
        console: Console | None = None,
    ) -> None:
        self.config = config or ComparisonBenchmarkConfig.from_env()
        self.console = console or Console()
        
        self.run_id = f"comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.start_time: float = 0.0
        
        # Evaluators (initialized lazily)
        self._evaluators: dict[str, BaseEvaluator] = {}
        
        # Redis client for cache testing
        self._redis: redis.Redis | None = None
        
        # Output directory
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self) -> "ComparisonBenchmarkRunner":
        """Async context manager entry."""
        await self._initialize_evaluators()
        self._init_redis()
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self._cleanup()
    
    def _init_redis(self) -> None:
        """Initialize Redis client for cache management."""
        if self.config.cache_testing:
            try:
                self._redis = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    password=self.config.redis_password,
                    decode_responses=True,
                )
                self._redis.ping()
                self.console.print(f"  [green]✓[/green] Redis connected ({self.config.redis_host}:{self.config.redis_port})")
            except Exception as e:
                self.console.print(f"  [yellow]⚠[/yellow] Redis not available: {e}")
                self._redis = None
    
    def _clear_cache(self) -> int:
        """
        Clear Redis cache and return number of keys deleted.
        
        Returns:
            Number of keys deleted.
        """
        if not self._redis:
            return 0
        
        try:
            # Get all OHI-related keys
            keys = self._redis.keys("ohi:*")
            if keys:
                deleted = self._redis.delete(*keys)
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return 0
    
    def _set_cache_enabled(self, enabled: bool) -> bool:
        """
        Set OHI API cache mode via environment variable or API call.
        
        Note: This is a placeholder. The actual implementation depends on
        how the OHI API handles cache configuration.
        
        Args:
            enabled: Whether to enable caching.
            
        Returns:
            True if successful.
        """
        # The OHI API currently doesn't have a runtime cache toggle.
        # This method is here for future extensibility.
        # For now, we use the cache testing by clearing the cache
        # before runs to simulate "cold" cache performance.
        return True
    
    async def _initialize_evaluators(self) -> None:
        """Initialize all configured evaluators."""
        active_evaluators = self.config.get_active_evaluators()
        
        self.console.print(Panel(
            f"[bold cyan]OHI Comparison Benchmark[/bold cyan]\n"
            f"Run ID: {self.run_id}\n"
            f"Evaluators: {', '.join(active_evaluators)}\n"
            f"Metrics: {', '.join(self.config.metrics)}",
            border_style="cyan",
        ))
        
        for eval_name in active_evaluators:
            try:
                evaluator = get_evaluator(eval_name, self.config)
                is_healthy = await evaluator.health_check()
                
                if is_healthy:
                    self._evaluators[eval_name] = evaluator
                    self.console.print(f"  [green]✓[/green] {evaluator.name} ready")
                else:
                    self.console.print(f"  [yellow]⚠[/yellow] {eval_name} not available (health check failed)")
            except Exception as e:
                self.console.print(f"  [red]✗[/red] {eval_name} failed: {e}")
        
        if not self._evaluators:
            raise RuntimeError("No evaluators available")
    
    async def _cleanup(self) -> None:
        """Close all evaluators."""
        for evaluator in self._evaluators.values():
            await evaluator.close()
    
    async def run_comparison(self) -> ComparisonReport:
        """
        Run full comparison benchmark.
        
        Supports special modes:
        - OHI Strategy Comparison: If ohi_all_strategies=True, runs all strategies
        - Cache Testing: If cache_testing=True, runs with cold cache vs warm cache
        
        Returns:
            ComparisonReport with all evaluator metrics.
        """
        self.start_time = time.perf_counter()
        
        # Build mode description
        mode_info = []
        if self.config.ohi_all_strategies:
            mode_info.append(f"OHI Strategies: {', '.join(self.config.ohi_strategies)}")
        if self.config.cache_testing:
            mode_info.append("Cache Testing: ON")
        
        report = ComparisonReport(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_summary={
                "evaluators": list(self._evaluators.keys()),
                "metrics": self.config.metrics,
                "hallucination_dataset": str(self.config.hallucination_dataset),
                "ohi_all_strategies": self.config.ohi_all_strategies,
                "cache_testing": self.config.cache_testing,
            },
        )
        
        # Standard comparison mode
        if not self.config.ohi_all_strategies and not self.config.cache_testing:
            await self._run_standard_comparison(report)
        
        # OHI Strategy Comparison Mode
        elif self.config.ohi_all_strategies:
            await self._run_strategy_comparison(report)
        
        # Cache Testing Mode
        elif self.config.cache_testing:
            await self._run_cache_comparison(report)
        
        # Generate reports and charts
        await self._generate_outputs(report)
        
        # Print final comparison
        self._print_comparison_table(report)
        
        return report
    
    async def _run_standard_comparison(self, report: ComparisonReport) -> None:
        """Run standard evaluator comparison."""
        for eval_name, evaluator in self._evaluators.items():
            self.console.print(f"\n[bold]Benchmarking {evaluator.name}...[/bold]")
            
            metrics = await self._benchmark_evaluator(evaluator)
            report.add_evaluator(metrics)
            
            self._print_evaluator_summary(metrics)
    
    async def _run_strategy_comparison(self, report: ComparisonReport) -> None:
        """
        Run OHI strategy comparison mode.
        
        Tests each verification strategy separately to find the optimal one.
        """
        self.console.print(Panel(
            f"[bold yellow]OHI Strategy Comparison Mode[/bold yellow]\n"
            f"Testing {len(self.config.ohi_strategies)} strategies",
            border_style="yellow",
        ))
        
        # Run non-OHI evaluators first (standard mode)
        for eval_name, evaluator in self._evaluators.items():
            if eval_name != "ohi":
                self.console.print(f"\n[bold]Benchmarking {evaluator.name}...[/bold]")
                metrics = await self._benchmark_evaluator(evaluator)
                report.add_evaluator(metrics)
                self._print_evaluator_summary(metrics)
        
        # Run each OHI strategy
        ohi_evaluator = self._evaluators.get("ohi")
        if ohi_evaluator:
            for strategy in self.config.ohi_strategies:
                self.console.print(f"\n[bold cyan]Testing OHI Strategy: {strategy}[/bold cyan]")
                
                # Create strategy-specific evaluator
                strategy_config = ComparisonBenchmarkConfig.from_env()
                strategy_config.ohi_strategy = strategy
                
                from benchmark.evaluators import OHIEvaluator
                strategy_evaluator = OHIEvaluator(strategy_config)
                
                try:
                    if await strategy_evaluator.health_check():
                        metrics = await self._benchmark_evaluator(strategy_evaluator)
                        metrics.evaluator_name = f"OHI ({strategy})"
                        report.add_evaluator(metrics)
                        self._print_evaluator_summary(metrics)
                    else:
                        self.console.print(f"  [yellow]⚠ Strategy {strategy} health check failed[/yellow]")
                finally:
                    await strategy_evaluator.close()
    
    async def _run_cache_comparison(self, report: ComparisonReport) -> None:
        """
        Run cache comparison mode.
        
        Tests each evaluator twice:
        1. Cold cache: Cache cleared before run
        2. Warm cache: Cache populated from previous run
        """
        self.console.print(Panel(
            f"[bold yellow]Cache Testing Mode[/bold yellow]\n"
            f"Testing with cold vs warm cache",
            border_style="yellow",
        ))
        
        for eval_name, evaluator in self._evaluators.items():
            # Test 1: Cold cache (cleared before run)
            self.console.print(f"\n[bold]Benchmarking {evaluator.name} (Cold Cache)...[/bold]")
            
            deleted = self._clear_cache()
            self.console.print(f"  [dim]Cleared {deleted} cache keys[/dim]")
            
            metrics_cold = await self._benchmark_evaluator(evaluator)
            metrics_cold.evaluator_name = f"{evaluator.name} (Cold)"
            report.add_evaluator(metrics_cold)
            self._print_evaluator_summary(metrics_cold)
            
            # Test 2: Warm cache (use cache from previous run)
            self.console.print(f"\n[bold]Benchmarking {evaluator.name} (Warm Cache)...[/bold]")
            
            metrics_warm = await self._benchmark_evaluator(evaluator)
            metrics_warm.evaluator_name = f"{evaluator.name} (Warm)"
            report.add_evaluator(metrics_warm)
            self._print_evaluator_summary(metrics_warm)
            
            # Print cache impact
            if metrics_cold.latency.p50 > 0 and metrics_warm.latency.p50 > 0:
                speedup = metrics_cold.latency.p50 / metrics_warm.latency.p50
                self.console.print(
                    f"  [green]Cache speedup: {speedup:.2f}x (P50: {metrics_cold.latency.p50:.0f}ms → {metrics_warm.latency.p50:.0f}ms)[/green]"
                )
        
        return report
    
    async def _benchmark_evaluator(
        self,
        evaluator: BaseEvaluator,
    ) -> EvaluatorMetrics:
        """
        Run all benchmarks for a single evaluator.
        
        Args:
            evaluator: The evaluator to benchmark.
            
        Returns:
            EvaluatorMetrics with all metric results.
        """
        metrics = EvaluatorMetrics(evaluator_name=evaluator.name)
        
        # 1. Hallucination Detection
        if "hallucination" in self.config.metrics:
            halluc_metrics, latencies = await self._run_hallucination_benchmark(evaluator)
            metrics.hallucination = halluc_metrics
            metrics.latency.latencies_ms.extend(latencies)
        
        # 2. TruthfulQA
        if "truthfulqa" in self.config.metrics:
            tqa_metrics, latencies = await self._run_truthfulqa_benchmark(evaluator)
            metrics.truthfulqa = tqa_metrics
            metrics.latency.latencies_ms.extend(latencies)
        
        # 3. FActScore
        if "factscore" in self.config.metrics:
            fac_metrics, latencies = await self._run_factscore_benchmark(evaluator)
            metrics.factscore = fac_metrics
            metrics.latency.latencies_ms.extend(latencies)
        
        return metrics
    
    async def _run_hallucination_benchmark(
        self,
        evaluator: BaseEvaluator,
    ) -> tuple[HallucinationMetrics, list[float]]:
        """Run hallucination detection benchmark."""
        loader = HallucinationLoader(self.config.hallucination_dataset)
        
        # Load dataset
        try:
            if self.config.extended_dataset and self.config.extended_dataset.exists():
                dataset = loader.load_combined(
                    csv_path=self.config.hallucination_dataset,
                    include_huggingface=False,
                )
            else:
                dataset = loader.load_csv()
        except FileNotFoundError:
            # Try loading from HuggingFace only
            dataset = loader.load_from_huggingface(max_samples=200)
        
        metrics = HallucinationMetrics(total=dataset.total)
        latencies: list[float] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Hallucination ({evaluator.name})",
                total=len(dataset.cases),
            )
            
            # Process in batches
            batch_size = self.config.concurrency
            for i in range(0, len(dataset.cases), batch_size):
                batch = dataset.cases[i:i + batch_size]
                claims = [case.text for case in batch]
                
                results = await evaluator.verify_batch(claims, concurrency=batch_size)
                
                for case, result in zip(batch, results, strict=True):
                    latencies.append(result.latency_ms)
                    
                    # Compare prediction to ground truth
                    predicted = result.predicted_label
                    expected = case.label
                    
                    if predicted and expected:
                        metrics.true_positives += 1
                        metrics.correct += 1
                    elif not predicted and not expected:
                        metrics.true_negatives += 1
                        metrics.correct += 1
                    elif predicted and not expected:
                        metrics.false_positives += 1  # Dangerous!
                    else:
                        metrics.false_negatives += 1
                
                progress.update(task, advance=len(batch))
        
        return metrics, latencies
    
    async def _run_truthfulqa_benchmark(
        self,
        evaluator: BaseEvaluator,
    ) -> tuple[TruthfulQAMetrics, list[float]]:
        """Run TruthfulQA benchmark."""
        loader = TruthfulQALoader()
        
        try:
            claims = loader.load_for_verification(
                max_samples=self.config.truthfulqa.max_samples,
                categories=self.config.truthfulqa.categories,
            )
        except Exception as e:
            logger.warning(f"Failed to load TruthfulQA: {e}")
            return TruthfulQAMetrics(), []
        
        metrics = TruthfulQAMetrics(total_questions=len(claims))
        latencies: list[float] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]TruthfulQA ({evaluator.name})",
                total=len(claims),
            )
            
            # Process claims
            for claim_text, is_correct in claims:
                result = await evaluator.verify(claim_text)
                latencies.append(result.latency_ms)
                
                # Check if evaluator agrees with ground truth
                predicted = result.predicted_label
                if predicted == is_correct:
                    metrics.correct_predictions += 1
                
                progress.update(task, advance=1)
        
        return metrics, latencies
    
    async def _run_factscore_benchmark(
        self,
        evaluator: BaseEvaluator,
    ) -> tuple[FActScoreMetrics, list[float]]:
        """Run FActScore benchmark."""
        # Sample texts for FActScore evaluation
        sample_texts = [
            "Albert Einstein was born in Germany in 1879. He developed the theory of relativity and won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
            "The Eiffel Tower is located in Paris, France. It was constructed in 1889 for the World's Fair and stands at 324 meters tall. It is made of iron and was designed by Gustave Eiffel.",
            "Python is a programming language created by Guido van Rossum in 1991. It is known for its simple syntax and is widely used in web development, data science, and artificial intelligence.",
            "The human heart has four chambers and pumps blood throughout the body. It beats approximately 100,000 times per day and is located in the chest cavity.",
            "World War II ended in 1945. It involved most of the world's nations and resulted in significant geopolitical changes including the formation of the United Nations.",
        ]
        
        max_samples = self.config.factscore.max_samples or len(sample_texts)
        sample_texts = sample_texts[:max_samples]
        
        metrics = FActScoreMetrics(total_texts=len(sample_texts))
        latencies: list[float] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]FActScore ({evaluator.name})",
                total=len(sample_texts),
            )
            
            for text in sample_texts:
                result = await evaluator.decompose_and_verify(text)
                latencies.append(result.latency_ms)
                
                metrics.total_facts += result.total_facts
                metrics.supported_facts += result.supported_facts
                metrics.facts_per_text.append(result.total_facts)
                
                if result.total_facts > 0:
                    metrics.scores.append(result.factscore)
                
                progress.update(task, advance=1)
        
        return metrics, latencies
    
    def _print_evaluator_summary(self, metrics: EvaluatorMetrics) -> None:
        """Print summary for a single evaluator."""
        table = Table(title=f"{metrics.evaluator_name} Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Accuracy", f"{metrics.hallucination.accuracy:.1%}")
        table.add_row("F1 Score", f"{metrics.hallucination.f1_score:.1%}")
        table.add_row("Halluc. Pass Rate", f"{metrics.hallucination.hallucination_pass_rate:.1%}")
        table.add_row("TruthfulQA", f"{metrics.truthfulqa.accuracy:.1%}")
        table.add_row("FActScore", f"{metrics.factscore.avg_factscore:.1%}")
        table.add_row("P50 Latency", f"{metrics.latency.p50:.0f}ms")
        table.add_row("P95 Latency", f"{metrics.latency.p95:.0f}ms")
        
        self.console.print(table)
    
    def _print_comparison_table(self, report: ComparisonReport) -> None:
        """Print final comparison table."""
        self.console.print("\n")
        
        table = Table(
            title="🏆 Comparison Results",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Evaluator", style="bold")
        table.add_column("Accuracy", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Safety", justify="right")
        table.add_column("TruthfulQA", justify="right")
        table.add_column("FActScore", justify="right")
        table.add_column("P95 Latency", justify="right")
        table.add_column("Throughput", justify="right")
        
        # Sort by F1 score (OHI should be first)
        ranking = report.get_ranking("f1_score")
        
        for i, name in enumerate(ranking):
            m = report.evaluators[name]
            
            # Highlight winner
            style = "green" if i == 0 else ""
            medal = "🥇 " if i == 0 else ("🥈 " if i == 1 else ("🥉 " if i == 2 else "   "))
            
            table.add_row(
                f"{medal}{name}",
                f"[{style}]{m.hallucination.accuracy:.1%}[/{style}]",
                f"[{style}]{m.hallucination.f1_score:.1%}[/{style}]",
                f"[{style}]{1 - m.hallucination.hallucination_pass_rate:.1%}[/{style}]",
                f"[{style}]{m.truthfulqa.accuracy:.1%}[/{style}]",
                f"[{style}]{m.factscore.avg_factscore:.1%}[/{style}]",
                f"[{style}]{m.latency.p95:.0f}ms[/{style}]",
                f"[{style}]{m.latency.throughput:.1f} req/s[/{style}]",
            )
        
        self.console.print(table)
        
        # Print winner announcement
        winner = ranking[0]
        self.console.print(Panel(
            f"[bold green]🏆 Winner: {winner}[/bold green]\n\n"
            f"Best overall performance across hallucination detection, "
            f"truthfulness, and factual accuracy metrics.",
            border_style="green",
        ))
    
    async def _generate_outputs(self, report: ComparisonReport) -> None:
        """Generate all output files and charts."""
        import json as json_module
        
        # Save JSON report
        json_path = self.output_dir / f"{self.run_id}_report.json"
        with open(json_path, "w") as f:
            json_module.dump(report.to_dict(), f, indent=2)
        
        self.console.print(f"[dim]Saved report: {json_path}[/dim]")
        
        # Generate comparison charts
        try:
            from benchmark.reporters.charts import ChartsReporter
            
            charts_reporter = ChartsReporter(
                self.output_dir,
                dpi=self.config.chart_dpi,
            )
            chart_files = charts_reporter.generate_comparison_charts(
                report,
                prefix=f"{self.run_id}_",
            )
            
            for chart_file in chart_files:
                self.console.print(f"[dim]Generated chart: {chart_file.name}[/dim]")
        except ImportError:
            self.console.print("[yellow]Charts not generated (matplotlib not available)[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]Chart generation failed: {e}[/yellow]")


async def run_comparison_benchmark(
    config: ComparisonBenchmarkConfig | None = None,
) -> ComparisonReport:
    """
    Convenience function to run comparison benchmark.
    
    Args:
        config: Optional configuration (loads from env if not provided)
        
    Returns:
        ComparisonReport with all results
    """
    async with ComparisonBenchmarkRunner(config) as runner:
        return await runner.run_comparison()
