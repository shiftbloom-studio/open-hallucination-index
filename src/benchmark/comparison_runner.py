"""
Comparison Benchmark Runner
============================

Main orchestration for multi-evaluator benchmark comparison.

Runs three evaluators (OHI, GPT-4, VectorRAG) across four metrics:
- Hallucination Detection
- TruthfulQA
- FActScore
- Latency

Generates comprehensive comparison reports and visualizations.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

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
        
        # Output directory
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self) -> "ComparisonBenchmarkRunner":
        """Async context manager entry."""
        await self._initialize_evaluators()
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self._cleanup()
    
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
        
        Returns:
            ComparisonReport with all evaluator metrics.
        """
        self.start_time = time.perf_counter()
        
        report = ComparisonReport(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_summary={
                "evaluators": list(self._evaluators.keys()),
                "metrics": self.config.metrics,
                "hallucination_dataset": str(self.config.hallucination_dataset),
            },
        )
        
        # Run benchmarks for each evaluator
        for eval_name, evaluator in self._evaluators.items():
            self.console.print(f"\n[bold]Benchmarking {evaluator.name}...[/bold]")
            
            metrics = await self._benchmark_evaluator(evaluator)
            report.add_evaluator(metrics)
            
            # Print summary
            self._print_evaluator_summary(metrics)
        
        # Generate reports and charts
        await self._generate_outputs(report)
        
        # Print final comparison
        self._print_comparison_table(report)
        
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
