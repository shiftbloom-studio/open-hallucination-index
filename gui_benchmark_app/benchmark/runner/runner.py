"""
Main ComparisonBenchmarkRunner orchestrator.

This is the primary entry point for benchmark execution.
It orchestrates:
- Evaluator initialization
- Mode selection and execution
- Report generation
- Output file creation

Usage:
    ```python
    from benchmark.runner import run_comparison_benchmark
    
    report = await run_comparison_benchmark()
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.comparison_benchmark import ComparisonReport
from benchmark.evaluators import BaseEvaluator, get_evaluator
from benchmark.runner._cache import CacheManager
from benchmark.runner._modes import (
    run_cache_comparison,
    run_standard_comparison,
    run_strategy_comparison,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("OHI-Comparison-Benchmark")


class ComparisonBenchmarkRunner:
    """
    Orchestrates multi-evaluator benchmark comparison.
    
    Runs OHI, GPT-4, VectorRAG, and other evaluators across multiple metrics
    and generates comprehensive comparison reports.
    
    Supported Modes:
        - Standard: Compare all evaluators head-to-head
        - Strategy: Compare OHI verification strategies
        - Cache: Compare cold vs warm cache performance
    
    Usage:
        ```python
        async with ComparisonBenchmarkRunner() as runner:
            report = await runner.run_comparison()
        ```
    
    Attributes:
        config: Benchmark configuration
        console: Rich console for output
        run_id: Unique identifier for this run
        output_dir: Directory for output files
    """
    
    # Maximum latency before aborting (5 minutes)
    MAX_LATENCY_MS: float = 5 * 60 * 1000
    
    def __init__(
        self,
        config: ComparisonBenchmarkConfig | None = None,
        console: Console | None = None,
    ) -> None:
        """
        Initialize the benchmark runner.
        
        Args:
            config: Optional configuration (loads from env if not provided)
            console: Optional Rich console (creates new if not provided)
        """
        from benchmark.runner._display import create_optimized_console
        
        self.config = config or ComparisonBenchmarkConfig.from_env()
        self.console = console or create_optimized_console()
        
        # Generate unique run ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"comparison_{timestamp}_{uuid.uuid4().hex[:6]}"
        
        # Timing
        self.start_time: float = 0.0
        
        # Output directory
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._evaluators: dict[str, BaseEvaluator] = {}
        self._current_report: ComparisonReport | None = None
        self._cache = CacheManager(
            host=self.config.redis_host,
            port=self.config.redis_port,
            password=self.config.redis_password,
        )
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    async def __aenter__(self) -> ComparisonBenchmarkRunner:
        """Async context manager entry."""
        await self._initialize_evaluators()
        self._cache.connect(self.console)
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self._cleanup()
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    async def run_comparison(self) -> ComparisonReport:
        """
        Run the full comparison benchmark.
        
        Automatically selects the appropriate mode based on configuration:
        - Standard comparison (default)
        - OHI Strategy comparison (if ohi_all_strategies=True)
        - Cache testing (if cache_testing=True)
        
        Returns:
            ComparisonReport with all evaluator metrics
        """
        self.start_time = time.perf_counter()
        
        # Build report
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
        self._current_report = report
        
        # Select and run mode
        if self.config.ohi_all_strategies:
            await run_strategy_comparison(
                evaluators=self._evaluators,
                report=report,
                config=self.config,
                cache=self._cache,
                console=self.console,
                max_latency_ms=self.MAX_LATENCY_MS,
            )
        elif self.config.cache_testing:
            await run_cache_comparison(
                evaluators=self._evaluators,
                report=report,
                config=self.config,
                cache=self._cache,
                console=self.console,
                max_latency_ms=self.MAX_LATENCY_MS,
            )
        else:
            await run_standard_comparison(
                evaluators=self._evaluators,
                report=report,
                config=self.config,
                cache=self._cache,
                console=self.console,
                max_latency_ms=self.MAX_LATENCY_MS,
            )
        
        # Generate outputs
        await self._generate_outputs(report)
        
        # Print final comparison
        self._print_comparison_table(report)
        
        return report
    
    async def save_partial_results(self) -> bool:
        """
        Persist partial results if available.
        
        Useful for saving progress on interrupt.
        
        Returns:
            True if results were saved
        """
        if not self._current_report:
            return False
        await self._generate_outputs(self._current_report)
        return True
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
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
                evaluator = get_evaluator(
                    eval_name,
                    self.config,
                )
                is_healthy = await evaluator.health_check()
                
                if is_healthy:
                    self._evaluators[eval_name] = evaluator
                    self.console.print(f"  [green]✓[/green] {evaluator.name} ready")
                else:
                    self.console.print(
                        f"  [yellow]⚠[/yellow] {eval_name} not available (health check failed)"
                    )
            except Exception as e:
                self.console.print(f"  [red]✗[/red] {eval_name} failed: {e}")
        
        if not self._evaluators:
            raise RuntimeError("No evaluators available")
    
    async def _cleanup(self) -> None:
        """Close all evaluators and connections."""
        for evaluator in self._evaluators.values():
            await evaluator.close()
        self._cache.close()
    
    # =========================================================================
    # Output Generation
    # =========================================================================
    
    async def _generate_outputs(self, report: ComparisonReport) -> None:
        """Generate all output files and charts."""
        # Save JSON report
        json_path = self.output_dir / f"{self.run_id}_report.json"
        json_payload = json.dumps(report.to_dict(), indent=2)
        await asyncio.to_thread(json_path.write_text, json_payload, encoding="utf-8")
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
    
    def _print_comparison_table(self, report: ComparisonReport) -> None:
        """Print final comparison results table."""
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
        
        # Sort by F1 score
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
        
        # Winner announcement
        if ranking:
            winner = ranking[0]
            self.console.print(Panel(
                f"[bold green]🏆 Winner: {winner}[/bold green]\n\n"
                f"Best overall performance across hallucination detection, "
                f"truthfulness, and factual accuracy metrics.",
                border_style="green",
            ))


# =============================================================================
# Convenience Function
# =============================================================================


async def run_comparison_benchmark(
    config: ComparisonBenchmarkConfig | None = None,
) -> ComparisonReport:
    """
    Convenience function to run comparison benchmark.
    
    Args:
        config: Optional configuration (loads from env if not provided)
        
    Returns:
        ComparisonReport with all results
        
    Example:
        ```python
        report = await run_comparison_benchmark()
        winner = report.get_ranking("f1_score")[0]
        print(f"Winner: {winner}")
        ```
    """
    async with ComparisonBenchmarkRunner(config) as runner:
        return await runner.run_comparison()
