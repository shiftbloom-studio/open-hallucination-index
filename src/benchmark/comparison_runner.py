"""
Comparison Benchmark Runner
============================

Main orchestration for multi-evaluator benchmark comparison.

Runs multiple evaluators (OHI profiles, GPT-4, VectorRAG, GraphRAG) across four metrics:
- Hallucination Detection
- TruthfulQA
- FActScore
- Latency

Additional modes:
- OHI Strategy Comparison: Compare all verification strategies
- Cache Testing: Compare performance with/without Redis cache

Generates comprehensive comparison reports and visualizations.

Features:
- Real-time live progress display (updates every second)
- Rich console styling with KPI cards and visual metrics
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis
from rich.align import Align
from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

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


# ============================================================================
# Formatting Constants (from console.py style)
# ============================================================================

@dataclass(frozen=True)
class _Fmt:
    """Color scheme consistent with ConsoleReporter."""
    good: str = "green"
    warn: str = "yellow"
    bad: str = "red"
    dim: str = "dim"
    cyan: str = "cyan"
    accent: str = "bright_magenta"


FMT = _Fmt()


# ============================================================================
# Live Benchmark Display
# ============================================================================

@dataclass
class LiveStats:
    """Real-time benchmark statistics."""
    # Overall progress
    total_evaluators: int = 0
    completed_evaluators: int = 0
    current_evaluator: str = ""
    current_metric: str = ""
    
    # Current task progress
    current_total: int = 0
    current_completed: int = 0
    current_correct: int = 0
    current_errors: int = 0
    
    # Timing
    start_time: float = field(default_factory=time.perf_counter)
    current_latencies: list[float] = field(default_factory=list)
    
    # Accumulated results
    correct: int = 0
    errors: int = 0
    total_processed: int = 0
    
    # Per-evaluator results
    evaluator_results: dict[str, dict[str, Any]] = field(default_factory=dict)


class LiveBenchmarkDisplay:
    """
    Real-time benchmark display with Rich Live.
    
    Updates every second with:
    - Overall progress (evaluators, metrics)
    - Current task progress bar
    - Live KPI cards (throughput, accuracy, latency)
    - Running statistics table
    """
    
    # Git Bash on Windows in VS Code can freeze with high refresh rates.
    # We use a conservative rate and manual refreshing to prevent buffer overflows.
    REFRESH_RATE = 2.0
    
    def __init__(self, console: Console, stats: LiveStats) -> None:
        self.console = console
        self.stats = stats
        self._live: Live | None = None
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self._task_id: int | None = None
    
    def __enter__(self) -> "LiveBenchmarkDisplay":
        """Start the live display."""
        # Use auto_refresh=False and manual updates to prevent freezing in some formatting contexts
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.REFRESH_RATE,
            auto_refresh=False,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args) -> None:
        """Stop the live display."""
        if self._live:
            self._live.__exit__(*args)
    
    def start_task(self, description: str, total: int) -> None:
        """Start a new progress task."""
        self.stats.current_metric = description
        self.stats.current_total = total
        self.stats.current_completed = 0
        self.stats.current_correct = 0
        self.stats.current_errors = 0
        self.stats.current_latencies = []
        
        # Reset or create progress task
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(description, total=total)
        self._update()
    
    def advance(self, n: int = 1, latency_ms: float | None = None) -> None:
        """Advance progress by n items."""
        self.stats.current_completed += n
        self.stats.total_processed += n
        
        if latency_ms is not None:
            self.stats.current_latencies.append(latency_ms)
        
        if self._task_id is not None:
            self._progress.update(self._task_id, advance=n)
        self._update()
    
    def set_evaluator(self, name: str) -> None:
        """Set current evaluator being tested."""
        self.stats.current_evaluator = name
        if name not in self.stats.evaluator_results:
            self.stats.evaluator_results[name] = {
                "accuracy": 0.0,
                "f1": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "status": "running",
            }
        else:
            self.stats.evaluator_results[name]["status"] = "running"
        self._update()
    
    def complete_evaluator(self, name: str, metrics: dict[str, Any]) -> None:
        """Mark an evaluator as complete with its metrics."""
        self.stats.completed_evaluators += 1
        metrics["status"] = "complete"
        self.stats.evaluator_results[name] = metrics
        self._update()
    
    def add_result(self, correct: bool, error: bool = False) -> None:
        """Record a result."""
        if correct:
            self.stats.correct += 1
            self.stats.current_correct += 1
        if error:
            self.stats.errors += 1
            self.stats.current_errors += 1
        self._update()
    
    def _update(self) -> None:
        """Update the live display."""
        if self._live:
            self._live.update(self._render(), refresh=True)
    
    def _render(self) -> Group:
        """Render the full display."""
        return Group(
            self._render_header(),
            self._render_progress(),
            self._render_kpis(),
            self._render_results_table(),
        )
    
    def _render_header(self) -> Panel:
        """Render header panel."""
        elapsed = time.perf_counter() - self.stats.start_time
        
        status = "[bold green]● RUNNING[/bold green]"
        if self.stats.completed_evaluators == self.stats.total_evaluators:
            status = "[bold cyan]✓ COMPLETE[/bold cyan]"
        
        lines = [
            f"{status}  [dim]Evaluator[/dim] [bold]{self.stats.current_evaluator or 'initializing'}[/bold]",
            f"[dim]Progress[/dim] {self.stats.completed_evaluators}/{self.stats.total_evaluators} evaluators • [dim]elapsed[/dim] {elapsed:.1f}s",
        ]
        
        return Panel(
            Align.left("\n".join(lines)),
            border_style="cyan",
            box=ROUNDED,
            padding=(0, 2),
        )
    
    def _render_progress(self) -> Panel:
        """Render progress bar panel."""
        return Panel(
            self._progress,
            title=f"[dim]{self.stats.current_metric}[/dim]",
            border_style="dim",
            box=ROUNDED,
            padding=(0, 1),
        )
    
    def _render_kpis(self) -> Columns:
        """Render live KPI cards."""
        elapsed = time.perf_counter() - self.stats.start_time
        
        # Calculate live metrics
        throughput = self.stats.total_processed / elapsed if elapsed > 0 else 0.0
        accuracy = (
            (self.stats.current_correct / self.stats.current_completed * 100)
            if self.stats.current_completed > 0
            else 0.0
        )
        
        # Latency stats
        p50 = p95 = avg_lat = 0.0
        if self.stats.current_latencies:
            sorted_lat = sorted(self.stats.current_latencies)
            avg_lat = statistics.mean(sorted_lat)
            p50 = sorted_lat[int(len(sorted_lat) * 0.5)] if sorted_lat else 0
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0
        
        cards = [
            self._kpi_card("⚡ Throughput", f"{throughput:.2f} req/s", style="bold cyan"),
            self._kpi_card("🎯 Accuracy", f"{accuracy:.1f}%", 
                          style="bold green" if accuracy >= 80 else ("bold yellow" if accuracy >= 60 else "bold red")),
            self._kpi_card("⏱️ P50 / P95", f"{p50:.0f}ms / {p95:.0f}ms"),
            self._kpi_card("📊 Processed", f"{self.stats.current_completed}", style="bold"),
            self._kpi_card("⚠️ Errors", f"{self.stats.current_errors}", 
                          style="bold red" if self.stats.current_errors > 0 else "dim"),
        ]
        
        return Columns(cards, equal=True, expand=True)
    
    def _kpi_card(self, title: str, value: str, style: str = "") -> Panel:
        """Create a styled KPI card (consistent with ConsoleReporter)."""
        txt = Text()
        txt.append(title + "\n", style="dim")
        txt.append(str(value), style=style or "bold")
        return Panel(txt, box=ROUNDED, padding=(0, 1), border_style="dim")
    
    def _render_results_table(self) -> Panel:
        """Render completed evaluator results table."""
        if not self.stats.evaluator_results:
            return Panel("[dim]No results yet...[/dim]", border_style="dim", box=ROUNDED)
        
        table = Table(box=ROUNDED, expand=True, show_header=True, header_style="bold cyan")
        table.add_column("Evaluator", style="cyan", no_wrap=True)
        table.add_column("Accuracy", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("Status", justify="center")
        
        for name, metrics in self.stats.evaluator_results.items():
            acc = metrics.get("accuracy", 0) * 100
            f1 = metrics.get("f1", 0) * 100
            p50 = metrics.get("p50", 0)
            p95 = metrics.get("p95", 0)
            status = metrics.get("status", "running")
            
            acc_style = "green" if acc >= 80 else ("yellow" if acc >= 60 else "red")
            status_label = "[green]✓[/green]" if status == "complete" else "[yellow]…[/yellow]"
            
            table.add_row(
                name,
                f"[{acc_style}]{acc:.1f}%[/{acc_style}]",
                f"{f1:.1f}%",
                f"{p50:.0f}ms",
                f"{p95:.0f}ms",
                status_label,
            )
        
        return Panel(table, title="[dim]Completed Evaluators[/dim]", border_style="dim", box=ROUNDED)


# ============================================================================
# Main Runner Class
# ============================================================================


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

        # Track current report for partial saves
        self._current_report: ComparisonReport | None = None
        
        # Redis client for cache testing
        self._redis: redis.Redis | None = None
        
        # Output directory
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Hard stop for stuck requests (5 minutes per text)
        self._max_latency_ms = 5 * 60 * 1000
    
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
        try:
            self._redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=True,
            )
            self._redis.ping()
            self.console.print(
                f"  [green]✓[/green] Redis connected ({self.config.redis_host}:{self.config.redis_port})"
            )
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

    def _flush_cache_full(self, reason: str) -> None:
        """Flush Redis cache fully (all keys in current DB)."""
        if not self._redis:
            return
        try:
            self._redis.flushdb()
            logger.info(f"Redis cache flushed ({reason})")
        except Exception as e:
            logger.warning(f"Failed to flush Redis cache: {e}")

    def _effective_concurrency(self, evaluator: BaseEvaluator) -> int:
        """Determine evaluator-specific concurrency cap."""
        configured = self.config.concurrency
        max_concurrency = getattr(evaluator, "max_concurrency", None)
        if isinstance(max_concurrency, int) and max_concurrency > 0:
            return max(1, min(configured, max_concurrency))
        return max(1, configured)
    
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
                evaluator = get_evaluator(
                    eval_name,
                    self.config,
                    fair_mode=self.config.vector_rag_fair_mode,
                )
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
                "vector_rag_fair_mode": self.config.vector_rag_fair_mode,
            },
        )
        self._current_report = report
        
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

    async def save_partial_results(self) -> bool:
        """Persist partial results if available."""
        if not self._current_report:
            return False
        await self._generate_outputs(self._current_report)
        return True
    
    async def _run_standard_comparison(self, report: ComparisonReport) -> None:
        """Run standard evaluator comparison with live display."""
        # Initialize live stats
        stats = LiveStats(
            total_evaluators=len(self._evaluators),
            start_time=time.perf_counter(),
        )
        
        with LiveBenchmarkDisplay(self.console, stats) as display:
            for _eval_name, evaluator in self._evaluators.items():
                display.set_evaluator(evaluator.name)
                
                metrics = await self._benchmark_evaluator_live(evaluator, display, stats)
                report.add_evaluator(metrics)
                
                # Record completed evaluator
                display.complete_evaluator(
                    evaluator.name,
                    {
                        "accuracy": metrics.hallucination.accuracy,
                        "f1": metrics.hallucination.f1_score,
                        "p50": metrics.latency.p50,
                        "p95": metrics.latency.p95,
                    }
                )
    
    async def _run_strategy_comparison(self, report: ComparisonReport) -> None:
        """
        Run OHI strategy comparison mode with live display.
        
        Tests each verification strategy separately to find the optimal one.
        """
        self.console.print(Panel(
            f"[bold yellow]OHI Strategy Comparison Mode[/bold yellow]\n"
            f"Testing {len(self.config.ohi_strategies)} strategies",
            border_style="yellow",
        ))
        
        # Calculate total evaluators: non-OHI + OHI strategies
        non_ohi_count = sum(1 for e in self._evaluators if e != "ohi")
        total_evals = non_ohi_count + len(self.config.ohi_strategies)
        
        stats = LiveStats(
            total_evaluators=total_evals,
            start_time=time.perf_counter(),
        )
        
        with LiveBenchmarkDisplay(self.console, stats) as display:
            # Run non-OHI evaluators first
            for eval_name, evaluator in self._evaluators.items():
                if eval_name != "ohi":
                    display.set_evaluator(evaluator.name)
                    metrics = await self._benchmark_evaluator_live(evaluator, display, stats)
                    report.add_evaluator(metrics)
                    display.complete_evaluator(
                        evaluator.name,
                        {
                            "accuracy": metrics.hallucination.accuracy,
                            "f1": metrics.hallucination.f1_score,
                            "p50": metrics.latency.p50,
                            "p95": metrics.latency.p95,
                        }
                    )
            
            # Run each OHI strategy
            ohi_evaluator = self._evaluators.get("ohi")
            if ohi_evaluator:
                for strategy in self.config.ohi_strategies:
                    display.set_evaluator(f"OHI ({strategy})")
                    
                    # Create strategy-specific evaluator
                    strategy_config = ComparisonBenchmarkConfig.from_env()
                    strategy_config.ohi_strategy = strategy
                    
                    from benchmark.evaluators import OHIEvaluator
                    strategy_evaluator = OHIEvaluator(strategy_config)
                    
                    try:
                        if await strategy_evaluator.health_check():
                            metrics = await self._benchmark_evaluator_live(strategy_evaluator, display, stats)
                            metrics.evaluator_name = f"OHI ({strategy})"
                            report.add_evaluator(metrics)
                            display.complete_evaluator(
                                f"OHI ({strategy})",
                                {
                                    "accuracy": metrics.hallucination.accuracy,
                                    "f1": metrics.hallucination.f1_score,
                                    "p50": metrics.latency.p50,
                                    "p95": metrics.latency.p95,
                                }
                            )
                    finally:
                        await strategy_evaluator.close()
    
    async def _run_cache_comparison(self, report: ComparisonReport) -> None:
        """
        Run cache comparison mode with live display.
        
        Tests each evaluator twice:
        1. Cold cache: Cache cleared before run
        2. Warm cache: Cache populated from previous run
        """
        self.console.print(Panel(
            "[bold yellow]Cache Testing Mode[/bold yellow]\n"
            "Testing with cold vs warm cache",
            border_style="yellow",
        ))
        
        # Each evaluator runs twice (cold + warm)
        stats = LiveStats(
            total_evaluators=len(self._evaluators) * 2,
            start_time=time.perf_counter(),
        )
        
        with LiveBenchmarkDisplay(self.console, stats) as display:
            for _eval_name, evaluator in self._evaluators.items():
                # Test 1: Cold cache (cleared before run)
                display.set_evaluator(f"{evaluator.name} (Cold)")
                
                deleted = self._clear_cache()
                logger.info(f"Cleared {deleted} cache keys")
                
                metrics_cold = await self._benchmark_evaluator_live(evaluator, display, stats)
                metrics_cold.evaluator_name = f"{evaluator.name} (Cold)"
                report.add_evaluator(metrics_cold)
                
                display.complete_evaluator(
                    f"{evaluator.name} (Cold)",
                    {
                        "accuracy": metrics_cold.hallucination.accuracy,
                        "f1": metrics_cold.hallucination.f1_score,
                        "p50": metrics_cold.latency.p50,
                        "p95": metrics_cold.latency.p95,
                    }
                )
                
                # Test 2: Warm cache (use cache from previous run)
                display.set_evaluator(f"{evaluator.name} (Warm)")
                
                metrics_warm = await self._benchmark_evaluator_live(evaluator, display, stats)
                metrics_warm.evaluator_name = f"{evaluator.name} (Warm)"
                report.add_evaluator(metrics_warm)
                
                display.complete_evaluator(
                    f"{evaluator.name} (Warm)",
                    {
                        "accuracy": metrics_warm.hallucination.accuracy,
                        "f1": metrics_warm.hallucination.f1_score,
                        "p50": metrics_warm.latency.p50,
                        "p95": metrics_warm.latency.p95,
                    }
                )
                
                # Print cache impact after live display
                if metrics_cold.latency.p50 > 0 and metrics_warm.latency.p50 > 0:
                    speedup = metrics_cold.latency.p50 / metrics_warm.latency.p50
                    logger.info(
                        f"Cache speedup for {evaluator.name}: {speedup:.2f}x "
                        f"(P50: {metrics_cold.latency.p50:.0f}ms → {metrics_warm.latency.p50:.0f}ms)"
                    )
        
        # Print cache comparison summary
        self.console.print(Rule("📊 Cache Impact Summary", style="cyan"))
        for eval_name in self._evaluators:
            cold_name = f"{eval_name} (Cold)"
            warm_name = f"{eval_name} (Warm)"
            if cold_name in report.evaluators and warm_name in report.evaluators:
                cold = report.evaluators[cold_name]
                warm = report.evaluators[warm_name]
                if cold.latency.p50 > 0 and warm.latency.p50 > 0:
                    speedup = cold.latency.p50 / warm.latency.p50
                    self.console.print(
                        f"  [green]✓[/green] {eval_name}: {speedup:.2f}x speedup "
                        f"(P50: {cold.latency.p50:.0f}ms → {warm.latency.p50:.0f}ms)"
                    )
    
    async def _benchmark_evaluator_live(
        self,
        evaluator: BaseEvaluator,
        display: LiveBenchmarkDisplay,
        stats: LiveStats,
    ) -> EvaluatorMetrics:
        """
        Run all benchmarks for a single evaluator with live display updates.
        
        Args:
            evaluator: The evaluator to benchmark.
            display: Live display for progress updates.
            stats: Shared statistics object.
            
        Returns:
            EvaluatorMetrics with all metric results.
        """
        metrics = EvaluatorMetrics(evaluator_name=evaluator.name)

        await self._warmup_evaluator(evaluator)
        self._flush_cache_full(f"after:warmup:{evaluator.name}")
        
        # 1. Hallucination Detection
        if "hallucination" in self.config.metrics:
            start = time.perf_counter()
            halluc_metrics, latencies = await self._run_hallucination_benchmark_live(
                evaluator, display, stats
            )
            metrics.hallucination = halluc_metrics
            metrics.latency.latencies_ms.extend(latencies)
            self._flush_cache_full(f"after:hallucination:{evaluator.name}")
            logger.info(
                f"{evaluator.name} hallucination metric took {time.perf_counter() - start:.2f}s"
            )
        
        # 2. TruthfulQA
        if "truthfulqa" in self.config.metrics:
            start = time.perf_counter()
            tqa_metrics, latencies = await self._run_truthfulqa_benchmark_live(
                evaluator, display, stats
            )
            metrics.truthfulqa = tqa_metrics
            metrics.latency.latencies_ms.extend(latencies)
            self._flush_cache_full(f"after:truthfulqa:{evaluator.name}")
            logger.info(
                f"{evaluator.name} truthfulqa metric took {time.perf_counter() - start:.2f}s"
            )
        
        # 3. FActScore
        if "factscore" in self.config.metrics:
            start = time.perf_counter()
            fac_metrics, latencies = await self._run_factscore_benchmark_live(
                evaluator, display, stats
            )
            metrics.factscore = fac_metrics
            metrics.latency.latencies_ms.extend(latencies)
            self._flush_cache_full(f"after:factscore:{evaluator.name}")
            logger.info(
                f"{evaluator.name} factscore metric took {time.perf_counter() - start:.2f}s"
            )
        
        if metrics.latency.latencies_ms:
            metrics.latency.latencies_ms.pop(0)

        return metrics

    def _guard_latency(self, latency_ms: float, evaluator_name: str, label: str) -> None:
        if latency_ms >= self._max_latency_ms:
            raise RuntimeError(
                f"Latency exceeded {self._max_latency_ms / 1000:.0f}s "
                f"for {evaluator_name} ({label}). Aborting benchmark."
            )

    async def _verify_with_timeout(
        self,
        evaluator: BaseEvaluator,
        claim: str,
        label: str,
    ) -> EvaluatorResult:
        try:
            return await asyncio.wait_for(
                evaluator.verify(claim),
                timeout=self._max_latency_ms / 1000,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Latency exceeded {self._max_latency_ms / 1000:.0f}s "
                f"for {evaluator.name} ({label}). Aborting benchmark."
            ) from exc

    async def _decompose_with_timeout(
        self,
        evaluator: BaseEvaluator,
        text: str,
        label: str,
    ) -> FActScoreResult:
        try:
            return await asyncio.wait_for(
                evaluator.decompose_and_verify(text),
                timeout=self._max_latency_ms / 1000,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Latency exceeded {self._max_latency_ms / 1000:.0f}s "
                f"for {evaluator.name} ({label}). Aborting benchmark."
            ) from exc

    async def _warmup_evaluator(self, evaluator: BaseEvaluator) -> None:
        """Warm up evaluator API and model before measuring metrics."""
        warmup_requests = max(0, int(self.config.warmup_requests))
        if warmup_requests == 0:
            return

        samples = [
            "The Eiffel Tower is in Paris.",
            "Albert Einstein was born in 1879.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Python was created by Guido van Rossum.",
            "The Earth orbits the Sun.",
        ]
        warmup_claims = samples[: min(warmup_requests, len(samples))]

        try:
            concurrency = min(self._effective_concurrency(evaluator), len(warmup_claims))
            if hasattr(evaluator, "verify_batch"):
                await evaluator.verify_batch(warmup_claims, concurrency=concurrency)
            else:
                for claim in warmup_claims:
                    await evaluator.verify(claim)

            if "factscore" in self.config.metrics:
                warmup_text = (
                    "The Moon orbits the Earth. It reflects sunlight and affects tides."
                )
                await evaluator.decompose_and_verify(warmup_text)

        except Exception as e:
            logger.warning(f"Warmup failed for {evaluator.name}: {e}")
    
    # ========================================================================
    # Live Benchmark Methods (with real-time display updates)
    # ========================================================================
    
    async def _run_hallucination_benchmark_live(
        self,
        evaluator: BaseEvaluator,
        display: LiveBenchmarkDisplay,
        stats: LiveStats,
    ) -> tuple[HallucinationMetrics, list[float]]:
        """Run hallucination detection benchmark with live display."""
        loader = HallucinationLoader(self.config.hallucination_dataset)
        
        # Load dataset
        try:
            if self.config.extended_dataset and self.config.extended_dataset.exists():
                dataset = loader.load_combined(
                    csv_path=self.config.hallucination_dataset,
                    include_huggingface=False,
                    hf_max_samples=self.config.hallucination_max_samples,
                )
            elif self.config.hallucination_dataset and self.config.hallucination_dataset.exists():
                dataset = loader.load_csv()
            else:
                dataset = loader.load_from_huggingface(
                    max_samples=self.config.hallucination_max_samples
                )
        except FileNotFoundError:
            # Try loading from HuggingFace only
            dataset = loader.load_from_huggingface(
                max_samples=self.config.hallucination_max_samples
            )
        
        metrics = HallucinationMetrics(total=dataset.total)
        latencies: list[float] = []
        
        # Start task in live display
        display.start_task(
            f"Hallucination Detection ({evaluator.name})",
            total=len(dataset.cases)
        )
        
        # Process concurrently
        semaphore = asyncio.Semaphore(self._effective_concurrency(evaluator))

        async def _process_case(case) -> tuple[EvaluatorResult, Any]:
            async with semaphore:
                res = await self._verify_with_timeout(
                    evaluator,
                    case.text,
                    "hallucination",
                )
                return res, case

        tasks = [_process_case(case) for case in dataset.cases]

        for future in asyncio.as_completed(tasks):
            result, case = await future

            latencies.append(result.latency_ms)
            self._guard_latency(result.latency_ms, evaluator.name, "hallucination")
            
            # Compare prediction to ground truth
            predicted = result.predicted_label
            expected = case.label
            
            is_correct = False
            if predicted and expected:
                metrics.true_positives += 1
                metrics.correct += 1
                is_correct = True
            elif not predicted and not expected:
                metrics.true_negatives += 1
                metrics.correct += 1
                is_correct = True
            elif predicted and not expected:
                metrics.false_positives += 1  # Dangerous!
            else:
                metrics.false_negatives += 1
            
            # Update live display
            display.advance(1, latency_ms=result.latency_ms)
            display.add_result(correct=is_correct, error=result.error is not None)
        
        return metrics, latencies
    
    async def _run_truthfulqa_benchmark_live(
        self,
        evaluator: BaseEvaluator,
        display: LiveBenchmarkDisplay,
        stats: LiveStats,
    ) -> tuple[TruthfulQAMetrics, list[float]]:
        """Run TruthfulQA benchmark with live display."""
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
        
        # Start task in live display
        display.start_task(
            f"TruthfulQA ({evaluator.name})",
            total=len(claims)
        )
        
        # Function to process single item
        semaphore = asyncio.Semaphore(self._effective_concurrency(evaluator))

        async def _process_item(item: tuple[str, bool]) -> tuple[EvaluatorResult, bool]:
            claim_text, is_correct = item
            async with semaphore:
                res = await self._verify_with_timeout(
                    evaluator,
                    claim_text,
                    "truthfulqa",
                )
                return res, is_correct

        # Process claims concurrently and update live
        tasks = [_process_item(item) for item in claims]
        
        for future in asyncio.as_completed(tasks):
            result, is_correct = await future
            
            latencies.append(result.latency_ms)
            self._guard_latency(result.latency_ms, evaluator.name, "truthfulqa")
            
            # Check if evaluator agrees with ground truth
            predicted = result.predicted_label
            correct = (predicted == is_correct)
            if correct:
                metrics.correct_predictions += 1
            
            # Update live display
            display.advance(1, latency_ms=result.latency_ms)
            is_error = result.error is not None
            if is_error:
                logger.error(f"TruthfulQA Error (Claim: ...): {result.error}")
            
            display.add_result(correct=correct, error=is_error)
        
        return metrics, latencies
    
    async def _run_factscore_benchmark_live(
        self,
        evaluator: BaseEvaluator,
        display: LiveBenchmarkDisplay,
        stats: LiveStats,
    ) -> tuple[FActScoreMetrics, list[float]]:
        """Run FActScore benchmark with live display."""
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
        
        # Start task in live display
        display.start_task(
            f"FActScore ({evaluator.name})",
            total=len(sample_texts)
        )
        
        semaphore = asyncio.Semaphore(self._effective_concurrency(evaluator))

        async def _process_text(text: str) -> FActScoreResult:
            async with semaphore:
                return await self._decompose_with_timeout(
                    evaluator,
                    text,
                    "factscore",
                )

        tasks = [_process_text(text) for text in sample_texts]

        for future in asyncio.as_completed(tasks):
            result = await future
            latencies.append(result.latency_ms)
            self._guard_latency(result.latency_ms, evaluator.name, "factscore")
            
            metrics.total_facts += result.total_facts
            metrics.supported_facts += result.supported_facts
            metrics.facts_per_text.append(result.total_facts)
            
            if result.total_facts > 0:
                metrics.scores.append(result.factscore)
            
            # Update live display
            display.advance(1, latency_ms=result.latency_ms)
            display.add_result(correct=True)  # FActScore doesn't have pass/fail
        
        return metrics, latencies
    
    # ========================================================================
    # Legacy Benchmark Methods (without live display, kept for compatibility)
    # ========================================================================
    
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
                    hf_max_samples=self.config.hallucination_max_samples,
                )
            elif self.config.hallucination_dataset and self.config.hallucination_dataset.exists():
                dataset = loader.load_csv()
            else:
                dataset = loader.load_from_huggingface(
                    max_samples=self.config.hallucination_max_samples
                )
        except FileNotFoundError:
            # Try loading from HuggingFace only
            dataset = loader.load_from_huggingface(
                max_samples=self.config.hallucination_max_samples
            )
        
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
            batch_size = self._effective_concurrency(evaluator)
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
        json_payload = json_module.dumps(report.to_dict(), indent=2)
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
