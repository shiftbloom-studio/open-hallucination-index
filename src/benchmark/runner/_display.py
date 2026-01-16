"""
Live benchmark display using Rich library.

Provides real-time progress visualization with:
- Overall progress header
- Current task progress bar
- Live KPI cards (throughput, accuracy, latency)
- Completed evaluator results table

Design Philosophy:
- Decoupled from benchmark execution logic
- Throttled updates to prevent UI freezing
- Thread-safe for async contexts
"""

from __future__ import annotations

import time
from typing import Any

from rich.align import Align
from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
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
from rich.table import Table
from rich.text import Text

from benchmark.runner._types import COLORS, LiveStats


class LiveBenchmarkDisplay:
    """
    Real-time benchmark display with Rich Live.
    
    Updates dynamically with:
    - Overall progress (evaluators, metrics)
    - Current task progress bar
    - Live KPI cards (throughput, accuracy, latency)
    - Running statistics table
    
    Usage:
        ```python
        stats = LiveStats(total_evaluators=3)
        with LiveBenchmarkDisplay(console, stats) as display:
            display.set_evaluator("OHI")
            display.start_task("Hallucination", total=100)
            for i in range(100):
                display.advance(1, latency_ms=50.0)
            display.complete_evaluator("OHI", {"accuracy": 0.95})
        ```
    
    Threading Notes:
        - Uses `auto_refresh=True` for background rendering
        - `_update()` is throttled to prevent blocking the event loop
        - Safe to call from async contexts
    """
    
    # Background refresh rate (Hz)
    REFRESH_RATE: float = 4.0
    
    # Minimum interval between manual updates (seconds)
    UPDATE_THROTTLE: float = 0.5
    
    def __init__(self, console: Console, stats: LiveStats) -> None:
        """
        Initialize the live display.
        
        Args:
            console: Rich Console instance for output
            stats: Shared LiveStats instance for progress tracking
        """
        self.console = console
        self.stats = stats
        self._live: Live | None = None
        self._last_update_time: float = 0.0
        
        # Progress bar component
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
    
    def __enter__(self) -> LiveBenchmarkDisplay:
        """Start the live display context."""
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.REFRESH_RATE,
            auto_refresh=True,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop the live display context."""
        if self._live:
            self._live.__exit__(*args)
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def start_task(self, description: str, total: int) -> None:
        """
        Start a new progress task.
        
        Args:
            description: Task description shown in progress bar
            total: Total number of items to process
        """
        self.stats.current_metric = description
        self.stats.current_total = total
        self.stats.reset_task()
        
        # Reset or create progress task
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(description, total=total)
        self._update(force=True)
    
    def advance(self, n: int = 1, latency_ms: float | None = None) -> None:
        """
        Advance progress by n items.
        
        Args:
            n: Number of items completed
            latency_ms: Optional latency measurement to record
        """
        self.stats.current_completed += n
        self.stats.total_processed += n
        
        if latency_ms is not None:
            self.stats.current_latencies.append(latency_ms)
        
        if self._task_id is not None:
            self._progress.update(self._task_id, advance=n)
        self._update()
    
    def set_evaluator(self, name: str) -> None:
        """
        Set current evaluator being tested.
        
        Args:
            name: Evaluator name to display
        """
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
        self._update(force=True)
    
    def complete_evaluator(self, name: str, metrics: dict[str, Any]) -> None:
        """
        Mark an evaluator as complete with its metrics.
        
        Args:
            name: Evaluator name
            metrics: Dict with accuracy, f1, p50, p95 keys
        """
        self.stats.completed_evaluators += 1
        metrics["status"] = "complete"
        self.stats.evaluator_results[name] = metrics
        self._update(force=True)
    
    def add_result(self, correct: bool, error: bool = False) -> None:
        """
        Record a single result.
        
        Args:
            correct: Whether the prediction was correct
            error: Whether an error occurred
        """
        if correct:
            self.stats.correct += 1
            self.stats.current_correct += 1
        if error:
            self.stats.errors += 1
            self.stats.current_errors += 1
        self._update()
    
    def force_refresh(self) -> None:
        """Force an immediate display refresh (use sparingly)."""
        self._update(force=True)
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _update(self, force: bool = False) -> None:
        """
        Update the live display with throttling.
        
        Args:
            force: Bypass throttling and update immediately
        """
        if not self._live:
            return
        
        now = time.perf_counter()
        if force or (now - self._last_update_time >= self.UPDATE_THROTTLE):
            self._live.update(self._render(), refresh=True)
            self._last_update_time = now
    
    def _render(self) -> Group:
        """Render the complete display layout."""
        return Group(
            self._render_header(),
            self._render_progress(),
            self._render_kpis(),
            self._render_results_table(),
        )
    
    def _render_header(self) -> Panel:
        """Render the header panel with status and progress."""
        elapsed = time.perf_counter() - self.stats.start_time
        
        if self.stats.completed_evaluators == self.stats.total_evaluators:
            status = f"[bold {COLORS.cyan}]✓ COMPLETE[/bold {COLORS.cyan}]"
        else:
            status = f"[bold {COLORS.good}]● RUNNING[/bold {COLORS.good}]"
        
        evaluator_name = self.stats.current_evaluator or "initializing"
        lines = [
            f"{status}  [dim]Evaluator[/dim] [bold]{evaluator_name}[/bold]",
            f"[dim]Progress[/dim] {self.stats.completed_evaluators}/{self.stats.total_evaluators} evaluators • [dim]elapsed[/dim] {elapsed:.1f}s",
        ]
        
        return Panel(
            Align.left("\n".join(lines)),
            border_style="cyan",
            box=ROUNDED,
            padding=(0, 2),
        )
    
    def _render_progress(self) -> Panel:
        """Render the progress bar panel."""
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
        
        # Latency statistics
        p50 = p95 = 0.0
        if self.stats.current_latencies:
            sorted_lat = sorted(self.stats.current_latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[int(n * 0.5)] if n > 0 else 0
            p95 = sorted_lat[int(n * 0.95)] if n > 0 else 0
        
        # Style accuracy based on value
        if accuracy >= 80:
            acc_style = f"bold {COLORS.good}"
        elif accuracy >= 60:
            acc_style = f"bold {COLORS.warn}"
        else:
            acc_style = f"bold {COLORS.bad}"
        
        error_style = f"bold {COLORS.bad}" if self.stats.current_errors > 0 else "dim"
        
        cards = [
            self._kpi_card("⚡ Throughput", f"{throughput:.2f} req/s", style="bold cyan"),
            self._kpi_card("🎯 Accuracy", f"{accuracy:.1f}%", style=acc_style),
            self._kpi_card("⏱️ P50 / P95", f"{p50:.0f}ms / {p95:.0f}ms"),
            self._kpi_card("📊 Processed", str(self.stats.current_completed), style="bold"),
            self._kpi_card("⚠️ Errors", str(self.stats.current_errors), style=error_style),
        ]
        
        return Columns(cards, equal=True, expand=True)
    
    def _kpi_card(self, title: str, value: str, style: str = "") -> Panel:
        """Create a styled KPI card."""
        txt = Text()
        txt.append(f"{title}\n", style="dim")
        txt.append(value, style=style or "bold")
        return Panel(txt, box=ROUNDED, padding=(0, 1), border_style="dim")
    
    def _render_results_table(self) -> Panel:
        """Render completed evaluator results table."""
        if not self.stats.evaluator_results:
            return Panel(
                "[dim]No results yet...[/dim]",
                border_style="dim",
                box=ROUNDED,
            )
        
        table = Table(
            box=ROUNDED,
            expand=True,
            show_header=True,
            header_style="bold cyan",
        )
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
            
            acc_style = COLORS.good if acc >= 80 else (COLORS.warn if acc >= 60 else COLORS.bad)
            status_icon = f"[{COLORS.good}]✓[/{COLORS.good}]" if status == "complete" else f"[{COLORS.warn}]…[/{COLORS.warn}]"
            
            table.add_row(
                name,
                f"[{acc_style}]{acc:.1f}%[/{acc_style}]",
                f"{f1:.1f}%",
                f"{p50:.0f}ms",
                f"{p95:.0f}ms",
                status_icon,
            )
        
        return Panel(
            table,
            title="[dim]Completed Evaluators[/dim]",
            border_style="dim",
            box=ROUNDED,
        )
