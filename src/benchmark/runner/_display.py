"""
Live benchmark display using Rich library.

Provides real-time progress visualization with:
- Overall progress header
- Current task progress bar
- Live KPI cards (throughput, accuracy, latency)
- Completed evaluator results table

Optimized for:
- Docker exec environments
- Git Bash within VS Code
- Standard terminal emulators

Design Philosophy:
- Decoupled from benchmark execution logic
- Throttled updates to prevent UI freezing
- Thread-safe for async contexts
- Graceful fallback for limited terminals
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from rich.align import Align
from rich.box import ROUNDED, SIMPLE, ASCII
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


def _detect_terminal_capabilities() -> dict[str, bool]:
    """
    Detect terminal capabilities for optimal rendering.
    
    Returns:
        Dict with capability flags:
        - unicode: Can render Unicode characters
        - colors: Supports ANSI colors
        - live: Supports live updating (cursor control)
        - wide: Terminal is at least 80 columns
    """
    # Check for Docker environment
    in_docker = os.path.exists("/.dockerenv") or bool(os.environ.get("DOCKER_CONTAINER"))
    
    # Check terminal type
    term = os.environ.get("TERM", "").lower()
    colorterm = os.environ.get("COLORTERM", "").lower()
    
    # Git Bash detection
    in_gitbash = "MINGW" in os.environ.get("MSYSTEM", "") or "msys" in term
    
    # VS Code integrated terminal
    in_vscode = os.environ.get("TERM_PROGRAM") == "vscode"
    
    # Force color support for common environments
    force_color = os.environ.get("FORCE_COLOR", "").lower() in ("1", "true", "yes")
    
    # Detect capabilities
    has_unicode = (
        sys.stdout.encoding
        and sys.stdout.encoding.lower() in ("utf-8", "utf8")
    )
    
    has_colors = (
        force_color
        or colorterm in ("truecolor", "24bit")
        or "256color" in term
        or "color" in term
        or in_vscode
        or in_docker  # Docker usually supports colors
    )
    
    # Live updates work in most modern terminals
    # But can be problematic in piped/redirected output
    has_live = sys.stdout.isatty() and not os.environ.get("CI")
    
    # Width detection
    try:
        width = os.get_terminal_size().columns
        is_wide = width >= 80
    except OSError:
        is_wide = True  # Assume wide if can't detect
    
    return {
        "unicode": has_unicode,
        "colors": has_colors,
        "live": has_live,
        "wide": is_wide,
        "docker": in_docker,
        "gitbash": in_gitbash,
        "vscode": in_vscode,
    }


def create_optimized_console() -> Console:
    """
    Create a Rich Console optimized for the current environment.
    
    Handles Docker exec, Git Bash, and VS Code integrated terminal.
    
    Returns:
        Configured Console instance
    """
    caps = _detect_terminal_capabilities()
    
    # Force certain settings for problematic environments
    force_terminal = caps["colors"] or caps["docker"] or caps["vscode"]
    
    return Console(
        force_terminal=force_terminal,
        force_interactive=caps["live"],
        color_system="auto" if caps["colors"] else None,
        width=120 if not caps["wide"] else None,
        legacy_windows=False,  # We handle Git Bash separately
        soft_wrap=True,
    )


class LiveBenchmarkDisplay:
    """
    Real-time benchmark display with Rich Live.
    
    Updates dynamically with:
    - Overall progress (evaluators, metrics)
    - Current task progress bar
    - Live KPI cards (throughput, accuracy, latency)
    - Running statistics table
    
    Optimized for Docker exec and Git Bash environments with:
    - ASCII fallback for box drawing
    - Reduced refresh rate to prevent flickering
    - Simpler spinners that render reliably
    - Graceful degradation for limited terminals
    
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
    
    # Background refresh rate (Hz) - very low to prevent jitter
    REFRESH_RATE: float = 1.0
    
    # Minimum interval between manual updates (seconds) - higher to reduce flicker
    UPDATE_THROTTLE: float = 1.0
    
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
        
        # Detect terminal capabilities
        self._caps = _detect_terminal_capabilities()
        
        # Choose appropriate box style
        self._box = ROUNDED if self._caps["unicode"] else ASCII
        self._simple_box = SIMPLE if self._caps["unicode"] else ASCII
        
        # Choose spinner - dots work better in Docker/Git Bash
        spinner_name = "dots" if self._caps["unicode"] else "line"
        
        # Progress bar component with environment-aware configuration
        self._progress = Progress(
            SpinnerColumn(spinner_name=spinner_name),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]|[/dim]"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            transient=False,
        )
        self._task_id: int | None = None
    
    def __enter__(self) -> LiveBenchmarkDisplay:
        """Start the live display context."""
        # Use very low refresh rate to minimize jitter in Docker/Git Bash
        # We rely on manual _update() calls for responsiveness
        refresh_rate = 1.0
        
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=refresh_rate,
            auto_refresh=False,  # Disable auto-refresh to prevent jitter
            transient=False,
            vertical_overflow="ellipsis",  # Prevent height changes
            # Don't redirect in Docker - causes issues
            redirect_stdout=False,
            redirect_stderr=False,
            screen=False,  # Don't use alternate screen
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop the live display context."""
        if self._live:
            # Final render before exit
            self._live.update(self._render(), refresh=True)
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
        """Request a display refresh (still respects minimum throttle to prevent jitter)."""
        self._update(force=False)  # Use normal throttling to prevent jitter
    
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
            try:
                self._live.update(self._render(), refresh=True)
                self._last_update_time = now
            except Exception:
                # Silently handle rendering errors in constrained environments
                pass
    
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
        
        # Use ASCII-safe status indicators
        if self.stats.completed_evaluators == self.stats.total_evaluators:
            status_icon = "[+]" if not self._caps["unicode"] else "✓"
            status = f"[bold {COLORS.cyan}]{status_icon} COMPLETE[/bold {COLORS.cyan}]"
        else:
            status_icon = "[>]" if not self._caps["unicode"] else "●"
            status = f"[bold {COLORS.good}]{status_icon} RUNNING[/bold {COLORS.good}]"
        
        evaluator_name = self.stats.current_evaluator or "initializing"
        
        # Format elapsed time nicely
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        lines = [
            f"{status}  [dim]Evaluator:[/dim] [bold white]{evaluator_name}[/bold white]",
            f"[dim]Progress:[/dim] {self.stats.completed_evaluators}/{self.stats.total_evaluators} evaluators  [dim]|[/dim]  [dim]Elapsed:[/dim] {time_str}",
        ]
        
        return Panel(
            Align.left("\n".join(lines)),
            title="[bold cyan]OHI Benchmark[/bold cyan]",
            border_style="cyan",
            box=self._box,
            padding=(0, 2),
        )
    
    def _render_progress(self) -> Panel:
        """Render the progress bar panel."""
        return Panel(
            self._progress,
            title=f"[dim]{self.stats.current_metric}[/dim]",
            border_style="blue",
            box=self._simple_box,
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
            p95 = sorted_lat[min(int(n * 0.95), n - 1)] if n > 0 else 0
        
        # Style accuracy based on value
        if accuracy >= 80:
            acc_style = f"bold {COLORS.good}"
        elif accuracy >= 60:
            acc_style = f"bold {COLORS.warn}"
        else:
            acc_style = f"bold {COLORS.bad}"
        
        error_style = f"bold {COLORS.bad}" if self.stats.current_errors > 0 else "dim"
        
        # Use ASCII-safe icons
        if self._caps["unicode"]:
            icons = {"speed": "⚡", "target": "🎯", "time": "⏱️", "chart": "📊", "warn": "⚠️"}
        else:
            icons = {"speed": ">", "target": "*", "time": "@", "chart": "#", "warn": "!"}
        
        cards = [
            self._kpi_card(f"{icons['speed']} Throughput", f"{throughput:.2f}/s", style="bold cyan"),
            self._kpi_card(f"{icons['target']} Accuracy", f"{accuracy:.1f}%", style=acc_style),
            self._kpi_card(f"{icons['time']} P50/P95", f"{p50:.0f}/{p95:.0f}ms"),
            self._kpi_card(f"{icons['chart']} Done", f"{self.stats.current_completed}", style="bold"),
            self._kpi_card(f"{icons['warn']} Exceptions", str(self.stats.current_errors), style=error_style),
        ]
        
        return Columns(cards, equal=True, expand=True)
    
    def _kpi_card(self, title: str, value: str, style: str = "") -> Panel:
        """Create a styled KPI card."""
        txt = Text()
        txt.append(f"{title}\n", style="dim")
        txt.append(value, style=style or "bold white")
        return Panel(
            txt,
            box=self._simple_box,
            padding=(0, 1),
            border_style="dim",
        )
    
    def _render_results_table(self) -> Panel:
        """Render completed evaluator results table."""
        if not self.stats.evaluator_results:
            return Panel(
                "[dim]Waiting for results...[/dim]",
                border_style="dim",
                box=self._simple_box,
            )
        
        table = Table(
            box=self._box,
            expand=True,
            show_header=True,
            header_style="bold cyan",
            row_styles=["", "dim"],  # Alternate row styling
        )
        table.add_column("Evaluator", style="bold white", no_wrap=True, min_width=12)
        table.add_column("Accuracy", justify="right", min_width=8)
        table.add_column("F1", justify="right", min_width=8)
        table.add_column("P50", justify="right", min_width=8)
        table.add_column("P95", justify="right", min_width=8)
        table.add_column("Status", justify="center", min_width=8)
        
        # Status icons
        if self._caps["unicode"]:
            complete_icon = f"[{COLORS.good}]✓[/{COLORS.good}]"
            running_icon = f"[{COLORS.warn}]…[/{COLORS.warn}]"
        else:
            complete_icon = f"[{COLORS.good}][OK][/{COLORS.good}]"
            running_icon = f"[{COLORS.warn}][..][/{COLORS.warn}]"
        
        for name, metrics in self.stats.evaluator_results.items():
            acc = metrics.get("accuracy", 0) * 100
            f1 = metrics.get("f1", 0) * 100
            p50 = metrics.get("p50", 0)
            p95 = metrics.get("p95", 0)
            status = metrics.get("status", "running")
            
            # Color accuracy based on value
            if acc >= 80:
                acc_style = COLORS.good
            elif acc >= 60:
                acc_style = COLORS.warn
            else:
                acc_style = COLORS.bad
            
            status_icon = complete_icon if status == "complete" else running_icon
            
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
            title="[bold]Evaluator Results[/bold]",
            border_style="cyan",
            box=self._box,
            padding=(0, 0),
        )
