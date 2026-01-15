"""
Console Reporter
================

Rich console output for benchmark results with live progress and tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric, StrategyReport
    from benchmark.metrics import ConfusionMatrix


class ConsoleReporter(BaseReporter):
    """
    Rich console reporter with tables and formatting.

    Provides beautiful terminal output for benchmark results.
    """

    def __init__(self, output_dir: Path, console: Console | None = None) -> None:
        """
        Initialize console reporter.

        Args:
            output_dir: Directory for HTML export.
            console: Optional Rich console instance.
        """
        super().__init__(output_dir)
        self.console = console or Console(record=True)

    @property
    def file_extension(self) -> str:
        return "html"

    def generate(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
    ) -> str:
        """Generate HTML from recorded console output."""
        self.display_summary(report)
        return self.console.export_html()

    def display_summary(self, report: BenchmarkReport) -> None:
        """Display full benchmark summary to console."""
        self.console.print("\n")
        self.console.print(
            Panel("[bold green]✔ Benchmark Complete[/bold green]", border_style="green")
        )

        # Main performance table
        self._display_performance_table(report)

        # Calibration table
        self._display_calibration_table(report)

        # Domain analysis for best strategy
        self._display_domain_analysis(report)

        # Worst cases
        self._display_worst_cases(report)

        # Statistical significance
        self._display_significance(report)

        # Runtime info
        if report.strategy_reports:
            total_tests = sum(
                sr.confusion_matrix.total for sr in report.strategy_reports.values()
            )
            if total_tests > 0:
                self.console.print(
                    f"\n[dim]Total runtime: {report.total_runtime_seconds:.1f}s "
                    f"({report.total_runtime_seconds / total_tests:.2f}s per test)[/dim]"
                )

    def _display_performance_table(self, report: BenchmarkReport) -> None:
        """Display main strategy comparison table."""
        table = Table(
            title="📊 Strategy Performance Comparison",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("MCC", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("Halluc.%", justify="right", style="red")
        table.add_column("P95 (ms)", justify="right")
        table.add_column("Errors", justify="right")

        for strat, sr in report.strategy_reports.items():
            cm = sr.confusion_matrix
            is_best_f1 = strat == report.best_strategy_f1

            # Add confidence interval if available
            f1_str = f"{cm.f1_score:.3f}"
            if sr.f1_ci:
                f1_str = f"{cm.f1_score:.3f} ±{sr.f1_ci.margin_of_error:.3f}"

            table.add_row(
                f"{'★ ' if is_best_f1 else ''}{strat}",
                f"{cm.accuracy:.1%}",
                f"{cm.precision:.1%}",
                f"{cm.recall:.1%}",
                f1_str,
                f"{cm.mcc:.3f}",
                f"{sr.roc.auc:.3f}" if sr.roc else "N/A",
                f"{cm.hallucination_pass_rate:.1%}",
                f"{sr.latency.p95_ms:.0f}" if sr.latency else "N/A",
                f"{sr.error_count}",
                style="bold green" if is_best_f1 else "",
            )

        self.console.print(table)

    def _display_calibration_table(self, report: BenchmarkReport) -> None:
        """Display calibration and threshold analysis."""
        table = Table(title="📈 Calibration & Threshold Analysis")
        table.add_column("Strategy", style="cyan")
        table.add_column("Brier Score", justify="right")
        table.add_column("ECE", justify="right")
        table.add_column("Optimal Threshold", justify="right")
        table.add_column("Youden's J", justify="right")

        for strat, sr in report.strategy_reports.items():
            table.add_row(
                strat,
                f"{sr.calibration.brier_score:.4f}" if sr.calibration else "N/A",
                f"{sr.calibration.ece:.4f}" if sr.calibration else "N/A",
                f"{sr.roc.optimal_threshold:.3f}" if sr.roc else "N/A",
                f"{sr.roc.youden_j:.3f}" if sr.roc else "N/A",
            )

        self.console.print(table)

    def _display_domain_analysis(self, report: BenchmarkReport) -> None:
        """Display domain breakdown for best strategy."""
        best_strat = report.best_strategy_f1
        if not best_strat or best_strat not in report.strategy_reports:
            return

        sr = report.strategy_reports[best_strat]
        if not sr.by_domain:
            return

        table = Table(title=f"🏷️ Domain Analysis ({best_strat})")
        table.add_column("Domain", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Halluc.%", justify="right", style="red")
        table.add_column("n", justify="right", style="dim")

        for domain in sorted(sr.by_domain.keys()):
            cm = sr.by_domain[domain]
            table.add_row(
                domain,
                f"{cm.accuracy:.1%}",
                f"{cm.f1_score:.3f}",
                f"{cm.hallucination_pass_rate:.1%}",
                str(cm.total),
            )

        self.console.print(table)

    def _display_worst_cases(self, report: BenchmarkReport) -> None:
        """Display worst false positives."""
        best_strat = report.best_strategy_f1
        if not best_strat or best_strat not in report.strategy_reports:
            return

        sr = report.strategy_reports[best_strat]
        if not sr.worst_fp_cases:
            return

        self.console.print(
            "\n[bold red]⚠️ Worst False Positives (Hallucinations Believed)[/bold red]"
        )
        for i, fp in enumerate(sr.worst_fp_cases[:3], 1):
            self.console.print(
                f"  {i}. [dim]Case {fp['case_id']}[/dim] (score: {fp['trust_score']:.2f})"
            )
            text = fp.get("text", "")[:80]
            self.console.print(f'     "{text}..."')

    def _display_significance(self, report: BenchmarkReport) -> None:
        """Display statistical significance results."""
        if not report.comparisons and not report.mcnemar_results:
            return

        self.console.print("\n[bold]📊 Statistical Significance[/bold]")

        # New comparisons format
        if report.comparisons:
            for comp_name, comp in report.comparisons.items():
                sig_marker = (
                    "[green]✓[/green]" if comp.is_significant else "[dim]✗[/dim]"
                )
                self.console.print(
                    f"  {comp.strategy_a} vs {comp.strategy_b}: "
                    f"McNemar p={comp.mcnemar_p_value:.4f} "
                    f"DeLong p={comp.delong_p_value:.4f} {sig_marker}"
                )
        # Legacy mcnemar_results format
        elif report.mcnemar_results:
            for comp, res in report.mcnemar_results.items():
                sig_marker = (
                    "[green]✓[/green]" if res.get("significant_at_05") else "[dim]✗[/dim]"
                )
                self.console.print(f"  {comp}: p={res.get('p_value', 1.0):.4f} {sig_marker}")

    def save(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
        filename: str | None = None,
    ) -> Path:
        """Save console output as HTML."""
        self.display_summary(report)

        if filename is None:
            filename = f"ohi_benchmark_{report.run_id}"

        filepath = self.output_dir / f"{filename}.html"
        self.console.save_html(str(filepath))

        return filepath
