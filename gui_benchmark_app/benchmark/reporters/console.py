"""
Console Reporter
================

Rich console output for benchmark results with live progress and tables.
Improved layout: hero header, KPI blocks, sorted tables, better styling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.box import ROUNDED
from rich.align import Align
from rich.rule import Rule
from rich.markup import escape

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


@dataclass(frozen=True)
class _Fmt:
    good: str = "green"
    warn: str = "yellow"
    bad: str = "red"
    dim: str = "dim"
    cyan: str = "cyan"
    accent: str = "bright_magenta"


class ConsoleReporter(BaseReporter):
    """
    Rich console reporter with tables and formatting.

    Provides beautiful terminal output for benchmark results.
    """

    def __init__(self, output_dir: Path, console: Console | None = None) -> None:
        super().__init__(output_dir)
        self.console = console or Console(record=True)
        self._fmt = _Fmt()

    @property
    def file_extension(self) -> str:
        return "html"

    def generate(self, report: "BenchmarkReport", results: list["ResultMetric"]) -> str:
        """Generate HTML from recorded console output."""
        self.display_summary(report, results)
        return self.console.export_html()

    def display_summary(self, report: "BenchmarkReport", results: list["ResultMetric"] | None = None) -> None:
        """Display full benchmark summary to console."""
        results = results or []

        self.console.print("\n")
        self._print_hero(report, results)

        self._display_kpis(report, results)

        self.console.print(Rule("📊 Results", style="dim"))
        self._display_performance_table(report)

        self._display_calibration_table(report)

        self._display_domain_analysis(report)

        self._display_worst_cases(report)

        self.console.print(Rule("📎 Statistics", style="dim"))
        self._display_significance(report)

        self._display_runtime(report)

    # -----------------------------
    # Header / KPIs
    # -----------------------------

    def _print_hero(self, report: "BenchmarkReport", results: list["ResultMetric"]) -> None:
        run_id = getattr(report, "run_id", "n/a")
        dataset_size = getattr(report, "dataset_size", None)
        threshold = getattr(report, "threshold_used", None)
        runtime = getattr(report, "total_runtime_seconds", None)

        total = len(results) if results else None
        errors = sum(1 for r in results if getattr(r, "has_error", False)) if results else None

        left_lines = []
        left_lines.append(f"[bold green]✔ Benchmark Complete[/bold green]  [dim]run[/dim] [bold]{run_id}[/bold]")
        meta_bits = []
        if dataset_size is not None:
            meta_bits.append(f"[dim]dataset[/dim] {dataset_size}")
        if threshold is not None:
            meta_bits.append(f"[dim]threshold[/dim] {threshold}")
        if runtime is not None:
            meta_bits.append(f"[dim]runtime[/dim] {runtime:.2f}s")
        if total is not None:
            meta_bits.append(f"[dim]cases[/dim] {total}")
        if errors is not None:
            meta_bits.append(f"[dim]errors[/dim] {errors}")

        if meta_bits:
            left_lines.append(" • ".join(meta_bits))

        title = "\n".join(left_lines)
        panel = Panel(
            Align.left(title),
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
        self.console.print(panel)

    def _display_kpis(self, report: "BenchmarkReport", results: list["ResultMetric"]) -> None:
        # Best strategy by F1
        best = getattr(report, "best_strategy_f1", None)
        sr_best = None
        if best and getattr(report, "strategy_reports", None) and best in report.strategy_reports:
            sr_best = report.strategy_reports[best]

        # Compute global counts
        total = len(results) if results else 0
        errors = sum(1 for r in results if getattr(r, "has_error", False)) if results else 0
        valid = total - errors

        runtime = float(getattr(report, "total_runtime_seconds", 0.0) or 0.0)
        overall_rps = (valid / runtime) if runtime > 0 and valid > 0 else 0.0

        # Extract latency KPIs from best SR if present
        p50 = p95 = avg = None
        if sr_best and getattr(sr_best, "latency", None):
            lat = sr_best.latency
            p50 = getattr(lat, "p50_ms", None)
            p95 = getattr(lat, "p95_ms", None)
            avg = getattr(lat, "avg_ms", None)

        # Extract quality KPIs
        f1 = acc = mcc = halluc = None
        if sr_best and getattr(sr_best, "confusion_matrix", None):
            cm = sr_best.confusion_matrix
            f1 = getattr(cm, "f1_score", None)
            acc = getattr(cm, "accuracy", None)
            mcc = getattr(cm, "mcc", None)
            halluc = getattr(cm, "hallucination_pass_rate", None)

        cards = []
        cards.append(self._kpi_card("🏆 Best Strategy", best or "N/A", style="bold cyan"))
        cards.append(self._kpi_card("✅ Valid / Total", f"{valid} / {total}" if total else "N/A"))
        cards.append(self._kpi_card("⚡ Overall Throughput", f"{overall_rps:.3f} req/s" if overall_rps else "N/A"))
        cards.append(self._kpi_card("🧪 Best F1", self._fmt_float(f1, ".3f"), style="bold green" if f1 and f1 >= 0.8 else "bold"))
        cards.append(self._kpi_card("🎯 Best Acc", self._fmt_pct(acc), style="green" if acc and acc >= 0.8 else ""))
        cards.append(self._kpi_card("🧠 MCC", self._fmt_float(mcc, ".3f")))
        cards.append(self._kpi_card("🧨 Halluc.%", self._fmt_pct(halluc), style="red" if halluc and halluc >= 0.1 else ""))
        cards.append(self._kpi_card("⏱️ P50 / P95", f"{self._fmt_ms(p50)} / {self._fmt_ms(p95)}"))
        cards.append(self._kpi_card("📉 Avg Latency", self._fmt_ms(avg)))

        self.console.print(Columns(cards, equal=True, expand=True))

        # Quick insight line
        if sr_best and getattr(report, "strategy_reports", None):
            insight = self._best_vs_second_insight(report)
            if insight:
                self.console.print(Panel(insight, border_style="dim", box=ROUNDED, padding=(0, 2)))

    def _kpi_card(self, title: str, value: str, style: str = "") -> Panel:
        txt = Text()
        txt.append(title + "\n", style="dim")
        txt.append(str(value), style=style or "bold")
        return Panel(txt, box=ROUNDED, padding=(1, 2), border_style="dim")

    def _best_vs_second_insight(self, report: "BenchmarkReport") -> str | None:
        srs = getattr(report, "strategy_reports", None)
        if not srs:
            return None

        rows = []
        for name, sr in srs.items():
            cm = getattr(sr, "confusion_matrix", None)
            if not cm:
                continue
            f1 = getattr(cm, "f1_score", None)
            p95 = getattr(getattr(sr, "latency", None), "p95_ms", None)
            err = getattr(sr, "error_count", 0)
            if f1 is None:
                continue
            rows.append((name, float(f1), float(p95) if p95 is not None else None, int(err)))

        if len(rows) < 2:
            return None

        rows.sort(key=lambda x: x[1], reverse=True)
        (a, f1a, p95a, erra), (b, f1b, p95b, errb) = rows[0], rows[1]

        delta = f1a - f1b
        parts = [f"[bold]Insight:[/bold] [cyan]{a}[/cyan] leads [cyan]{b}[/cyan] by [bold]{delta:+.3f}[/bold] F1."]

        if p95a is not None and p95b is not None:
            faster = "faster" if p95a < p95b else "slower"
            parts.append(f"Tail latency (P95) is [bold]{faster}[/bold] ({p95a:.0f}ms vs {p95b:.0f}ms).")

        if erra or errb:
            parts.append(f"Errors: {a}={erra}, {b}={errb}.")

        return " ".join(parts)

    # -----------------------------
    # Tables
    # -----------------------------

    def _display_performance_table(self, report: "BenchmarkReport") -> None:
        """Display main strategy comparison table (sorted by F1 desc)."""
        srs = getattr(report, "strategy_reports", None) or {}
        if not srs:
            self.console.print("[dim]No strategy reports available.[/dim]")
            return

        table = Table(
            title="📊 Strategy Performance Comparison",
            show_header=True,
            header_style="bold cyan",
            box=ROUNDED,
            row_styles=["", "dim"],
        )
        table.add_column("Rank", justify="right", style="dim", width=4)
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("Acc", justify="right")
        table.add_column("Prec", justify="right")
        table.add_column("Rec", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("MCC", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("Halluc.%", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("Err", justify="right")

        # Prepare sortable rows
        rows: list[tuple[str, Any]] = []
        for strat, sr in srs.items():
            cm = getattr(sr, "confusion_matrix", None)
            if not cm:
                continue
            f1 = getattr(cm, "f1_score", None)
            rows.append((strat, float(f1) if f1 is not None else -1.0))

        rows.sort(key=lambda x: x[1], reverse=True)

        best = getattr(report, "best_strategy_f1", None)

        for idx, (strat, _f1) in enumerate(rows, 1):
            sr = srs[strat]
            cm = sr.confusion_matrix

            is_best = (best == strat) or (idx == 1 and best is None)

            # F1 string with CI if present
            f1_str = f"{cm.f1_score:.3f}"
            if getattr(sr, "f1_ci", None):
                mo = getattr(sr.f1_ci, "margin_of_error", None)
                if mo is not None:
                    f1_str = f"{cm.f1_score:.3f} ±{mo:.3f}"

            auc_str = "N/A"
            if getattr(sr, "roc", None) and getattr(sr.roc, "auc", None) is not None:
                auc_str = f"{sr.roc.auc:.3f}"

            lat = getattr(sr, "latency", None)
            p50 = getattr(lat, "p50_ms", None) if lat else None
            p95 = getattr(lat, "p95_ms", None) if lat else None

            # Heat-ish styling: hallucination rate
            halluc = getattr(cm, "hallucination_pass_rate", None)
            halluc_style = self._rate_style(halluc)

            # Style row
            row_style = "bold green" if is_best else ""

            table.add_row(
                str(idx),
                f"{'★ ' if is_best else ''}{strat}",
                self._fmt_pct(cm.accuracy),
                self._fmt_pct(cm.precision),
                self._fmt_pct(cm.recall),
                f1_str,
                self._fmt_float(cm.mcc, ".3f"),
                auc_str,
                Text(self._fmt_pct(halluc), style=halluc_style),
                self._fmt_ms(p50),
                self._fmt_ms(p95),
                str(getattr(sr, "error_count", 0)),
                style=row_style,
            )

        self.console.print(table)

    def _display_calibration_table(self, report: "BenchmarkReport") -> None:
        srs = getattr(report, "strategy_reports", None) or {}
        if not srs:
            return

        table = Table(title="📈 Calibration & Threshold Analysis", box=ROUNDED)
        table.add_column("Strategy", style="cyan")
        table.add_column("Brier", justify="right")
        table.add_column("ECE", justify="right")
        table.add_column("Opt Thr", justify="right")
        table.add_column("Youden J", justify="right")

        for strat, sr in srs.items():
            cal = getattr(sr, "calibration", None)
            roc = getattr(sr, "roc", None)
            table.add_row(
                strat,
                self._fmt_float(getattr(cal, "brier_score", None), ".4f"),
                self._fmt_float(getattr(cal, "ece", None), ".4f"),
                self._fmt_float(getattr(roc, "optimal_threshold", None), ".3f"),
                self._fmt_float(getattr(roc, "youden_j", None), ".3f"),
            )

        self.console.print(table)

    def _display_domain_analysis(self, report: "BenchmarkReport") -> None:
        best_strat = getattr(report, "best_strategy_f1", None)
        srs = getattr(report, "strategy_reports", None) or {}
        if not best_strat or best_strat not in srs:
            return

        sr = srs[best_strat]
        by_domain = getattr(sr, "by_domain", None)
        if not by_domain:
            return

        table = Table(title=f"🏷️ Domain Analysis ({best_strat})", box=ROUNDED)
        table.add_column("Domain", style="cyan")
        table.add_column("Acc", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Halluc.%", justify="right")
        table.add_column("n", justify="right", style="dim")

        # Sort domains by n desc
        domains = sorted(by_domain.keys(), key=lambda d: getattr(by_domain[d], "total", 0), reverse=True)
        for domain in domains:
            cm = by_domain[domain]
            halluc = getattr(cm, "hallucination_pass_rate", None)
            table.add_row(
                domain,
                self._fmt_pct(getattr(cm, "accuracy", None)),
                self._fmt_float(getattr(cm, "f1_score", None), ".3f"),
                Text(self._fmt_pct(halluc), style=self._rate_style(halluc)),
                str(getattr(cm, "total", 0)),
            )

        self.console.print(table)

    def _display_worst_cases(self, report: "BenchmarkReport") -> None:
        best_strat = getattr(report, "best_strategy_f1", None)
        srs = getattr(report, "strategy_reports", None) or {}
        if not best_strat or best_strat not in srs:
            return

        sr = srs[best_strat]
        worst = getattr(sr, "worst_fp_cases", None)
        if not worst:
            return

        self.console.print(Rule("⚠️ Worst False Positives (Hallucinations Believed)", style="red"))

        panels = []
        for i, fp in enumerate(worst[:3], 1):
            case_id = fp.get("case_id", "n/a")
            score = fp.get("trust_score", None)
            text = fp.get("text", "") or ""
            text_short = text.strip().replace("\n", " ")
            if len(text_short) > 240:
                text_short = text_short[:240] + "…"

            header = f"[bold]#{i}[/bold]  [dim]case[/dim] {case_id}"
            if score is not None:
                header += f"  •  [dim]trust_score[/dim] [bold]{score:.3f}[/bold]"

            body = f"[dim]text[/dim]\n“{escape(text_short)}”"

            panels.append(
                Panel(
                    body,
                    title=header,
                    border_style="red",
                    box=ROUNDED,
                    padding=(1, 2),
                )
            )

        self.console.print(Columns(panels, equal=True, expand=True))

    def _display_significance(self, report: "BenchmarkReport") -> None:
        comparisons = getattr(report, "comparisons", None)
        mcnemar_results = getattr(report, "mcnemar_results", None)

        if not comparisons and not mcnemar_results:
            self.console.print("[dim]No significance tests available.[/dim]")
            return

        table = Table(title="📊 Statistical Significance", box=ROUNDED)
        table.add_column("Comparison", style="cyan")
        table.add_column("McNemar p", justify="right")
        table.add_column("DeLong p", justify="right")
        table.add_column("Significant", justify="center")

        if comparisons:
            for _name, comp in comparisons.items():
                sig = getattr(comp, "is_significant", False)
                sig_marker = "[green]✓[/green]" if sig else "[dim]✗[/dim]"
                table.add_row(
                    f"{comp.strategy_a} vs {comp.strategy_b}",
                    self._fmt_float(getattr(comp, "mcnemar_p_value", None), ".4f"),
                    self._fmt_float(getattr(comp, "delong_p_value", None), ".4f"),
                    sig_marker,
                )
        else:
            # legacy format
            for comp, res in (mcnemar_results or {}).items():
                sig = bool(res.get("significant_at_05"))
                sig_marker = "[green]✓[/green]" if sig else "[dim]✗[/dim]"
                table.add_row(
                    str(comp),
                    f"{res.get('p_value', 1.0):.4f}",
                    "N/A",
                    sig_marker,
                )

        self.console.print(table)

    def _display_runtime(self, report: "BenchmarkReport") -> None:
        srs = getattr(report, "strategy_reports", None) or {}
        if not srs:
            return

        total_tests = 0
        for sr in srs.values():
            cm = getattr(sr, "confusion_matrix", None)
            if cm and getattr(cm, "total", None) is not None:
                total_tests += int(cm.total)

        runtime = getattr(report, "total_runtime_seconds", None)
        if runtime and total_tests > 0:
            self.console.print(
                f"\n[dim]Total runtime: {float(runtime):.2f}s  "
                f"({float(runtime) / total_tests:.3f}s per test)[/dim]"
            )

    # -----------------------------
    # Formatting helpers
    # -----------------------------

    def _fmt_float(self, v: Any, fmt: str) -> str:
        if v is None:
            return "N/A"
        try:
            return format(float(v), fmt)
        except Exception:
            return "N/A"

    def _fmt_pct(self, v: Any) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.1%}"
        except Exception:
            return "N/A"

    def _fmt_ms(self, v: Any) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.0f}ms"
        except Exception:
            return "N/A"

    def _rate_style(self, rate: Any) -> str:
        # red if >=10%, yellow if >=5%, else dim/green-ish
        try:
            r = float(rate)
        except Exception:
            return "dim"
        if r >= 0.10:
            return "bold red"
        if r >= 0.05:
            return "yellow"
        return "green"

    def save(self, report: "BenchmarkReport", results: list["ResultMetric"], filename: str | None = None) -> Path:
        """Save console output as HTML."""
        self.display_summary(report, results)

        if filename is None:
            filename = f"ohi_benchmark_{getattr(report, 'run_id', 'run')}"

        filepath = self.output_dir / f"{filename}.html"
        self.console.save_html(str(filepath))
        return filepath
