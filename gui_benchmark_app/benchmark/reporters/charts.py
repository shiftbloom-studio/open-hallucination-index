"""
Performance Charts Reporter
============================

Generate publication-ready performance visualization charts for RAG / verification benchmarks.

Charts (PNG):
1) latency_boxplot.png              - Box plot comparing latency distributions (P50/P95 annotated)
2) throughput_bar.png               - Throughput bars (req/s) with latency annotations
3) latency_histogram.png            - Overlapping histograms with smoothed count lines (no SciPy required)

Additional charts:
4) latency_violin.png               - Violin plot (distribution shape + quartiles)
5) latency_ecdf.png                 - ECDF plot (empirical CDF) per strategy
6) latency_percentile_curves.png    - Percentile curves (P50..P99) per strategy
7) latency_quantiles_heatmap.png    - Heatmap of key latency quantiles (P50/P90/P95/P99)
8) latency_ranking.png              - Ranked dotplot of medians with IQR whiskers
9) throughput_latency_frontier.png  - Scatter of throughput vs P95 latency (trade-off view)
10) error_rate.png                  - (Optional) Error rate per strategy if errors exist in results

COMPARISON CHARTS (Multi-Evaluator):
11) comparison_radar.png            - Radar chart comparing evaluators across all metrics
12) comparison_grouped_bar.png      - Grouped bar chart for metric comparison
13) comparison_heatmap.png          - Heatmap of normalized scores per evaluator
14) comparison_latency_violin.png   - Violin plot of latencies by evaluator
15) comparison_factscore_dist.png   - FActScore distribution comparison
16) comparison_summary_table.png    - Summary table visualization
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.comparison_benchmark import ComparisonReport, EvaluatorMetrics
    from benchmark.models import BenchmarkReport, ResultMetric


# Modern color palette (accessible & print-friendly)
CHART_COLORS = {
    "primary": "#2563eb",      # Blue
    "secondary": "#7c3aed",    # Purple
    "success": "#059669",      # Green
    "warning": "#d97706",      # Orange
    "danger": "#dc2626",       # Red
    "accent": "#0891b2",       # Cyan
    "neutral": "#6b7280",      # Gray
}

STRATEGY_COLORS = [
    "#2563eb",  # Blue
    "#7c3aed",  # Purple
    "#059669",  # Green
    "#d97706",  # Orange
    "#dc2626",  # Red
    "#0891b2",  # Cyan
    "#84cc16",  # Lime
    "#f59e0b",  # Amber
]


class ChartsReporter(BaseReporter):
    """
    Generate performance visualization charts using matplotlib.

    Produces multiple publication-ready PNG charts into output_dir/charts.
    """

    def __init__(self, output_dir: Path, dpi: int = 160) -> None:
        super().__init__(output_dir)
        self.dpi = dpi
        self._charts_dir = output_dir / "charts"
        self._charts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def file_extension(self) -> str:
        return "png"

    def generate(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
    ) -> str:
        chart_files = self.generate_all_charts(report, results)
        return f"Generated {len(chart_files)} charts:\n" + "\n".join(
            f"  - {f.name}" for f in chart_files
        )

    def generate_all_charts(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
        base_filename: str | None = None,
    ) -> list[Path]:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            return []

        prefix = f"{base_filename}_" if base_filename else ""
        chart_files: list[Path] = []

        if not results:
            return chart_files

        # Split valid vs all (keep "results" untouched)
        valid_results = [r for r in results if not getattr(r, "has_error", False)]
        if not valid_results:
            # Still allow error chart if everything errored
            err_chart = self._create_error_rate_chart(results, report, prefix)
            if err_chart:
                chart_files.append(err_chart)
            return chart_files

        # Strategy set: prefer report ordering, but include any extra found in results
        report_strategies = list(getattr(report, "strategy_reports", {}).keys())
        result_strategies = sorted({getattr(r, "strategy", "unknown") for r in results})
        strategies = []
        seen = set()
        for s in report_strategies + result_strategies:
            if s not in seen:
                seen.add(s)
                strategies.append(s)

        results_by_strategy_valid = {
            s: [r for r in valid_results if getattr(r, "strategy", "unknown") == s]
            for s in strategies
        }
        results_by_strategy_all = {
            s: [r for r in results if getattr(r, "strategy", "unknown") == s]
            for s in strategies
        }

        # Remove strategies that are truly empty across all results
        strategies = [s for s in strategies if results_by_strategy_all.get(s)]
        results_by_strategy_valid = {s: results_by_strategy_valid.get(s, []) for s in strategies}
        results_by_strategy_all = {s: results_by_strategy_all.get(s, []) for s in strategies}

        if not strategies:
            return chart_files

        # Core 3 charts (compatible filenames)
        c1 = self._create_latency_boxplot(results_by_strategy_valid, strategies, report, prefix)
        if c1:
            chart_files.append(c1)

        c2 = self._create_throughput_chart(results_by_strategy_valid, strategies, report, prefix)
        if c2:
            chart_files.append(c2)

        c3 = self._create_latency_histogram(results_by_strategy_valid, strategies, report, prefix)
        if c3:
            chart_files.append(c3)

        # More / different charts
        c4 = self._create_latency_violinplot(results_by_strategy_valid, strategies, report, prefix)
        if c4:
            chart_files.append(c4)

        c5 = self._create_latency_ecdf(results_by_strategy_valid, strategies, report, prefix)
        if c5:
            chart_files.append(c5)

        c6 = self._create_latency_percentile_curves(results_by_strategy_valid, strategies, report, prefix)
        if c6:
            chart_files.append(c6)

        c7 = self._create_latency_quantile_heatmap(results_by_strategy_valid, strategies, report, prefix)
        if c7:
            chart_files.append(c7)

        c8 = self._create_latency_ranking_dotplot(results_by_strategy_valid, report, prefix)
        if c8:
            chart_files.append(c8)

        c9 = self._create_frontier_scatter(results_by_strategy_valid, report, prefix)
        if c9:
            chart_files.append(c9)

        # Optional: error rate chart (uses ALL results including has_error)
        c10 = self._create_error_rate_chart(results, report, prefix)
        if c10:
            chart_files.append(c10)

        return chart_files

    # -------------------------
    # Styling / helpers
    # -------------------------

    @contextmanager
    def _mpl_style(self):
        import matplotlib as mpl

        old = mpl.rcParams.copy()
        mpl.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "#fafafa",
                "axes.edgecolor": "#e5e7eb",
                "axes.labelcolor": "#374151",
                "text.color": "#111827",
                "xtick.color": "#374151",
                "ytick.color": "#374151",
                "grid.color": "#d1d5db",
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
                "axes.grid": True,
                "axes.axisbelow": True,
                "axes.titleweight": "bold",
                "font.size": 11,
            }
        )
        try:
            yield
        finally:
            mpl.rcParams.update(old)

    def _save_fig(self, fig, filename: str, report: Any = None) -> Path:
        import matplotlib.pyplot as plt

        if report:
            meta = self._get_run_metadata(report)
            if meta:
                fig.text(
                    0.5,
                    0.01,
                    meta,
                    ha="center",
                    fontsize=9,
                    color="#6b7280",  # neutral
                    weight="light",
                )

        filepath = self._charts_dir / filename
        fig.savefig(str(filepath), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return filepath

    def _get_run_metadata(self, report: Any) -> str:
        """Extract standardized run metadata for footer."""
        # Case 1: ComparisonReport (multi-evaluator)
        if hasattr(report, "evaluators") and isinstance(report.evaluators, dict):
            evaluators = report.evaluators
            count = len(evaluators)
            claims = 0
            if evaluators:
                # Use first evaluator to estimate dataset size
                first = next(iter(evaluators.values()))
                if hasattr(first, "hallucination") and hasattr(first.hallucination, "total"):
                    claims = first.hallucination.total
            
            run_id = getattr(report, "run_id", "n/a")
            # Timestamp if available
            ts = getattr(report, "timestamp", "")
            if ts and len(str(ts)) > 10:
                ts = str(ts)[:10]  # Just date
                return f"Run: {run_id} | Date: {ts} | Evaluators: {count} | Claims Tested: {claims}"
            return f"Run: {run_id} | Evaluators: {count} | Claims Tested: {claims}"

        # Case 2: BenchmarkReport (single strategy)
        dataset_size = getattr(report, "dataset_size", None)
        if dataset_size is not None:
            threshold = getattr(report, "threshold_used", None)
            run_id = getattr(report, "run_id", "n/a")
            parts = [f"Run: {run_id}", f"Cases: {dataset_size}"]
            if threshold is not None:
                parts.append(f"Threshold: {threshold}")
            return " | ".join(parts)
        
        return ""

    def _meta_line(self, report: BenchmarkReport) -> str:
        # Legacy / text-in-axis backup
        # We now prefer _save_fig footer, so this can return empty 
        # to avoid duplication, OR we keep it if it's used inside axes.
        # Given we are adding a footer, we interpret the user request as
        # "extend" not "replace and break layout". 
        # However, double info is ugly. 
        # We will keep this method but make it return "" so caller logic
        # simplifies to just chart specific info.
        return ""

    def _latencies_for(self, rs: list[ResultMetric]) -> np.ndarray:
        vals: list[float] = []
        for r in rs:
            v = getattr(r, "latency_ms", None)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv) and fv >= 0:
                vals.append(fv)
        return np.asarray(vals, dtype=float)

    def _quantiles(self, x: np.ndarray, qs: list[float]) -> dict[float, float]:
        if x.size == 0:
            return {q: float("nan") for q in qs}
        return {q: float(np.percentile(x, q)) for q in qs}

    def _smooth_counts(self, counts: np.ndarray, sigma_bins: float = 1.2) -> np.ndarray:
        # Lightweight Gaussian smoothing on histogram counts (no SciPy dependency)
        if counts.size < 5:
            return counts.astype(float)

        radius = int(max(2, round(3 * sigma_bins)))
        kx = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (kx / sigma_bins) ** 2)
        kernel /= kernel.sum()
        return np.convolve(counts.astype(float), kernel, mode="same")

    # -------------------------
    # Chart 1: Boxplot
    # -------------------------

    def _create_latency_boxplot(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        latencies = [self._latencies_for(results_by_strategy.get(s, [])) for s in strategies]
        if not any(arr.size for arr in latencies):
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.5, 7.2))

            bp = ax.boxplot(
                [arr.tolist() for arr in latencies],
                labels=[self._format_strategy_name(s) for s in strategies],
                patch_artist=True,
                showfliers=True,
                flierprops=dict(
                    marker="o",
                    markerfacecolor=CHART_COLORS["danger"],
                    markersize=3.5,
                    alpha=0.35,
                    markeredgecolor="none",
                ),
                medianprops=dict(color="white", linewidth=2.2),
                whiskerprops=dict(color="#374151", linewidth=1.3),
                capprops=dict(color="#374151", linewidth=1.3),
            )

            for i, patch in enumerate(bp["boxes"]):
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                patch.set_facecolor(color)
                patch.set_alpha(0.82)
                patch.set_edgecolor("#111827")
                patch.set_linewidth(1.2)

            # Annotate P50/P95 above each strategy (avoid huge offsets)
            y_max = 0.0
            for arr in latencies:
                if arr.size:
                    y_max = max(y_max, float(np.percentile(arr, 98)))
            y_pad = max(25.0, y_max * 0.06)

            for i, arr in enumerate(latencies):
                if arr.size == 0:
                    continue
                p50 = float(np.median(arr))
                p95 = float(np.percentile(arr, 95))
                ax.annotate(
                    f"P50 {p50:.0f}ms\nP95 {p95:.0f}ms",
                    xy=(i + 1, p95),
                    xytext=(i + 1, p95 + y_pad),
                    fontsize=9,
                    color="#374151",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#e5e7eb"),
                )

            ax.set_title("Response Latency Distribution by Strategy", fontsize=16, pad=18)
            ax.set_xlabel("Verification Strategy", fontsize=12)
            ax.set_ylabel("Latency (ms)", fontsize=12)

            plt.xticks(rotation=12, ha="right")

            # Footnote meta line (Replaced by unified footer in _save_fig)
            # ax.text removed to avoid duplication

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}latency_boxplot.png", report=report)

    # -------------------------
    # Chart 2: Throughput bars
    # -------------------------

    def _create_throughput_chart(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        throughputs: list[float] = []
        avg_latencies: list[float] = []
        p95_latencies: list[float] = []
        p50_latencies: list[float] = []

        for s in strategies:
            arr = self._latencies_for(results_by_strategy.get(s, []))
            if arr.size:
                p50 = float(np.median(arr))
                p95 = float(np.percentile(arr, 95))
                avg = float(np.mean(arr))
                tp = 1000.0 / p50 if p50 > 0 else 0.0
            else:
                p50, p95, avg, tp = 0.0, 0.0, 0.0, 0.0
            p50_latencies.append(p50)
            p95_latencies.append(p95)
            avg_latencies.append(avg)
            throughputs.append(tp)

        if not any(throughputs):
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.5, 7.2))
            x = np.arange(len(strategies))

            bars = ax.bar(
                x,
                throughputs,
                width=0.62,
                color=[STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i in range(len(strategies))],
                edgecolor="#111827",
                linewidth=1.1,
                alpha=0.9,
            )

            # Labels / annotations
            y_max = max(throughputs) if throughputs else 1.0
            for i, (bar, tp, p50, p95, avg) in enumerate(zip(bars, throughputs, p50_latencies, p95_latencies, avg_latencies)):
                h = bar.get_height()
                ax.annotate(
                    f"{tp:.2f} req/s",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    fontweight="bold",
                )
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h * 0.52,
                        f"P50 {p50:.0f}ms\nP95 {p95:.0f}ms\nAvg {avg:.0f}ms",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        fontweight="medium",
                    )

            ax.set_title("Throughput Comparison by Strategy", fontsize=16, pad=18)
            ax.set_xlabel("Verification Strategy", fontsize=12)
            ax.set_ylabel("Throughput (requests/second)", fontsize=12)

            ax.set_xticks(x)
            ax.set_xticklabels([self._format_strategy_name(s) for s in strategies], rotation=12, ha="right")

            ax.set_ylim(0, y_max * 1.25)

            # Overall throughput info if report provides runtime
            total_time = getattr(report, "total_runtime_seconds", None)
            total_requests = sum(len(results_by_strategy.get(s, [])) for s in strategies)
            if total_time and total_time > 0:
                overall = total_requests / float(total_time)
                box = f"Total: {total_requests} requests\nRuntime: {float(total_time):.2f}s\nOverall: {overall:.3f} req/s"
            else:
                box = f"Total: {total_requests} requests"
            ax.text(
                0.02,
                0.98,
                box,
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#e5e7eb"),
            )

            ax.text(
                0.5,
                -0.12,
                "Throughput computed as 1000 / P50 latency (sustainable estimate)",
                transform=ax.transAxes,
                fontsize=10,
                color=CHART_COLORS["neutral"],
                ha="center",
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}throughput_bar.png", report=report)

    # -------------------------
    # Chart 3: Histogram overlay (no SciPy)
    # -------------------------

    def _create_latency_histogram(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        all_lat = np.concatenate([self._latencies_for(results_by_strategy.get(s, [])) for s in strategies if results_by_strategy.get(s)], axis=0)
        if all_lat.size == 0:
            return None

        # Cap at P99 to avoid extreme outliers dominating
        cap = float(np.percentile(all_lat, 99))
        cap = max(cap, 1.0)

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(13.8, 7.0))

            bins = np.linspace(0.0, cap, 44)
            centers = 0.5 * (bins[:-1] + bins[1:])

            for i, s in enumerate(strategies):
                arr = self._latencies_for(results_by_strategy.get(s, []))
                if arr.size == 0:
                    continue

                arr = arr[arr <= cap]
                if arr.size == 0:
                    continue

                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                counts, _ = np.histogram(arr, bins=bins)

                ax.hist(
                    arr,
                    bins=bins,
                    alpha=0.35,
                    color=color,
                    edgecolor=color,
                    linewidth=1.2,
                    label=f"{self._format_strategy_name(s)} (n={arr.size})",
                )

                smooth = self._smooth_counts(counts, sigma_bins=1.2)
                ax.plot(centers, smooth, color=color, linewidth=2.2, alpha=0.95)

            # Overall P50/P95 markers (within capped range)
            capped_all = all_lat[all_lat <= cap]
            if capped_all.size:
                p50 = float(np.median(capped_all))
                p95 = float(np.percentile(capped_all, 95))
                ax.axvline(p50, color=CHART_COLORS["success"], linestyle="--", linewidth=2, label=f"Overall P50 {p50:.0f}ms")
                ax.axvline(p95, color=CHART_COLORS["danger"], linestyle="--", linewidth=2, label=f"Overall P95 {p95:.0f}ms")

            ax.set_title("Response Time Distribution (Histogram + Smoothed Counts)", fontsize=16, pad=18)
            ax.set_xlabel("Latency (ms)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_xlim(0, cap * 1.02)

            ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="#e5e7eb")

            # Performance bands
            if cap > 500:
                ax.axvspan(0, 500, alpha=0.06, color=CHART_COLORS["success"])
            if cap > 2000:
                ax.axvspan(500, 2000, alpha=0.06, color=CHART_COLORS["warning"])
                ax.axvspan(2000, cap, alpha=0.06, color=CHART_COLORS["danger"])
            elif cap > 500:
                ax.axvspan(500, cap, alpha=0.06, color=CHART_COLORS["warning"])

            ax.text(
                0.5,
                -0.10,
                f"Distribution capped at P99 ({cap:.0f}ms) | Total samples: {all_lat.size}",
                transform=ax.transAxes,
                fontsize=10,
                color=CHART_COLORS["neutral"],
                ha="center",
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}latency_histogram.png", report=report)

    # -------------------------
    # Chart 4: Violin plot
    # -------------------------

    def _create_latency_violinplot(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        data = [self._latencies_for(results_by_strategy.get(s, [])) for s in strategies]
        if not any(arr.size for arr in data):
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.5, 7.2))

            vp = ax.violinplot(
                [arr.tolist() for arr in data],
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            for i, body in enumerate(vp["bodies"]):
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                body.set_facecolor(color)
                body.set_edgecolor("#111827")
                body.set_alpha(0.75)
                body.set_linewidth(1.0)

            # Median line style
            if "cmedians" in vp:
                vp["cmedians"].set_color("white")
                vp["cmedians"].set_linewidth(2.2)

            # Add quartile whiskers (P25/P75)
            for i, arr in enumerate(data):
                if arr.size == 0:
                    continue
                q25 = float(np.percentile(arr, 25))
                q75 = float(np.percentile(arr, 75))
                ax.vlines(i + 1, q25, q75, color="#111827", linewidth=2.2, alpha=0.9)
                ax.scatter([i + 1], [float(np.median(arr))], s=28, color="white", edgecolors="#111827", linewidths=0.8, zorder=3)

            ax.set_title("Latency Distribution (Violin + IQR)", fontsize=16, pad=18)
            ax.set_xlabel("Verification Strategy", fontsize=12)
            ax.set_ylabel("Latency (ms)", fontsize=12)

            ax.set_xticks(np.arange(1, len(strategies) + 1))
            ax.set_xticklabels([self._format_strategy_name(s) for s in strategies], rotation=12, ha="right")

            ax.text(
                0.5,
                -0.12,
                "IQR whiskers show P25–P75; white dot = median",
                transform=ax.transAxes,
                fontsize=10,
                color=CHART_COLORS["neutral"],
                ha="center",
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}latency_violin.png", report=report)

    # -------------------------
    # Chart 5: ECDF
    # -------------------------

    def _create_latency_ecdf(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        # Use global cap to keep plot readable
        all_lat = np.concatenate([self._latencies_for(results_by_strategy.get(s, [])) for s in strategies if results_by_strategy.get(s)], axis=0)
        if all_lat.size == 0:
            return None
        cap = float(np.percentile(all_lat, 99))
        cap = max(cap, 1.0)

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.8, 7.2))

            for i, s in enumerate(strategies):
                arr = self._latencies_for(results_by_strategy.get(s, []))
                if arr.size == 0:
                    continue
                arr = np.sort(arr[arr <= cap])
                if arr.size == 0:
                    continue
                y = np.arange(1, arr.size + 1) / arr.size
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                ax.plot(arr, y, color=color, linewidth=2.4, alpha=0.95, label=f"{self._format_strategy_name(s)} (n={arr.size})")

            ax.set_title("Latency ECDF by Strategy (capped at P99)", fontsize=16, pad=18)
            ax.set_xlabel("Latency (ms)", fontsize=12)
            ax.set_ylabel("Cumulative Probability", fontsize=12)
            ax.set_xlim(0, cap * 1.02)
            ax.set_ylim(0, 1.02)

            ax.legend(loc="lower right", fontsize=10, framealpha=0.95, edgecolor="#e5e7eb")

            # Reference lines for common SLO points
            for p, label in [(0.5, "P50"), (0.9, "P90"), (0.95, "P95")]:
                ax.axhline(p, color="#9ca3af", linewidth=1.2, alpha=0.8)
                ax.text(cap * 1.01, p, label, va="center", ha="left", fontsize=9, color="#6b7280")

            # Footnote added by _save_fig
            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}latency_ecdf.png", report=report)

    # -------------------------
    # Chart 6: Percentile curves
    # -------------------------

    def _create_latency_percentile_curves(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        percentiles = [50, 75, 90, 95, 97, 99]

        any_data = False
        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.8, 7.2))

            for i, s in enumerate(strategies):
                arr = self._latencies_for(results_by_strategy.get(s, []))
                if arr.size == 0:
                    continue
                qs = [float(np.percentile(arr, p)) for p in percentiles]
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                ax.plot(percentiles, qs, marker="o", linewidth=2.4, color=color, alpha=0.95, label=f"{self._format_strategy_name(s)} (n={arr.size})")
                any_data = True

            if not any_data:
                return None

            ax.set_title("Latency Percentile Curves by Strategy", fontsize=16, pad=18)
            ax.set_xlabel("Percentile", fontsize=12)
            ax.set_ylabel("Latency (ms)", fontsize=12)
            ax.set_xticks(percentiles)

            ax.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="#e5e7eb")

            ax.text(
                0.5,
                -0.12,
                "Shows tail behavior (P90–P99) clearly",
                transform=ax.transAxes,
                fontsize=10,
                color=CHART_COLORS["neutral"],
                ha="center",
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}latency_percentile_curves.png", report=report)

    # -------------------------
    # Chart 7: Quantile heatmap
    # -------------------------

    def _create_latency_quantile_heatmap(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        qs = [50, 90, 95, 99]
        rows = []
        labels = []
        for s in strategies:
            arr = self._latencies_for(results_by_strategy.get(s, []))
            if arr.size == 0:
                continue
            qvals = [float(np.percentile(arr, q)) for q in qs]
            rows.append(qvals)
            labels.append(self._format_strategy_name(s))

        if not rows:
            return None

        mat = np.asarray(rows, dtype=float)

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.6, 0.55 * len(labels) + 2.8))
            im = ax.imshow(mat, aspect="auto")

            ax.set_title("Latency Quantiles Heatmap (ms)", fontsize=16, pad=16)
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xticks(np.arange(len(qs)))
            ax.set_xticklabels([f"P{q}" for q in qs])

            # Annotate cells
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=10, color="#111827")

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Latency (ms)", fontsize=11)

            ax.grid(False)

            # Footnote added by _save_fig
            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}latency_quantiles_heatmap.png", report=report)

    # -------------------------
    # Chart 8: Ranked dotplot (median + IQR)
    # -------------------------

    def _create_latency_ranking_dotplot(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        rows = []
        for s, rs in results_by_strategy.items():
            arr = self._latencies_for(rs)
            if arr.size == 0:
                continue
            rows.append(
                (
                    s,
                    float(np.median(arr)),
                    float(np.percentile(arr, 25)),
                    float(np.percentile(arr, 75)),
                    arr.size,
                )
            )

        if not rows:
            return None

        # Sort by median (fastest first)
        rows.sort(key=lambda x: x[1])

        names = [self._format_strategy_name(r[0]) for r in rows]
        med = np.array([r[1] for r in rows], dtype=float)
        q25 = np.array([r[2] for r in rows], dtype=float)
        q75 = np.array([r[3] for r in rows], dtype=float)
        n = [r[4] for r in rows]

        y = np.arange(len(names))

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.8, 0.55 * len(names) + 3.0))

            # IQR whiskers
            ax.hlines(y, q25, q75, color="#111827", linewidth=2.2, alpha=0.9)
            # Median dots
            ax.scatter(med, y, s=70, color=CHART_COLORS["primary"], edgecolors="#111827", linewidths=0.8, zorder=3)

            # Labels
            for i, (m, nn) in enumerate(zip(med, n)):
                ax.text(m, i, f"  {m:.0f}ms (n={nn})", va="center", ha="left", fontsize=10, color="#111827")

            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()

            ax.set_title("Latency Ranking (Median with IQR)", fontsize=16, pad=16)
            ax.set_xlabel("Latency (ms)", fontsize=12)

            ax.text(
                0.5,
                -0.10,
                "Whiskers: P25–P75 | Dot: P50",
                transform=ax.transAxes,
                fontsize=10,
                color=CHART_COLORS["neutral"],
                ha="center",
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}latency_ranking.png", report=report)

    # -------------------------
    # Chart 9: Throughput vs tail latency frontier
    # -------------------------

    def _create_frontier_scatter(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        points = []
        for s, rs in results_by_strategy.items():
            arr = self._latencies_for(rs)
            if arr.size == 0:
                continue
            p50 = float(np.median(arr))
            p95 = float(np.percentile(arr, 95))
            tp = 1000.0 / p50 if p50 > 0 else 0.0
            points.append((s, tp, p95, arr.size))

        if not points:
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.8, 7.2))

            for i, (s, tp, p95, n) in enumerate(points):
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                ax.scatter(tp, p95, s=120, color=color, alpha=0.9, edgecolors="#111827", linewidths=0.9)
                ax.annotate(
                    f"{self._format_strategy_name(s)}\n(n={n})",
                    xy=(tp, p95),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=10,
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#e5e7eb"),
                )

            ax.set_title("Throughput vs Tail Latency (P95) Frontier", fontsize=16, pad=18)
            ax.set_xlabel("Throughput (req/s, approx via 1000/P50)", fontsize=12)
            ax.set_ylabel("P95 Latency (ms)", fontsize=12)

            ax.text(
                0.5,
                -0.12,
                "Top-left is ideal (high throughput, low tail latency)",
                transform=ax.transAxes,
                fontsize=10,
                color=CHART_COLORS["neutral"],
                ha="center",
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}throughput_latency_frontier.png", report=report)

    # -------------------------
    # Chart 10: Error rate (optional)
    # -------------------------

    def _create_error_rate_chart(
        self,
        results: list[ResultMetric],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if not results:
            return None

        # Aggregate error rates by strategy (use ALL results)
        by_strategy: dict[str, list[ResultMetric]] = {}
        for r in results:
            s = getattr(r, "strategy", "unknown")
            by_strategy.setdefault(s, []).append(r)

        strategies = sorted(by_strategy.keys())
        totals = np.array([len(by_strategy[s]) for s in strategies], dtype=float)
        errs = np.array([sum(1 for r in by_strategy[s] if getattr(r, "has_error", False)) for s in strategies], dtype=float)

        if errs.sum() <= 0:
            return None

        rates = np.where(totals > 0, errs / totals, 0.0)

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.5, 6.6))
            x = np.arange(len(strategies))

            bars = ax.bar(
                x,
                rates * 100.0,
                width=0.62,
                color=CHART_COLORS["danger"],
                edgecolor="#111827",
                linewidth=1.1,
                alpha=0.88,
            )

            for i, (bar, r, e, t) in enumerate(zip(bars, rates, errs, totals)):
                h = bar.get_height()
                ax.annotate(
                    f"{r*100:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    fontweight="bold",
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(0.5, h * 0.55),
                    f"{int(e)}/{int(t)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="medium",
                )

            ax.set_title("Error Rate by Strategy", fontsize=16, pad=18)
            ax.set_xlabel("Verification Strategy", fontsize=12)
            ax.set_ylabel("Error Rate (%)", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([self._format_strategy_name(s) for s in strategies], rotation=12, ha="right")
            ax.set_ylim(0, max(rates * 100.0) * 1.25)

            # Footnote added by _save_fig
            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}error_rate.png", report=report)

    # -------------------------
    # Name formatting
    # -------------------------

    def _format_strategy_name(self, strategy: str) -> str:
        name_map = {
            "vector_semantic": "Vector\n(Semantic)",
            "graph_exact": "Graph\n(Exact)",
            "hybrid": "Hybrid",
            "cascading": "Cascading",
            "mcp_enhanced": "MCP\nEnhanced",
            "adaptive": "Adaptive",
            # Evaluator names
            "ohi": "OHI",
            "gpt4": "GPT-4",
            "gpt-4": "GPT-4",
            "vector_rag": "VectorRAG",
            "vectorrag": "VectorRAG",
        }
        return name_map.get(strategy.lower(), strategy.replace("_", " ").title())

    def save(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
        filename: str | None = None,
    ) -> Path:
        base_filename = filename or f"ohi_benchmark_{getattr(report, 'run_id', 'run')}"
        self.generate_all_charts(report, results, base_filename)
        return self._charts_dir

    # =========================================================================
    # COMPARISON CHARTS (Multi-Evaluator)
    # =========================================================================


    def generate_comparison_charts(
        self,
        comparison_report: ComparisonReport,
        prefix: str = "",
        consolidated: bool = True,
    ) -> list[Path]:
        """
        Generate comparison charts from a ComparisonReport.

        - consolidated=True: generates a single dashboard PNG (plus optional extra panels if data exists)
        - consolidated=False: generates individual charts (radar/bar/latency/heatmap/...)
        """
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            return []

        if not getattr(comparison_report, "evaluators", None):
            return []

        chart_files: list[Path] = []

        if consolidated:
            dashboard = self._create_comparison_dashboard(comparison_report, prefix)
            if dashboard:
                chart_files.append(dashboard)
            return chart_files

        # DETAILED MODE
        c1 = self._create_comparison_radar(comparison_report, prefix)
        if c1:
            chart_files.append(c1)

        c2 = self._create_comparison_grouped_bar(comparison_report, prefix)
        if c2:
            chart_files.append(c2)

        c3 = self._create_comparison_latency_boxplot(comparison_report, prefix)
        if c3:
            chart_files.append(c3)

        c4 = self._create_comparison_heatmap(comparison_report, prefix)
        if c4:
            chart_files.append(c4)

        # Optional panels (only if streams exist)
        c5 = self._create_comparison_risk_coverage(comparison_report, prefix)
        if c5:
            chart_files.append(c5)

        c6 = self._create_comparison_rag_retrieval_citation(comparison_report, prefix)
        if c6:
            chart_files.append(c6)

        c7 = self._create_comparison_factscore_violin(comparison_report, prefix)
        if c7:
            chart_files.append(c7)

        return chart_files


    def _create_comparison_dashboard(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """Create a consolidated dashboard (3x2 grid) with key + extended metrics."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            return None

        evaluators = getattr(comparison_report, "evaluators", None) or {}
        if not evaluators:
            return None

        with self._mpl_style():
            fig = plt.figure(figsize=(20, 21))
            gs = GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.28)

            # Row 0
            ax_radar = fig.add_subplot(gs[0, 0], polar=True)
            self._draw_radar_on_axis(ax_radar, evaluators)

            ax_heatmap = fig.add_subplot(gs[0, 1])
            self._draw_heatmap_on_axis(ax_heatmap, evaluators, fig)

            # Row 1
            ax_bar = fig.add_subplot(gs[1, 0])
            self._draw_grouped_bar_on_axis(ax_bar, evaluators)

            ax_latency = fig.add_subplot(gs[1, 1])
            self._draw_latency_boxplot_on_axis(ax_latency, evaluators)

            # Row 2
            ax_rc = fig.add_subplot(gs[2, 0])
            self._draw_risk_coverage_on_axis(ax_rc, evaluators)

            ax_rag = fig.add_subplot(gs[2, 1])
            self._draw_rag_retrieval_citation_on_axis(ax_rag, evaluators)

            fig.suptitle("OHI Benchmark Comparison Dashboard", fontsize=20, fontweight='bold', y=0.985)

            run_id = getattr(comparison_report, "run_id", "")
            # Footer added by _save_fig
            # Manual header removed to avoid duplication, or can be kept if distinct info needed.
            # But standardizing on footer is better for "all images".

            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
            return self._save_fig(fig, f"{prefix}comparison_dashboard.png", report=comparison_report)


    def _draw_radar_on_axis(self, ax, evaluators: dict) -> None:
        """Draw radar chart. Uses base metrics + optional AURC/BEIR/ALCE/RAG signals if present."""
        base_labels = [
            ("Accuracy", lambda m: float(m.hallucination.accuracy)),
            ("Precision", lambda m: float(m.hallucination.precision)),
            ("Recall", lambda m: float(m.hallucination.recall)),
            ("F1", lambda m: float(m.hallucination.f1_score)),
            ("Safety", lambda m: float(1.0 - m.hallucination.hallucination_pass_rate)),
            ("TruthfulQA", lambda m: float(m.truthfulqa.accuracy)),
            ("FActScore", lambda m: float(m.factscore.avg_factscore)),
            ("Speed", lambda m: float(min(1.0, 1000.0 / m.latency.p95) if getattr(m.latency, 'p95', 0) > 0 else 0.0)),
        ]

        # Optional extras (only show if any evaluator has non-trivial values)
        extra_specs = [
            ("Selective", lambda m: float(1.0 / (1.0 + float(getattr(m.hallucination, "aurc", 0.0)))) if getattr(m.hallucination, "aurc", None) is not None else 0.0),
            ("nDCG@10", lambda m: float((m.hallucination.retrieval_metrics(ks=(10,)) or {}).get("ndcg@10", 0.0))),
            ("CiteRate", lambda m: float((m.hallucination.alce_metrics() or {}).get("citation_rate", 0.0))),
            ("Faithful", lambda m: float((m.hallucination.ragas_proxy_metrics() or {}).get("faithfulness", 0.0))),
        ]

        labels = list(base_labels)
        for lab, fn in extra_specs:
            vals = [fn(m) for m in evaluators.values()]
            if any(v > 0 for v in vals):
                labels.append((lab, fn))

        metrics_labels = [l for l, _ in labels]
        num_metrics = len(metrics_labels)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]

        for i, (name, m) in enumerate(evaluators.items()):
            values = [float(fn(m)) for _, fn in labels]
            # Clip to [0,1] for radar safety
            values = [min(1.0, max(0.0, v)) for v in values]
            values += values[:1]
            color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            ax.plot(angles, values, 'o-', linewidth=2.2, label=self._format_strategy_name(name), color=color)
            ax.fill(angles, values, alpha=0.18, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, size=9)
        ax.set_ylim(0, 1.0)
        ax.set_title("Multi-Metric Overview", fontsize=12, pad=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)


    def _draw_grouped_bar_on_axis(self, ax, evaluators: dict) -> None:
        """Draw grouped bar chart for key metrics + optional new families."""
        evaluator_names = list(evaluators.keys())
        if not evaluator_names:
            ax.text(0.5, 0.5, "No evaluators", ha='center', va='center', transform=ax.transAxes)
            return

        # Base (always)
        specs = [
            ("Accuracy", lambda m: float(m.hallucination.accuracy)),
            ("F1", lambda m: float(m.hallucination.f1_score)),
            ("Safety", lambda m: float(1.0 - m.hallucination.hallucination_pass_rate)),
            ("TruthfulQA", lambda m: float(m.truthfulqa.accuracy)),
            ("FActScore", lambda m: float(m.factscore.avg_factscore)),
        ]

        # Optional extra bars
        opt = [
            ("Selective", lambda m: float(1.0 / (1.0 + getattr(m.hallucination, 'aurc', 0.0))) if getattr(m.hallucination, 'aurc', None) is not None else 0.0),
            ("nDCG@10", lambda m: float((m.hallucination.retrieval_metrics(ks=(10,)) or {}).get('ndcg@10', 0.0))),
            ("CiteRate", lambda m: float((m.hallucination.alce_metrics() or {}).get('citation_rate', 0.0))),
            ("Faithful", lambda m: float((m.hallucination.ragas_proxy_metrics() or {}).get('faithfulness', 0.0))),
        ]
        for label, fn in opt:
            vals = [fn(evaluators[n]) for n in evaluator_names]
            if any(v > 0 for v in vals):
                specs.append((label, fn))

        metric_labels = [s[0] for s in specs]
        x = np.arange(len(metric_labels))
        width = 0.82 / max(len(evaluator_names), 1)

        for i, name in enumerate(evaluator_names):
            m = evaluators[name]
            values = [min(1.0, max(0.0, float(fn(m)))) for _, fn in specs]
            color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            offset = (i - len(evaluator_names) / 2 + 0.5) * width
            ax.bar(x + offset, values, width * 0.92, label=self._format_strategy_name(name), color=color, edgecolor="#111827", linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (higher is better)", fontsize=10)
        ax.set_title("Key + Extended Metrics", fontsize=12, pad=10)
        ax.legend(fontsize=9, loc='upper right')

    def _draw_latency_boxplot_on_axis(self, ax, evaluators: dict) -> None:
        """Draw latency boxplot on provided axis."""
        latencies = []
        labels = []
        for name, m in evaluators.items():
            if m.latency.latencies_ms:
                latencies.append(m.latency.latencies_ms)
                labels.append(self._format_strategy_name(name))

        if not latencies:
            ax.text(0.5, 0.5, "No latency data", ha='center', va='center', transform=ax.transAxes)
            return

        bp = ax.boxplot(latencies, labels=labels, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markerfacecolor=CHART_COLORS["danger"], markersize=3, alpha=0.3))

        for i, patch in enumerate(bp["boxes"]):
            color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.set_title("Response Latency Distribution", fontsize=12, pad=10)

        # Annotate P50/P95
        for i, lat in enumerate(latencies):
            p50, p95 = np.median(lat), np.percentile(lat, 95)
            ax.annotate(f"P50:{p50:.0f}ms P95:{p95:.0f}ms", xy=(i + 1, p95),
                        fontsize=8, ha='center', va='bottom')
    

    def _draw_heatmap_on_axis(self, ax, evaluators: dict, fig) -> None:
        """Draw heatmap (normalized scores). Includes optional AURC/BEIR/ALCE/RAG signals if present."""
        evaluator_names = list(evaluators.keys())
        if not evaluator_names:
            ax.text(0.5, 0.5, "No evaluators", ha='center', va='center', transform=ax.transAxes)
            return

        # Build dynamic metric columns (all higher-is-better)
        specs = [
            ("Acc", lambda m: float(m.hallucination.accuracy)),
            ("Prec", lambda m: float(m.hallucination.precision)),
            ("Rec", lambda m: float(m.hallucination.recall)),
            ("F1", lambda m: float(m.hallucination.f1_score)),
            ("Safe", lambda m: float(1.0 - m.hallucination.hallucination_pass_rate)),
            ("TQA", lambda m: float(m.truthfulqa.accuracy)),
            ("Fact", lambda m: float(m.factscore.avg_factscore)),
            ("Spd", lambda m: float(min(1.0, 1000.0 / m.latency.p95) if getattr(m.latency, 'p95', 0) > 0 else 0.0)),
        ]

        optional = [
            ("Sel", lambda m: float(1.0 / (1.0 + getattr(m.hallucination, 'aurc', 0.0))) if getattr(m.hallucination, 'aurc', None) is not None else 0.0),
            ("nDCG10", lambda m: float((m.hallucination.retrieval_metrics(ks=(10,)) or {}).get('ndcg@10', 0.0))),
            ("Cite", lambda m: float((m.hallucination.alce_metrics() or {}).get('citation_rate', 0.0))),
            ("Faith", lambda m: float((m.hallucination.ragas_proxy_metrics() or {}).get('faithfulness', 0.0))),
        ]
        for lab, fn in optional:
            vals = [fn(evaluators[n]) for n in evaluator_names]
            if any(v > 0 for v in vals):
                specs.append((lab, fn))

        metrics = [lab for lab, _ in specs]
        data = []
        for name in evaluator_names:
            m = evaluators[name]
            row = [min(1.0, max(0.0, float(fn(m)))) for _, fn in specs]
            data.append(row)

        mat = np.array(data, dtype=float)
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=9, rotation=0)
        ax.set_yticks(np.arange(len(evaluator_names)))
        ax.set_yticklabels([self._format_strategy_name(n) for n in evaluator_names], fontsize=9)

        for i in range(len(evaluator_names)):
            for j in range(len(metrics)):
                val = mat[i, j]
                color = "white" if val < 0.45 else "#111827"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8, fontweight="bold", color=color)

        ax.set_title("Performance Heatmap", fontsize=12, pad=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    def _draw_risk_coverage_on_axis(self, ax, evaluators: dict) -> None:
        """Plot risk-coverage curves (selective prediction). Uses confidence_scores + correct_flags streams."""
        any_stream = False
        for i, (name, m) in enumerate(evaluators.items()):
            conf = getattr(m.hallucination, 'confidence_scores', None)
            corr = getattr(m.hallucination, 'correct_flags', None)
            if not conf or not corr or len(conf) != len(corr) or len(conf) < 5:
                continue
            any_stream = True
            conf_arr = np.asarray(conf, dtype=float)
            corr_arr = np.asarray(corr, dtype=bool)
            order = np.argsort(-conf_arr, kind='mergesort')
            corr_s = corr_arr[order]
            n = corr_s.size
            coverage = (np.arange(1, n + 1) / n)
            risk = np.cumsum(~corr_s) / np.arange(1, n + 1)
            color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            ax.plot(coverage, risk, color=color, linewidth=2.2, alpha=0.95, label=f"{self._format_strategy_name(name)}")

            aurc = getattr(m.hallucination, 'aurc', None)
            if aurc is None:
                # Approx AURC
                aurc = float(np.trapz(risk, coverage))
            ax.text(
                0.02,
                0.92 - i * 0.06,
                f"{self._format_strategy_name(name)}: AURC {float(aurc):.4f}",
                transform=ax.transAxes,
                fontsize=9,
                color=color,
            )

        if not any_stream:
            ax.text(0.5, 0.5, "No confidence/correctness streams for Risk-Coverage", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Risk-Coverage (AURC)", fontsize=12, pad=10)
            ax.set_xlabel("Coverage")
            ax.set_ylabel("Risk (error rate)")
            return

        ax.set_title("Risk-Coverage Curves (Selective Prediction)", fontsize=12, pad=10)
        ax.set_xlabel("Coverage (fraction kept)", fontsize=10)
        ax.set_ylabel("Risk (error rate on kept)", fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(fontsize=9, loc='upper right')

    def _draw_rag_retrieval_citation_on_axis(self, ax, evaluators: dict) -> None:
        """One compact panel for RAG-ish metrics: retrieval, citations, faithfulness."""
        names = list(evaluators.keys())
        if not names:
            ax.text(0.5, 0.5, "No evaluators", ha='center', va='center', transform=ax.transAxes)
            return

        # Choose a small set of high-signal metrics (0-1, higher is better)
        metric_specs = [
            ("nDCG@10", lambda m: float((m.hallucination.retrieval_metrics(ks=(10,)) or {}).get('ndcg@10', 0.0))),
            ("Recall@10", lambda m: float((m.hallucination.retrieval_metrics(ks=(10,)) or {}).get('recall@10', 0.0))),
            ("CiteRate", lambda m: float((m.hallucination.alce_metrics() or {}).get('citation_rate', 0.0))),
            ("Faithful", lambda m: float((m.hallucination.ragas_proxy_metrics() or {}).get('faithfulness', 0.0))),
        ]

        # Hide the whole chart if everything is empty
        all_vals = []
        for _, fn in metric_specs:
            all_vals.extend([fn(evaluators[n]) for n in names])
        if not any(v > 0 for v in all_vals):
            ax.text(0.5, 0.5, "No retrieval/citation/RAG signals", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("RAG Metrics", fontsize=12, pad=10)
            return

        x = np.arange(len(metric_specs))
        width = 0.82 / max(len(names), 1)

        for i, name in enumerate(names):
            m = evaluators[name]
            values = [min(1.0, max(0.0, fn(m))) for _, fn in metric_specs]
            color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            offset = (i - len(names) / 2 + 0.5) * width
            ax.bar(x + offset, values, width * 0.92, color=color, alpha=0.9, edgecolor="#111827", linewidth=0.5, label=self._format_strategy_name(name))

        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metric_specs], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title("Retrieval / Citations / Faithfulness", fontsize=12, pad=10)
        ax.legend(fontsize=9, loc='upper right')


    def _create_comparison_risk_coverage(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """Create Risk-Coverage curve chart (AURC) if confidence streams exist."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        evaluators = getattr(comparison_report, "evaluators", None) or {}
        if not evaluators:
            return None

        # quick check
        if not any(getattr(m.hallucination, 'confidence_scores', None) for m in evaluators.values()):
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.8, 7.2))
            self._draw_risk_coverage_on_axis(ax, evaluators)
            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}comparison_risk_coverage.png", report=comparison_report)

    def _create_comparison_rag_retrieval_citation(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """Create grouped bar chart for retrieval/citation/rag metrics if present."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        evaluators = getattr(comparison_report, "evaluators", None) or {}
        if not evaluators:
            return None

        # quick check
        any_vals = False
        for m in evaluators.values():
            if (m.hallucination.retrieval_metrics(ks=(10,)) or {}).get('ndcg@10', 0.0) > 0:
                any_vals = True
            if (m.hallucination.alce_metrics() or {}).get('citation_rate', 0.0) > 0:
                any_vals = True
            if (m.hallucination.ragas_proxy_metrics() or {}).get('faithfulness', 0.0) > 0:
                any_vals = True
        if not any_vals:
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12.8, 7.2))
            self._draw_rag_retrieval_citation_on_axis(ax, evaluators)
            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}comparison_rag_retrieval_citation.png", report=comparison_report)


    def _create_comparison_radar(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """
        Create radar chart comparing all evaluators across metrics.
        
        This is the PRIMARY comparison visualization showing all metrics
        at once for each evaluator.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            return None

        evaluators = comparison_report.evaluators
        if not evaluators:
            return None

        # Collect summary scores for each evaluator
        metrics_labels = [
            "Accuracy", "Precision", "Recall", "F1 Score",
            "Safety", "TruthfulQA", "FActScore", "Speed"
        ]
        
        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

            num_metrics = len(metrics_labels)
            angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Complete the loop

            # Plot each evaluator
            for i, (name, metrics) in enumerate(evaluators.items()):
                scores = metrics.get_summary_scores()
                values = [
                    scores.get("Accuracy", 0),
                    scores.get("Precision", 0),
                    scores.get("Recall", 0),
                    scores.get("F1 Score", 0),
                    scores.get("Safety (1-HPR)", 0),
                    scores.get("TruthfulQA", 0),
                    scores.get("FActScore", 0),
                    scores.get("Speed (1/P95)", 0),
                ]
                values += values[:1]  # Complete the loop

                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                ax.plot(angles, values, 'o-', linewidth=2.5, label=name, color=color)
                ax.fill(angles, values, alpha=0.25, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_labels, size=11)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9)
            ax.grid(True, linestyle='--', alpha=0.7)

            ax.set_title(
                "Multi-Evaluator Performance Comparison",
                fontsize=16, fontweight='bold', pad=20
            )

            # Legend
            ax.legend(
                loc='upper right', bbox_to_anchor=(1.3, 1.1),
                fontsize=11, framealpha=0.95
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}comparison_radar.png", report=comparison_report)

    def _create_comparison_grouped_bar(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """Create grouped bar chart comparing key metrics across evaluators."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        evaluators = comparison_report.evaluators
        if not evaluators:
            return None

        metrics = ["Accuracy", "F1 Score", "Safety", "TruthfulQA", "FActScore"]
        evaluator_names = list(evaluators.keys())
        
        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(14, 8))

            x = np.arange(len(metrics))
            width = 0.25
            offset = -(len(evaluator_names) - 1) * width / 2

            for i, name in enumerate(evaluator_names):
                m = evaluators[name]
                scores = m.get_summary_scores()
                values = [
                    scores.get("Accuracy", 0),
                    scores.get("F1 Score", 0),
                    scores.get("Safety (1-HPR)", 0),
                    scores.get("TruthfulQA", 0),
                    scores.get("FActScore", 0),
                ]
                
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                bars = ax.bar(
                    x + offset + i * width, values, width,
                    label=self._format_strategy_name(name),
                    color=color, edgecolor="#111827", linewidth=0.8
                )

                # Add value labels
                for bar, val in zip(bars, values, strict=True):
                    ax.annotate(
                        f'{val:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold'
                    )

            ax.set_xlabel('Metric', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Evaluator Performance Comparison', fontsize=16, pad=18)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=11, loc='upper right')

            # Winner highlight
            winner = comparison_report.get_ranking("f1_score")[0]
            ax.text(
                0.02, 0.98, f"🏆 Winner: {winner}",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', color=CHART_COLORS["success"]
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}comparison_grouped_bar.png", report=comparison_report)

    def _create_comparison_latency_boxplot(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """Create latency boxplot comparing all evaluators."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        evaluators = comparison_report.evaluators
        if not evaluators:
            return None

        # Collect latencies
        latencies = []
        labels = []
        for name, m in evaluators.items():
            if m.latency.latencies_ms:
                latencies.append(m.latency.latencies_ms)
                labels.append(self._format_strategy_name(name))

        if not latencies:
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12, 7))

            bp = ax.boxplot(
                latencies, labels=labels, patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markerfacecolor=CHART_COLORS["danger"], markersize=3.5, alpha=0.35),
                medianprops=dict(color="white", linewidth=2.2),
                whiskerprops=dict(color="#374151", linewidth=1.3),
                capprops=dict(color="#374151", linewidth=1.3),
            )

            for i, patch in enumerate(bp["boxes"]):
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                patch.set_facecolor(color)
                patch.set_alpha(0.82)
                patch.set_edgecolor("#111827")
                patch.set_linewidth(1.2)

            # Annotate P50/P95
            y_max = max(np.percentile(lat, 95) for lat in latencies if len(lat) > 0)
            y_pad = max(50, y_max * 0.08)

            for i, lat in enumerate(latencies):
                if len(lat) == 0:
                    continue
                p50 = float(np.median(lat))
                p95 = float(np.percentile(lat, 95))
                ax.annotate(
                    f"P50: {p50:.0f}ms\nP95: {p95:.0f}ms",
                    xy=(i + 1, p95),
                    xytext=(i + 1, p95 + y_pad),
                    fontsize=10, color="#374151", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#e5e7eb"),
                )

            ax.set_title("Latency Distribution Comparison", fontsize=16, pad=18)
            ax.set_xlabel("Evaluator", fontsize=12)
            ax.set_ylabel("Latency (ms)", fontsize=12)

            # Find fastest
            fastest_idx = min(range(len(latencies)), key=lambda i: np.median(latencies[i]))
            ax.text(
                0.98, 0.98, f"⚡ Fastest: {labels[fastest_idx]}",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                color=CHART_COLORS["success"]
            )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}comparison_latency_boxplot.png", report=comparison_report)

    def _create_comparison_heatmap(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """Create heatmap of normalized scores per evaluator."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        evaluators = comparison_report.evaluators
        if not evaluators:
            return None

        metrics = ["Accuracy", "Precision", "Recall", "F1", "Safety", "TruthfulQA", "FActScore", "Speed"]
        evaluator_names = list(evaluators.keys())

        data = []
        for name in evaluator_names:
            m = evaluators[name]
            scores = m.get_summary_scores()
            row = [
                scores.get("Accuracy", 0),
                scores.get("Precision", 0),
                scores.get("Recall", 0),
                scores.get("F1 Score", 0),
                scores.get("Safety (1-HPR)", 0),
                scores.get("TruthfulQA", 0),
                scores.get("FActScore", 0),
                scores.get("Speed (1/P95)", 0),
            ]
            data.append(row)

        mat = np.array(data, dtype=float)

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(14, 5))

            im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

            ax.set_xticks(np.arange(len(metrics)))
            ax.set_xticklabels(metrics, fontsize=11)
            ax.set_yticks(np.arange(len(evaluator_names)))
            ax.set_yticklabels([self._format_strategy_name(n) for n in evaluator_names], fontsize=11)

            # Annotate cells
            for i in range(len(evaluator_names)):
                for j in range(len(metrics)):
                    val = mat[i, j]
                    color = "white" if val < 0.5 else "#111827"
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)

            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label("Score (0-1)", fontsize=11)

            ax.set_title("Evaluator Performance Heatmap", fontsize=16, pad=18)
            ax.grid(False)

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}comparison_heatmap.png", report=comparison_report)

    def _create_comparison_factscore_violin(
        self,
        comparison_report: ComparisonReport,
        prefix: str,
    ) -> Path | None:
        """Create violin plot for FActScore distribution comparison."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        evaluators = comparison_report.evaluators
        if not evaluators:
            return None

        # Collect FActScore distributions
        scores = []
        labels = []
        for name, m in evaluators.items():
            if m.factscore.scores:
                scores.append(m.factscore.scores)
                labels.append(self._format_strategy_name(name))

        if not scores:
            return None

        with self._mpl_style():
            fig, ax = plt.subplots(figsize=(12, 7))

            vp = ax.violinplot(scores, showmeans=True, showmedians=True)

            for i, body in enumerate(vp["bodies"]):
                color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                body.set_facecolor(color)
                body.set_edgecolor("#111827")
                body.set_alpha(0.75)

            if "cmeans" in vp:
                vp["cmeans"].set_color("#111827")
                vp["cmeans"].set_linewidth(2)
            if "cmedians" in vp:
                vp["cmedians"].set_color("white")
                vp["cmedians"].set_linewidth(2)

            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_ylabel("FActScore", fontsize=12)
            ax.set_title("FActScore Distribution by Evaluator", fontsize=16, pad=18)
            ax.set_ylim(0, 1.05)

            # Add mean annotations
            for i, score_list in enumerate(scores):
                mean_val = float(np.mean(score_list))
                ax.text(
                    i + 1, mean_val + 0.05, f"μ={mean_val:.1%}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold"
                )

            plt.tight_layout()
            return self._save_fig(fig, f"{prefix}comparison_factscore_violin.png", report=comparison_report)
