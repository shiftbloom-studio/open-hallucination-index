"""
Performance Charts Reporter
============================

Generate publication-ready performance visualization charts:
1. Latency Distribution (Box Plot) - Compare response times across strategies
2. Throughput Comparison (Bar Chart) - Requests/second per strategy
3. Latency Histogram (Stacked) - Response time distribution details
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


# Modern color palette for charts (accessible & print-friendly)
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
    
    Produces 3 publication-ready PNG charts:
    1. latency_boxplot.png - Box plot comparing latency distributions
    2. throughput_bar.png - Bar chart of throughput per strategy
    3. latency_histogram.png - Stacked histogram of response times
    """
    
    def __init__(self, output_dir: Path, dpi: int = 150) -> None:
        """
        Initialize charts reporter.
        
        Args:
            output_dir: Directory for chart PNG files.
            dpi: Resolution for PNG export (default 150 for quality/size balance).
        """
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
        """
        Generate all charts and return summary of created files.
        
        Returns:
            Summary string listing created chart files.
        """
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
        """
        Generate all 3 performance charts.
        
        Args:
            report: Benchmark report with aggregated metrics.
            results: Raw result metrics.
            base_filename: Optional prefix for chart filenames.
            
        Returns:
            List of paths to generated chart files.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            # Return empty if matplotlib not available
            return []
        
        prefix = f"{base_filename}_" if base_filename else ""
        chart_files: list[Path] = []
        
        # Filter valid results (no errors)
        valid_results = [r for r in results if not r.has_error]
        
        if not valid_results:
            return chart_files
        
        # Group by strategy
        strategies = list(report.strategy_reports.keys())
        results_by_strategy = {
            s: [r for r in valid_results if r.strategy == s]
            for s in strategies
        }
        
        # 1. Latency Box Plot
        chart1 = self._create_latency_boxplot(
            results_by_strategy, strategies, report, prefix
        )
        if chart1:
            chart_files.append(chart1)
        
        # 2. Throughput Bar Chart
        chart2 = self._create_throughput_chart(
            results_by_strategy, strategies, report, prefix
        )
        if chart2:
            chart_files.append(chart2)
        
        # 3. Latency Histogram
        chart3 = self._create_latency_histogram(
            results_by_strategy, strategies, report, prefix
        )
        if chart3:
            chart_files.append(chart3)
        
        return chart_files
    
    def _create_latency_boxplot(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        """Create box plot comparing latency distributions across strategies."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        
        # Prepare data
        latencies = [
            [r.latency_ms for r in results_by_strategy.get(s, [])]
            for s in strategies
        ]
        
        # Skip if no data
        if not any(latencies):
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('#fafafa')
        
        # Create box plot with custom styling
        bp = ax.boxplot(
            latencies,
            labels=[self._format_strategy_name(s) for s in strategies],
            patch_artist=True,
            showfliers=True,
            flierprops=dict(
                marker='o',
                markerfacecolor='#dc2626',
                markersize=4,
                alpha=0.5,
                markeredgecolor='none',
            ),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color='#374151', linewidth=1.5),
            capprops=dict(color='#374151', linewidth=1.5),
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], STRATEGY_COLORS[:len(strategies)], strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('#1f2937')
            patch.set_linewidth(1.5)
        
        # Add percentile annotations
        for i, (_s, data) in enumerate(zip(strategies, latencies, strict=True)):
            if data:
                p50 = np.median(data)
                p95 = np.percentile(data, 95)
                ax.annotate(
                    f'P50: {p50:.0f}ms\nP95: {p95:.0f}ms',
                    xy=(i + 1, p95),
                    xytext=(i + 1.3, p95 + max(data) * 0.05),
                    fontsize=9,
                    color='#374151',
                    ha='left',
                    va='bottom',
                )
        
        # Styling
        ax.set_title(
            'Response Latency Distribution by Strategy',
            fontsize=16,
            fontweight='bold',
            color='#1f2937',
            pad=20,
        )
        ax.set_xlabel('Verification Strategy', fontsize=12, color='#374151')
        ax.set_ylabel('Latency (ms)', fontsize=12, color='#374151')
        
        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#d1d5db')
        ax.set_axisbelow(True)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=15, ha='right')
        
        # Add subtitle with test info
        ax.text(
            0.5, -0.12,
            f'Dataset: {report.dataset_size} cases | Threshold: {report.threshold_used} | Run: {report.run_id}',
            transform=ax.transAxes,
            fontsize=10,
            color='#6b7280',
            ha='center',
        )
        
        plt.tight_layout()
        
        # Save
        filepath = self._charts_dir / f"{prefix}latency_boxplot.png"
        fig.savefig(str(filepath), dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return filepath
    
    def _create_throughput_chart(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        """Create bar chart showing throughput (requests/second) per strategy."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        
        # Calculate throughput metrics
        throughputs = []
        avg_latencies = []
        p95_latencies = []
        
        for s in strategies:
            strategy_results = results_by_strategy.get(s, [])
            if strategy_results:
                latencies = [r.latency_ms for r in strategy_results]
                avg_lat = np.mean(latencies)
                p95_lat = np.percentile(latencies, 95)
                # Throughput based on P50 (realistic sustained throughput)
                p50_lat = np.median(latencies)
                throughput = 1000.0 / p50_lat if p50_lat > 0 else 0
                throughputs.append(throughput)
                avg_latencies.append(avg_lat)
                p95_latencies.append(p95_lat)
            else:
                throughputs.append(0)
                avg_latencies.append(0)
                p95_latencies.append(0)
        
        if not any(throughputs):
            return None
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
        ax1.set_facecolor('#fafafa')
        
        x = np.arange(len(strategies))
        width = 0.6
        
        # Throughput bars
        bars = ax1.bar(
            x,
            throughputs,
            width,
            color=[STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i in range(len(strategies))],
            edgecolor='#1f2937',
            linewidth=1.5,
            alpha=0.85,
        )
        
        # Add value labels on bars
        for bar, tp, avg, p95 in zip(bars, throughputs, avg_latencies, p95_latencies, strict=True):
            height = bar.get_height()
            ax1.annotate(
                f'{tp:.1f} req/s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color='#1f2937',
            )
            # Add latency details inside bar
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 0.5,
                    f'Avg: {avg:.0f}ms\nP95: {p95:.0f}ms',
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='white',
                    fontweight='medium',
                )
        
        # Styling
        ax1.set_title(
            'Throughput Comparison by Strategy',
            fontsize=16,
            fontweight='bold',
            color='#1f2937',
            pad=20,
        )
        ax1.set_xlabel('Verification Strategy', fontsize=12, color='#374151')
        ax1.set_ylabel('Throughput (requests/second)', fontsize=12, color='#374151')
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [self._format_strategy_name(s) for s in strategies],
            rotation=15,
            ha='right',
        )
        
        # Grid
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7, color='#d1d5db')
        ax1.set_axisbelow(True)
        
        # Y-axis starts at 0
        ax1.set_ylim(0, max(throughputs) * 1.25)
        
        # Add total throughput annotation
        total_requests = sum(len(results_by_strategy.get(s, [])) for s in strategies)
        total_time = report.total_runtime_seconds
        overall_throughput = total_requests / total_time if total_time > 0 else 0
        
        ax1.text(
            0.02, 0.98,
            f'Total: {total_requests} requests in {total_time:.1f}s\n'
            f'Overall: {overall_throughput:.2f} req/s',
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#d1d5db'),
        )
        
        # Subtitle
        ax1.text(
            0.5, -0.12,
            f'Throughput based on P50 latency (sustainable rate) | Run: {report.run_id}',
            transform=ax1.transAxes,
            fontsize=10,
            color='#6b7280',
            ha='center',
        )
        
        plt.tight_layout()
        
        # Save
        filepath = self._charts_dir / f"{prefix}throughput_bar.png"
        fig.savefig(str(filepath), dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return filepath
    
    def _create_latency_histogram(
        self,
        results_by_strategy: dict[str, list[ResultMetric]],
        strategies: list[str],
        report: BenchmarkReport,
        prefix: str,
    ) -> Path | None:
        """Create overlapping histogram showing latency distribution."""
        try:
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        
        # Prepare data
        all_latencies = []
        for s in strategies:
            strategy_results = results_by_strategy.get(s, [])
            all_latencies.extend([r.latency_ms for r in strategy_results])
        
        if not all_latencies:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
        ax.set_facecolor('#fafafa')
        
        # Determine bin edges (shared across all strategies)
        max_latency = np.percentile(all_latencies, 99)  # Exclude extreme outliers
        bins = np.linspace(0, max_latency, 40)
        
        # Plot overlapping histograms
        legend_handles = []
        for i, s in enumerate(strategies):
            strategy_results = results_by_strategy.get(s, [])
            if not strategy_results:
                continue
            
            latencies = [r.latency_ms for r in strategy_results if r.latency_ms <= max_latency]
            
            color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            
            counts, _, patches = ax.hist(
                latencies,
                bins=bins,
                alpha=0.5,
                color=color,
                edgecolor=color,
                linewidth=1.5,
                label=self._format_strategy_name(s),
            )
            
            # Add density line (KDE approximation)
            if len(latencies) > 10:
                from scipy import stats
                kde = stats.gaussian_kde(latencies)
                x_smooth = np.linspace(0, max_latency, 200)
                y_smooth = kde(x_smooth) * len(latencies) * (bins[1] - bins[0])
                ax.plot(
                    x_smooth,
                    y_smooth,
                    color=color,
                    linewidth=2.5,
                    linestyle='-',
                    alpha=0.9,
                )
            
            # Legend entry
            legend_handles.append(
                mpatches.Patch(
                    color=color,
                    alpha=0.6,
                    label=f'{self._format_strategy_name(s)} (n={len(latencies)})',
                )
            )
        
        # Add vertical lines for overall percentiles
        all_valid = [lat for lat in all_latencies if lat <= max_latency]
        if all_valid:
            p50 = np.median(all_valid)
            p95 = np.percentile(all_valid, 95)
            
            ax.axvline(
                p50,
                color='#059669',
                linestyle='--',
                linewidth=2,
                label=f'Overall P50: {p50:.0f}ms',
            )
            ax.axvline(
                p95,
                color='#dc2626',
                linestyle='--',
                linewidth=2,
                label=f'Overall P95: {p95:.0f}ms',
            )
        
        # Styling
        ax.set_title(
            'Response Time Distribution',
            fontsize=16,
            fontweight='bold',
            color='#1f2937',
            pad=20,
        )
        ax.set_xlabel('Latency (ms)', fontsize=12, color='#374151')
        ax.set_ylabel('Frequency', fontsize=12, color='#374151')
        
        # Legend
        ax.legend(
            loc='upper right',
            fontsize=10,
            framealpha=0.95,
            edgecolor='#d1d5db',
        )
        
        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#d1d5db')
        ax.set_axisbelow(True)
        
        # X-axis formatting
        ax.set_xlim(0, max_latency * 1.02)
        
        # Add performance zones
        # Fast (0-500ms), Medium (500-2000ms), Slow (2000ms+)
        if max_latency > 500:
            ax.axvspan(0, 500, alpha=0.08, color='#059669', label='Fast (<500ms)')
        if max_latency > 2000:
            ax.axvspan(500, 2000, alpha=0.08, color='#d97706', label='Medium')
            ax.axvspan(2000, max_latency, alpha=0.08, color='#dc2626', label='Slow')
        elif max_latency > 500:
            ax.axvspan(500, max_latency, alpha=0.08, color='#d97706', label='Medium')
        
        # Subtitle
        ax.text(
            0.5, -0.10,
            f'Distribution capped at P99 ({max_latency:.0f}ms) | Total samples: {len(all_latencies)} | Run: {report.run_id}',
            transform=ax.transAxes,
            fontsize=10,
            color='#6b7280',
            ha='center',
        )
        
        plt.tight_layout()
        
        # Save
        filepath = self._charts_dir / f"{prefix}latency_histogram.png"
        fig.savefig(str(filepath), dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return filepath
    
    def _format_strategy_name(self, strategy: str) -> str:
        """Format strategy name for display."""
        name_map = {
            "vector_semantic": "Vector\n(Semantic)",
            "graph_exact": "Graph\n(Exact)",
            "hybrid": "Hybrid",
            "cascading": "Cascading",
            "mcp_enhanced": "MCP\nEnhanced",
            "adaptive": "Adaptive",
        }
        return name_map.get(strategy, strategy.replace("_", " ").title())
    
    def save(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
        filename: str | None = None,
    ) -> Path:
        """
        Generate and save all charts.
        
        Returns:
            Path to the charts directory.
        """
        base_filename = filename or f"ohi_benchmark_{report.run_id}"
        self.generate_all_charts(report, results, base_filename)
        return self._charts_dir
