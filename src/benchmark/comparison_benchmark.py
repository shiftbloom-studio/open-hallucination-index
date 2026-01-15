#!/usr/bin/env python3
"""
Comparison Benchmark CLI
========================

Command-line interface for running multi-evaluator comparison benchmarks.

Compares OHI vs GPT-4 vs VectorRAG across:
- Hallucination Detection
- TruthfulQA
- FActScore
- Latency

Usage:
    python -m benchmark.comparison_benchmark
    python -m benchmark.comparison_benchmark --evaluators ohi,gpt4
    python -m benchmark.comparison_benchmark --metrics hallucination,latency
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.comparison_runner import ComparisonBenchmarkRunner, run_comparison_benchmark


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OHI Comparison Benchmark - Compare verification systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comparison with all evaluators
  python -m benchmark.comparison_benchmark

  # Run only OHI vs GPT-4
  python -m benchmark.comparison_benchmark --evaluators ohi,gpt4

  # Run only hallucination and latency metrics
  python -m benchmark.comparison_benchmark --metrics hallucination,latency

  # Use custom dataset
  python -m benchmark.comparison_benchmark --dataset ./my_dataset.csv

  # Generate charts only (requires existing report)
  python -m benchmark.comparison_benchmark --charts-only --report report.json

Environment Variables:
  OHI_API_BASE_URL     - OHI API base URL (default: http://localhost:8080)
  OPENAI_API_KEY       - OpenAI API key for GPT-4 evaluator
  QDRANT_HOST          - Qdrant host for VectorRAG (default: localhost)
  QDRANT_PORT          - Qdrant port (default: 6333)
        """,
    )

    # Evaluator selection
    parser.add_argument(
        "--evaluators", "-e",
        type=str,
        default="ohi,gpt4,vector_rag",
        help="Comma-separated list of evaluators (ohi, gpt4, vector_rag)",
    )

    # Metric selection
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        default="hallucination,truthfulqa,factscore,latency",
        help="Comma-separated list of metrics",
    )

    # Dataset paths
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=None,
        help="Path to hallucination dataset CSV",
    )

    parser.add_argument(
        "--extended-dataset",
        type=Path,
        default=None,
        help="Path to extended HuggingFace-converted dataset",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results and charts",
    )

    parser.add_argument(
        "--chart-dpi",
        type=int,
        default=160,
        help="DPI for generated charts",
    )

    # Performance options
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Number of concurrent requests",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds",
    )

    # TruthfulQA options
    parser.add_argument(
        "--truthfulqa-max",
        type=int,
        default=200,
        help="Maximum TruthfulQA samples",
    )

    parser.add_argument(
        "--truthfulqa-categories",
        type=str,
        default=None,
        help="Comma-separated TruthfulQA categories",
    )

    # FActScore options
    parser.add_argument(
        "--factscore-max",
        type=int,
        default=100,
        help="Maximum FActScore samples",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    # Special modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running benchmark",
    )

    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Generate charts from existing report",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to existing report JSON (for --charts-only)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ComparisonBenchmarkConfig:
    """Build configuration from command-line arguments."""
    # Parse evaluators
    evaluators = [e.strip() for e in args.evaluators.split(",")]

    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(",")]

    # Parse TruthfulQA categories
    categories = None
    if args.truthfulqa_categories:
        categories = [c.strip() for c in args.truthfulqa_categories.split(",")]

    # Start with environment-based config
    config = ComparisonBenchmarkConfig.from_env()

    # Override with CLI arguments
    config.evaluators = evaluators  # type: ignore
    config.metrics = metrics  # type: ignore
    config.output_dir = args.output_dir
    config.chart_dpi = args.chart_dpi
    config.concurrency = args.concurrency
    config.timeout_seconds = args.timeout

    if args.dataset:
        config.hallucination_dataset = args.dataset

    if args.extended_dataset:
        config.extended_dataset = args.extended_dataset

    # TruthfulQA config
    config.truthfulqa.max_samples = args.truthfulqa_max
    if categories:
        config.truthfulqa.categories = categories

    # FActScore config
    config.factscore.max_samples = args.factscore_max

    return config


async def run_benchmark(args: argparse.Namespace) -> int:
    """Run the comparison benchmark."""
    console = Console()

    try:
        config = build_config(args)

        if args.dry_run:
            console.print("[bold]Dry run mode - validating configuration...[/bold]")
            console.print(f"  Evaluators: {config.evaluators}")
            console.print(f"  Metrics: {config.metrics}")
            console.print(f"  Output: {config.output_dir}")
            console.print(f"  Dataset: {config.hallucination_dataset}")
            
            # Check evaluator health
            from benchmark.evaluators import get_evaluator
            
            for eval_name in config.get_active_evaluators():
                try:
                    evaluator = get_evaluator(eval_name, config)
                    is_healthy = await evaluator.health_check()
                    status = "[green]✓ Available[/green]" if is_healthy else "[yellow]⚠ Unavailable[/yellow]"
                    console.print(f"  {evaluator.name}: {status}")
                    await evaluator.close()
                except Exception as e:
                    console.print(f"  {eval_name}: [red]✗ Error: {e}[/red]")
            
            return 0

        # Run benchmark
        async with ComparisonBenchmarkRunner(config, console) as runner:
            report = await runner.run_comparison()

        console.print(f"\n[green]✓[/green] Benchmark complete!")
        console.print(f"  Results saved to: {config.output_dir}")
        console.print(f"  Run ID: {report.run_id}")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        return 130

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        return 1


async def generate_charts_only(args: argparse.Namespace) -> int:
    """Generate charts from an existing report."""
    import json

    console = Console()

    if not args.report:
        console.print("[red]Error: --report is required with --charts-only[/red]")
        return 1

    if not args.report.exists():
        console.print(f"[red]Error: Report not found: {args.report}[/red]")
        return 1

    try:
        with open(args.report) as f:
            report_data = json.load(f)

        # Reconstruct ComparisonReport
        from benchmark.comparison_metrics import (
            ComparisonReport,
            EvaluatorMetrics,
            FActScoreMetrics,
            HallucinationMetrics,
            LatencyMetrics,
            TruthfulQAMetrics,
        )

        report = ComparisonReport(
            run_id=report_data.get("run_id", ""),
            timestamp=report_data.get("timestamp", ""),
            config_summary=report_data.get("config", {}),
        )

        for name, eval_data in report_data.get("evaluators", {}).items():
            metrics = EvaluatorMetrics(evaluator_name=name)
            
            # Parse hallucination metrics
            h = eval_data.get("hallucination", {})
            metrics.hallucination = HallucinationMetrics(
                total=h.get("total", 0),
                correct=int(h.get("accuracy", 0) * h.get("total", 1)),
            )
            
            # Parse other metrics similarly...
            report.add_evaluator(metrics)

        # Generate charts
        from benchmark.reporters.charts import ChartsReporter

        charts_dir = args.output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        reporter = ChartsReporter(args.output_dir, dpi=args.chart_dpi)
        chart_files = reporter.generate_comparison_charts(report)

        console.print(f"[green]✓[/green] Generated {len(chart_files)} charts")
        for chart_file in chart_files:
            console.print(f"  - {chart_file.name}")

        return 0

    except Exception as e:
        console.print(f"[red]Error generating charts: {e}[/red]")
        if args.verbose:
            console.print_exception()
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    if args.charts_only:
        return asyncio.run(generate_charts_only(args))
    else:
        return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    sys.exit(main())
