"""
OHI Benchmark CLI
=================

Command-line interface for the benchmark suite.

Usage:
    python -m benchmark [OPTIONS]

Examples:
    python -m benchmark
    python -m benchmark --strategies vector_semantic,mcp_enhanced
    python -m benchmark --threshold 0.6 --concurrency 5
    python -m benchmark --verbose --warmup 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from benchmark.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_THRESHOLD,
    DEFAULT_WARMUP,
    get_config,
)
from benchmark.models import VerificationStrategy
from benchmark.runner import OHIBenchmarkRunner


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="OHI Benchmark Suite - Research-grade hallucination detection evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark
  python -m benchmark --strategies vector_semantic,mcp_enhanced
  python -m benchmark --threshold 0.6 --concurrency 5
  python -m benchmark --verbose --warmup 10
  python -m benchmark --output-dir ./results --bootstrap 2000

Environment Variables:
  OHI_API_HOST              API host (default: localhost)
  OHI_API_PORT              API port (default: 8080)
  BENCHMARK_DATASET         Path to CSV dataset
  BENCHMARK_OUTPUT_DIR      Output directory for reports
  BENCHMARK_CONCURRENCY     Parallel requests
  BENCHMARK_THRESHOLD       Decision threshold
        """,
    )

    # Strategy selection
    parser.add_argument(
        "--strategies",
        "-s",
        type=str,
        default="vector_semantic,mcp_enhanced",
        help="Comma-separated list of strategies to benchmark",
    )

    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Test all available strategies",
    )

    # Classification parameters
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Decision threshold for classification (default: {DEFAULT_THRESHOLD})",
    )

    # Execution parameters
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of parallel requests (default: {DEFAULT_CONCURRENCY})",
    )

    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup requests (default: {DEFAULT_WARMUP})",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )

    # Dataset and output
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to benchmark dataset CSV",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for reports",
    )

    # Statistical analysis
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for CI (default: 1000)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )

    # Output formats
    parser.add_argument(
        "--formats",
        type=str,
        default="csv,json,markdown,html",
        help="Comma-separated output formats (default: csv,json,markdown,html)",
    )

    # API configuration
    parser.add_argument(
        "--api-host",
        type=str,
        default=None,
        help="API host (overrides OHI_API_HOST)",
    )

    parser.add_argument(
        "--api-port",
        type=str,
        default=None,
        help="API port (overrides OHI_API_PORT)",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API caching during benchmark",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running benchmark",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 2.0.0",
    )

    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = create_parser()
    return parser.parse_args(args)


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    console = Console(record=True)

    # Display header
    console.print(
        Panel(
            "[bold magenta]OHI Benchmark Suite[/bold magenta]\n"
            "[dim]Research-Grade Hallucination Detection Evaluation[/dim]",
            border_style="magenta",
        )
    )

    # Parse strategies
    if args.all_strategies:
        strategies = VerificationStrategy.all_values()
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    # Build configuration
    config = get_config()

    overrides = {
        "strategies": strategies,
        "threshold": args.threshold,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup,
        "timeout_seconds": args.timeout,
        "bootstrap_iterations": args.bootstrap,
        "confidence_level": args.confidence,
        "verbose": args.verbose,
        "use_cache": not args.no_cache,
        "output_formats": [f.strip() for f in args.formats.split(",") if f.strip()],
    }

    if args.dataset:
        overrides["dataset_path"] = Path(args.dataset)
    if args.output_dir:
        overrides["output_dir"] = Path(args.output_dir)
    if args.api_host:
        overrides["api_host"] = args.api_host
    if args.api_port:
        overrides["api_port"] = args.api_port

    config = config.with_overrides(**overrides)

    # Dry run - just validate
    if args.dry_run:
        console.print("\n[bold cyan]Configuration (Dry Run):[/bold cyan]")
        console.print(f"  API: {config.api_base_url}")
        console.print(f"  Dataset: {config.dataset_path}")
        console.print(f"  Strategies: {', '.join(config.strategies)}")
        console.print(f"  Threshold: {config.threshold}")
        console.print(f"  Concurrency: {config.concurrency}")
        console.print(f"  Output: {config.output_dir}")
        console.print("\n[green]✔ Configuration valid[/green]")
        return 0

    # Run benchmark
    async with OHIBenchmarkRunner(config=config, console=console) as runner:
        try:
            await runner.run_benchmark()
            return 0
        except KeyboardInterrupt:
            console.print("\n[red]Benchmark aborted by user.[/red]")
            return 130
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 1


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parsed_args = parse_args(args)

    try:
        return asyncio.run(main_async(parsed_args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
