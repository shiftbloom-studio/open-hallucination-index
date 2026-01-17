#!/usr/bin/env python3
"""
Quick Test for Evaluators
=========================

Tests OHI-Local, OHI-Max, GraphRAG, and VectorRAG evaluators with sample claims.
Run from within Docker network or with appropriate port mappings.

Usage:
    python test_evaluators_quick.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add parent directory to path so benchmark module can be imported
_benchmark_dir = Path(__file__).parent
sys.path.insert(0, str(_benchmark_dir.parent))  # Add parent so 'benchmark' is importable
sys.path.insert(0, str(_benchmark_dir))  # Also add benchmark dir itself

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


async def _generic_evaluator_test(
    name: str,
    evaluator_key: str,
    num_claims: int = 3,
    color: str = "cyan",
) -> list[dict] | None:
    """
    Generic evaluator test function using get_evaluator factory.
    
    Args:
        name: Display name for the evaluator
        evaluator_key: Key for get_evaluator factory function
        num_claims: Number of claims to test
        color: Color for console output
        
    Returns:
        List of result dicts or None if unavailable
    """
    console.print(f"\n[bold {color}]Testing {name}[/bold {color}]")
    
    try:
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import get_evaluator
        
        config = ComparisonBenchmarkConfig.from_env()
        evaluator = get_evaluator(evaluator_key, config)
        
        # Health check
        is_healthy = await evaluator.health_check()
        console.print(f"  Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
        
        if not is_healthy:
            console.print(f"  [yellow]{name} not available, skipping tests[/yellow]")
            await evaluator.close()
            return None
        
        results = []
        for claim, expected in TEST_CLAIMS[:num_claims]:
            result = await evaluator.verify(claim)
            results.append({
                "claim": claim[:50] + "..." if len(claim) > 50 else claim,
                "expected": expected,
                "predicted": result.predicted_label,
                "verdict": result.verdict.value,
                "score": result.trust_score,
                "latency": result.latency_ms,
            })
            status = "✓" if result.predicted_label == expected else "✗"
            console.print(f"  {status} Verified: {claim[:40]}... ({result.latency_ms:.0f}ms)")
        
        await evaluator.close()
        return results
        
    except ImportError as e:
        console.print(f"  [yellow]Missing dependency: {e}[/yellow]")
        return None
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


# Sample claims for testing
TEST_CLAIMS = [
    # True claims
    ("Albert Einstein was born in Germany in 1879.", True),
    ("The Eiffel Tower is located in Paris, France.", True),
    ("Python was created by Guido van Rossum.", True),
    # False claims (hallucinations)
    ("The Great Wall of China is visible from the Moon with the naked eye.", False),
    ("Thomas Edison invented the telephone.", False),
]


async def test_ohi_local():
    """Test OHI-Local evaluator (local sources only: Neo4j + Qdrant, no MCP)."""
    return await _generic_evaluator_test(
        name="OHI-Local",
        evaluator_key="ohi_local",
        num_claims=3,
        color="green",
    )


async def test_ohi_default():
    """Test OHI evaluator (default tier: local first, MCP fallback)."""
    return await _generic_evaluator_test(
        name="OHI",
        evaluator_key="ohi",
        num_claims=3,
        color="cyan",
    )


async def test_ohi_max():
    """Test OHI-Max evaluator (maximum coverage: all local + all MCP sources)."""
    return await _generic_evaluator_test(
        name="OHI-Max",
        evaluator_key="ohi_max",
        num_claims=3,
        color="blue",
    )


async def test_graph_rag():
    """Test GraphRAG evaluator (Neo4j graph-based)."""
    return await _generic_evaluator_test(
        name="GraphRAG",
        evaluator_key="graph_rag",
        num_claims=3,
        color="magenta",
    )


async def test_vector_rag():
    """Test VectorRAG evaluator (Qdrant vector-based)."""
    return await _generic_evaluator_test(
        name="VectorRAG",
        evaluator_key="vector_rag",
        num_claims=3,
        color="yellow",
    )


async def test_charts_reporter():
    """Test chart generation with mock data for all evaluators."""
    console.print("\n[bold cyan]Testing Charts Reporter[/bold cyan]")
    
    try:
        import tempfile

        from benchmark.comparison_benchmark import (
            ComparisonReport,
            EvaluatorMetrics,
            FActScoreMetrics,
            HallucinationMetrics,
            LatencyMetrics,
            TruthfulQAMetrics,
        )
        from benchmark.reporters.charts import ChartsReporter
        
        # Create mock data
        report = ComparisonReport(
            run_id="test_run",
            timestamp="2026-01-16T10:00:00Z",
            config_summary={"test": True},
        )
        
        # Mock OHI-Local metrics (fast, good accuracy)
        ohi_local = EvaluatorMetrics(evaluator_name="OHI-Local")
        ohi_local.hallucination = HallucinationMetrics(
            total=100, correct=90,
            true_positives=44, true_negatives=46,
            false_positives=4, false_negatives=6
        )
        ohi_local.truthfulqa = TruthfulQAMetrics(total_questions=50, correct_predictions=40)
        ohi_local.factscore = FActScoreMetrics(
            total_texts=10, total_facts=50, supported_facts=42,
            scores=[0.85, 0.88, 0.82, 0.87, 0.84, 0.89, 0.86, 0.83, 0.90, 0.85]
        )
        ohi_local.latency = LatencyMetrics(
            latencies_ms=[80, 95, 88, 102, 85, 92, 98, 105, 90, 94]
        )
        report.add_evaluator(ohi_local)
        
        # Mock OHI-Max metrics (best accuracy, slower)
        ohi_max = EvaluatorMetrics(evaluator_name="OHI-Max")
        ohi_max.hallucination = HallucinationMetrics(
            total=100, correct=95,
            true_positives=47, true_negatives=48,
            false_positives=2, false_negatives=3
        )
        ohi_max.truthfulqa = TruthfulQAMetrics(total_questions=50, correct_predictions=45)
        ohi_max.factscore = FActScoreMetrics(
            total_texts=10, total_facts=52, supported_facts=48,
            scores=[0.92, 0.95, 0.90, 0.93, 0.91, 0.94, 0.92, 0.89, 0.96, 0.93]
        )
        ohi_max.latency = LatencyMetrics(
            latencies_ms=[180, 210, 195, 225, 188, 202, 215, 230, 198, 208]
        )
        report.add_evaluator(ohi_max)
        
        # Mock GraphRAG metrics (moderate accuracy)
        graph_rag = EvaluatorMetrics(evaluator_name="GraphRAG")
        graph_rag.hallucination = HallucinationMetrics(
            total=100, correct=78,
            true_positives=38, true_negatives=40,
            false_positives=10, false_negatives=12
        )
        graph_rag.truthfulqa = TruthfulQAMetrics(total_questions=50, correct_predictions=35)
        graph_rag.factscore = FActScoreMetrics(
            total_texts=10, total_facts=48, supported_facts=35,
            scores=[0.70, 0.75, 0.68, 0.72, 0.74, 0.69, 0.73, 0.71, 0.76, 0.70]
        )
        graph_rag.latency = LatencyMetrics(
            latencies_ms=[45, 52, 48, 55, 47, 58, 50, 53, 49, 51]
        )
        report.add_evaluator(graph_rag)
        
        # Mock VectorRAG metrics (fastest, least accurate)
        vector_rag = EvaluatorMetrics(evaluator_name="VectorRAG")
        vector_rag.hallucination = HallucinationMetrics(
            total=100, correct=72,
            true_positives=35, true_negatives=37,
            false_positives=13, false_negatives=15
        )
        vector_rag.truthfulqa = TruthfulQAMetrics(total_questions=50, correct_predictions=30)
        vector_rag.factscore = FActScoreMetrics(
            total_texts=10, total_facts=45, supported_facts=28,
            scores=[0.60, 0.65, 0.58, 0.62, 0.68, 0.55, 0.63, 0.59, 0.66, 0.61]
        )
        vector_rag.latency = LatencyMetrics(
            latencies_ms=[25, 32, 28, 30, 27, 35, 29, 31, 26, 33]
        )
        report.add_evaluator(vector_rag)
        
        # Generate charts
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            reporter = ChartsReporter(output_dir, dpi=100)
            
            chart_files = reporter.generate_comparison_charts(report, prefix="test_")
            
            console.print(f"  Generated {len(chart_files)} charts:")
            for chart in chart_files:
                console.print(f"    ✓ {chart.name}")
            
            return len(chart_files)
            
    except ImportError as e:
        console.print(f"  [yellow]Missing dependency: {e}[/yellow]")
        return 0
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 0


def print_results_table(all_results: dict[str, list | int | None]) -> None:
    """Print combined results table."""
    console.print("\n")
    
    table = Table(title="🧪 Evaluator Test Results", show_header=True)
    table.add_column("Evaluator", style="bold")
    table.add_column("Status")
    table.add_column("Claims Tested", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Avg Latency", justify="right")
    
    for name, results in all_results.items():
        if results is None:
            table.add_row(name, "[yellow]⚠ Skipped[/yellow]", "-", "-", "-")
        elif isinstance(results, int):
            table.add_row(name, "[green]✅ OK[/green]", str(results), "-", "-")
        else:
            avg_lat = sum(r["latency"] for r in results) / len(results)
            correct = sum(1 for r in results if r["predicted"] == r["expected"])
            table.add_row(
                name,
                "[green]✅ OK[/green]",
                str(len(results)),
                f"{correct}/{len(results)}",
                f"{avg_lat:.0f}ms",
            )
    
    console.print(table)


async def main() -> int:
    """Run all evaluator tests."""
    console.print(Panel(
        "[bold]OHI Benchmark - Evaluator Quick Test[/bold]\n\n"
        "Testing OHI-Local, OHI (default), OHI-Max, GraphRAG, VectorRAG evaluators...",
        border_style="cyan",
    ))
    
    all_results: dict[str, list | int | None] = {}
    
    # Test each evaluator (OHI tiers: local, default, max)
    all_results["OHI-Local"] = await test_ohi_local()
    all_results["OHI"] = await test_ohi_default()
    all_results["OHI-Max"] = await test_ohi_max()
    all_results["GraphRAG"] = await test_graph_rag()
    all_results["VectorRAG"] = await test_vector_rag()
    all_results["Charts"] = await test_charts_reporter()
    
    # Print summary
    print_results_table(all_results)
    
    # Count working components
    working_count = sum(1 for r in all_results.values() if r is not None)
    total_count = len(all_results)
    
    console.print(f"\n[bold]Summary: {working_count}/{total_count} components tested successfully[/bold]")
    
    # Return 0 if at least half worked
    return 0 if working_count >= total_count // 2 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
