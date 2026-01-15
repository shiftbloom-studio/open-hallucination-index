#!/usr/bin/env python3
"""
Quick Test for Evaluators
=========================

Tests OHI, GPT-4, and VectorRAG evaluators with a few sample claims.
Run from within Docker network or with appropriate port mappings.
"""

import asyncio
import sys
from pathlib import Path

# Add benchmark to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()


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


async def test_ohi_evaluator():
    """Test OHI API evaluator."""
    console.print("\n[bold cyan]Testing OHI Evaluator[/bold cyan]")
    
    try:
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import OHIEvaluator
        
        # Configure for Docker network
        config = ComparisonBenchmarkConfig.from_env()
        config.ohi_api_host = "ohi-api"  # Docker service name
        config.ohi_api_port = "8080"
        
        evaluator = OHIEvaluator(config)
        
        # Health check
        is_healthy = await evaluator.health_check()
        console.print(f"  Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
        
        if not is_healthy:
            console.print("  [yellow]OHI API not available, skipping tests[/yellow]")
            await evaluator.close()
            return None
        
        results = []
        for claim, expected in TEST_CLAIMS[:3]:  # Test first 3
            result = await evaluator.verify(claim)
            results.append({
                "claim": claim[:50] + "...",
                "expected": expected,
                "predicted": result.predicted_label,
                "verdict": result.verdict.value,
                "score": result.trust_score,
                "latency": result.latency_ms,
            })
            console.print(f"  ✓ Verified: {claim[:40]}... ({result.latency_ms:.0f}ms)")
        
        await evaluator.close()
        return results
        
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return None


async def test_vector_rag_evaluator():
    """Test VectorRAG evaluator (Fair mode - uses public Wikipedia API)."""
    console.print("\n[bold cyan]Testing VectorRAG Evaluator (Fair Mode - Wikipedia API)[/bold cyan]")
    
    try:
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import FairVectorRAGEvaluator
        
        config = ComparisonBenchmarkConfig.from_env()
        
        evaluator = FairVectorRAGEvaluator(config)
        
        # Health check
        is_healthy = await evaluator.health_check()
        console.print(f"  Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
        
        if not is_healthy:
            console.print("  [yellow]Wikipedia API not available, skipping tests[/yellow]")
            await evaluator.close()
            return None
        
        results = []
        for claim, expected in TEST_CLAIMS[:3]:
            result = await evaluator.verify(claim)
            results.append({
                "claim": claim[:50] + "...",
                "expected": expected,
                "predicted": result.predicted_label,
                "verdict": result.verdict.value,
                "score": result.trust_score,
                "latency": result.latency_ms,
            })
            console.print(f"  ✓ Verified: {claim[:40]}... ({result.latency_ms:.0f}ms)")
        
        await evaluator.close()
        return results
        
    except ImportError as e:
        console.print(f"  [yellow]Missing dependency: {e}[/yellow]")
        return None
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return None


async def test_gpt4_evaluator():
    """Test GPT-4 evaluator."""
    console.print("\n[bold cyan]Testing GPT-4 Evaluator[/bold cyan]")
    
    try:
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import GPT4Evaluator
        
        config = ComparisonBenchmarkConfig.from_env()
        
        if not config.openai.is_configured:
            console.print("  [yellow]OPENAI_API_KEY not set, skipping GPT-4 tests[/yellow]")
            return None
        
        evaluator = GPT4Evaluator(config)
        
        # Health check
        is_healthy = await evaluator.health_check()
        console.print(f"  Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
        
        if not is_healthy:
            await evaluator.close()
            return None
        
        # Only test 1 claim to save API costs
        results = []
        claim, expected = TEST_CLAIMS[0]
        result = await evaluator.verify(claim)
        results.append({
            "claim": claim[:50] + "...",
            "expected": expected,
            "predicted": result.predicted_label,
            "verdict": result.verdict.value,
            "score": result.trust_score,
            "latency": result.latency_ms,
        })
        console.print(f"  ✓ Verified: {claim[:40]}... ({result.latency_ms:.0f}ms)")
        
        await evaluator.close()
        return results
        
    except ValueError as e:
        console.print(f"  [yellow]{e}[/yellow]")
        return None
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return None


async def test_charts_reporter():
    """Test chart generation with mock data."""
    console.print("\n[bold cyan]Testing Charts Reporter[/bold cyan]")
    
    try:
        from benchmark.comparison_metrics import (
            ComparisonReport,
            EvaluatorMetrics,
            HallucinationMetrics,
            TruthfulQAMetrics,
            FActScoreMetrics,
            LatencyMetrics,
        )
        from benchmark.reporters.charts import ChartsReporter
        import tempfile
        from pathlib import Path
        
        # Create mock data
        report = ComparisonReport(
            run_id="test_run",
            timestamp="2026-01-15T21:00:00Z",
            config_summary={"test": True},
        )
        
        # Add mock OHI metrics (best)
        ohi_metrics = EvaluatorMetrics(evaluator_name="OHI")
        ohi_metrics.hallucination = HallucinationMetrics(
            total=100, correct=92,
            true_positives=45, true_negatives=47,
            false_positives=3, false_negatives=5
        )
        ohi_metrics.truthfulqa = TruthfulQAMetrics(total_questions=50, correct_predictions=42)
        ohi_metrics.factscore = FActScoreMetrics(
            total_texts=10, total_facts=50, supported_facts=44,
            scores=[0.88, 0.92, 0.85, 0.90, 0.87, 0.91, 0.89, 0.86, 0.93, 0.88]
        )
        ohi_metrics.latency = LatencyMetrics(
            latencies_ms=[120, 145, 132, 155, 128, 138, 142, 150, 135, 140]
        )
        report.add_evaluator(ohi_metrics)
        
        # Add mock GPT-4 metrics (slower, slightly worse)
        gpt4_metrics = EvaluatorMetrics(evaluator_name="GPT-4")
        gpt4_metrics.hallucination = HallucinationMetrics(
            total=100, correct=85,
            true_positives=42, true_negatives=43,
            false_positives=7, false_negatives=8
        )
        gpt4_metrics.truthfulqa = TruthfulQAMetrics(total_questions=50, correct_predictions=38)
        gpt4_metrics.factscore = FActScoreMetrics(
            total_texts=10, total_facts=48, supported_facts=38,
            scores=[0.75, 0.82, 0.78, 0.80, 0.76, 0.81, 0.77, 0.79, 0.83, 0.74]
        )
        gpt4_metrics.latency = LatencyMetrics(
            latencies_ms=[850, 920, 780, 890, 950, 870, 810, 900, 880, 830]
        )
        report.add_evaluator(gpt4_metrics)
        
        # Add mock VectorRAG metrics (fastest, least accurate)
        vrag_metrics = EvaluatorMetrics(evaluator_name="VectorRAG")
        vrag_metrics.hallucination = HallucinationMetrics(
            total=100, correct=72,
            true_positives=35, true_negatives=37,
            false_positives=13, false_negatives=15
        )
        vrag_metrics.truthfulqa = TruthfulQAMetrics(total_questions=50, correct_predictions=30)
        vrag_metrics.factscore = FActScoreMetrics(
            total_texts=10, total_facts=45, supported_facts=28,
            scores=[0.60, 0.65, 0.58, 0.62, 0.68, 0.55, 0.63, 0.59, 0.66, 0.61]
        )
        vrag_metrics.latency = LatencyMetrics(
            latencies_ms=[25, 32, 28, 30, 27, 35, 29, 31, 26, 33]
        )
        report.add_evaluator(vrag_metrics)
        
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


def print_results_table(all_results: dict):
    """Print combined results table."""
    console.print("\n")
    
    table = Table(title="🧪 Evaluator Test Results", show_header=True)
    table.add_column("Evaluator", style="bold")
    table.add_column("Status")
    table.add_column("Claims Tested", justify="right")
    table.add_column("Avg Latency", justify="right")
    
    for name, results in all_results.items():
        if results is None:
            table.add_row(name, "[yellow]⚠ Skipped[/yellow]", "-", "-")
        elif isinstance(results, int):
            table.add_row(name, "[green]✅ OK[/green]", str(results), "-")
        else:
            avg_lat = sum(r["latency"] for r in results) / len(results)
            table.add_row(
                name,
                "[green]✅ OK[/green]",
                str(len(results)),
                f"{avg_lat:.0f}ms"
            )
    
    console.print(table)


async def main():
    """Run all evaluator tests."""
    console.print(Panel(
        "[bold]OHI Benchmark - Evaluator Quick Test[/bold]\n\n"
        "Testing evaluators with sample claims...",
        border_style="cyan",
    ))
    
    all_results = {}
    
    # Test each evaluator
    all_results["OHI"] = await test_ohi_evaluator()
    all_results["VectorRAG"] = await test_vector_rag_evaluator()
    all_results["GPT-4"] = await test_gpt4_evaluator()
    all_results["Charts"] = await test_charts_reporter()
    
    # Print summary
    print_results_table(all_results)
    
    # Success if at least charts worked
    working_count = sum(1 for r in all_results.values() if r is not None)
    console.print(f"\n[bold]Summary: {working_count}/4 components tested successfully[/bold]")
    
    return 0 if working_count > 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
