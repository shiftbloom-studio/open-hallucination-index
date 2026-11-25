#!/usr/bin/env python3
"""
Detailed Evaluator Test
=======================

Tests evaluators with detailed output and checks verification correctness.
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path so benchmark module can be imported
_benchmark_dir = Path(__file__).parent
sys.path.insert(0, str(_benchmark_dir.parent))  # Add parent so 'benchmark' is importable
sys.path.insert(0, str(_benchmark_dir))  # Also add benchmark dir itself


console = Console()


# Sample claims for testing - balanced true/false
TEST_CLAIMS = [
    # True claims (should be SUPPORTED)
    ("Albert Einstein was born in Germany in 1879.", True),
    ("The Eiffel Tower is located in Paris, France.", True),
    ("Python was created by Guido van Rossum.", True),
    # False claims (should be REFUTED or UNVERIFIABLE)
    ("The Great Wall of China is visible from the Moon with the naked eye.", False),
    ("Thomas Edison invented the telephone.", False),  # Bell invented it
    ("The capital of Australia is Sydney.", False),  # Canberra
]


async def test_ohi_detailed():
    """Test OHI with detailed results."""
    console.print("\n[bold cyan]═══ OHI Evaluator Detailed Test ═══[/bold cyan]")

    from benchmark.comparison_config import ComparisonBenchmarkConfig
    from benchmark.evaluators import OHIEvaluator

    config = ComparisonBenchmarkConfig.from_env()
    config.ohi_api_host = "ohi-api"
    config.ohi_api_port = "8080"

    evaluator = OHIEvaluator(config)

    if not await evaluator.health_check():
        console.print("[red]OHI API not available[/red]")
        return

    table = Table(title="OHI Verification Results", show_header=True)
    table.add_column("Claim", max_width=40)
    table.add_column("Expected")
    table.add_column("Verdict")
    table.add_column("Score", justify="right")
    table.add_column("Correct?")
    table.add_column("Latency", justify="right")

    correct = 0
    for claim, expected_true in TEST_CLAIMS:
        result = await evaluator.verify(claim)

        predicted = result.predicted_label
        is_correct = predicted == expected_true
        correct += int(is_correct)

        table.add_row(
            claim[:38] + "..." if len(claim) > 40 else claim,
            "✓ True" if expected_true else "✗ False",
            result.verdict.value.upper(),
            f"{result.trust_score:.2f}",
            "[green]✓[/green]" if is_correct else "[red]✗[/red]",
            f"{result.latency_ms:.0f}ms",
        )

    console.print(table)
    console.print(
        f"\n[bold]Accuracy: {correct}/{len(TEST_CLAIMS)} ({100 * correct / len(TEST_CLAIMS):.0f}%)[/bold]"
    )

    await evaluator.close()


async def test_vectorrag_detailed():
    """Test VectorRAG with detailed results."""
    console.print("\n[bold cyan]═══ VectorRAG Evaluator Detailed Test ═══[/bold cyan]")

    from benchmark.comparison_config import ComparisonBenchmarkConfig
    from benchmark.evaluators import VectorRAGEvaluator

    config = ComparisonBenchmarkConfig.from_env()
    config.vector_rag.qdrant_host = "qdrant"
    config.vector_rag.collection_name = "wikipedia_hybrid"

    evaluator = VectorRAGEvaluator(config)

    if not await evaluator.health_check():
        console.print("[red]Qdrant not available[/red]")
        return

    table = Table(title="VectorRAG Verification Results", show_header=True)
    table.add_column("Claim", max_width=40)
    table.add_column("Expected")
    table.add_column("Verdict")
    table.add_column("Similarity", justify="right")
    table.add_column("Correct?")
    table.add_column("Latency", justify="right")

    correct = 0
    for claim, expected_true in TEST_CLAIMS:
        result = await evaluator.verify(claim)

        predicted = result.predicted_label
        is_correct = predicted == expected_true
        correct += int(is_correct)

        table.add_row(
            claim[:38] + "..." if len(claim) > 40 else claim,
            "✓ True" if expected_true else "✗ False",
            result.verdict.value.upper(),
            f"{result.trust_score:.2f}",
            "[green]✓[/green]" if is_correct else "[red]✗[/red]",
            f"{result.latency_ms:.0f}ms",
        )

    console.print(table)
    console.print(
        f"\n[bold]Accuracy: {correct}/{len(TEST_CLAIMS)} ({100 * correct / len(TEST_CLAIMS):.0f}%)[/bold]"
    )

    await evaluator.close()


async def test_factscore_decomposition():
    """Test atomic fact decomposition with OHI."""
    console.print("\n[bold cyan]═══ FActScore Decomposition Test ═══[/bold cyan]")

    from benchmark.comparison_config import ComparisonBenchmarkConfig
    from benchmark.evaluators import OHIEvaluator

    config = ComparisonBenchmarkConfig.from_env()
    config.ohi_api_host = "ohi-api"

    evaluator = OHIEvaluator(config)

    if not await evaluator.health_check():
        console.print("[red]OHI API not available[/red]")
        return

    test_text = (
        "Albert Einstein was born in Germany in 1879. "
        "He developed the theory of relativity and won the Nobel Prize in Physics in 1921 "
        "for his explanation of the photoelectric effect."
    )

    console.print(f"\n[dim]Test text:[/dim] {test_text}\n")

    result = await evaluator.decompose_and_verify(test_text)

    console.print(f"[bold]Total atomic facts:[/bold] {result.total_facts}")
    console.print(f"[bold]Supported facts:[/bold] {result.supported_facts}")
    console.print(f"[bold]FActScore:[/bold] {result.factscore:.1%}")
    console.print(f"[bold]Latency:[/bold] {result.latency_ms:.0f}ms")

    if result.atomic_facts:
        table = Table(title="Extracted Atomic Facts", show_header=True)
        table.add_column("#", width=3)
        table.add_column("Fact")
        table.add_column("Verified")

        for i, fact in enumerate(result.atomic_facts, 1):
            verified_str = (
                "[green]✓[/green]"
                if fact.verified
                else "[red]✗[/red]"
                if fact.verified is False
                else "[yellow]?[/yellow]"
            )
            table.add_row(str(i), fact.text[:60], verified_str)

        console.print(table)

    await evaluator.close()


async def main():
    """Run detailed tests."""
    console.print(
        Panel(
            "[bold]OHI Benchmark - Detailed Evaluator Test[/bold]",
            border_style="cyan",
        )
    )

    await test_ohi_detailed()
    await test_vectorrag_detailed()
    await test_factscore_decomposition()

    console.print("\n[bold green]All tests completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
