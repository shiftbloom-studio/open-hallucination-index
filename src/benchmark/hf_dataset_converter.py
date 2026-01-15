"""
Hugging Face Dataset Converter
==============================

Converts external hallucination datasets from Hugging Face Hub
into the OHI benchmark CSV format.

Supported datasets:
- aporia-ai/rag_hallucinations (recommended)

Usage:
    python -m benchmark.hf_dataset_converter --dataset aporia-ai/rag_hallucinations
    python -m benchmark.hf_dataset_converter --dataset aporia-ai/rag_hallucinations --merge
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed.")
    print("Install with: pip install datasets")
    sys.exit(1)

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# =============================================================================
# Constants
# =============================================================================

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_OUTPUT = BENCHMARK_DIR / "benchmark_dataset_extended.csv"
EXISTING_DATASET = BENCHMARK_DIR / "benchmark_dataset.csv"

# Domain classification keywords
DOMAIN_KEYWORDS = {
    "medical": [
        "patient", "disease", "treatment", "hospital", "doctor", "medicine",
        "symptom", "diagnosis", "therapy", "clinical", "health", "drug",
        "pharmaceutical", "surgery", "cancer", "virus", "infection"
    ],
    "technical": [
        "software", "code", "programming", "api", "database", "server",
        "algorithm", "python", "javascript", "docker", "kubernetes", "cloud",
        "machine learning", "ai", "neural", "model", "cpu", "gpu", "memory"
    ],
    "science": [
        "research", "experiment", "hypothesis", "physics", "chemistry",
        "biology", "quantum", "molecule", "atom", "cell", "evolution",
        "genetics", "dna", "climate", "planet", "galaxy", "universe"
    ],
    "legal": [
        "law", "court", "judge", "attorney", "legal", "regulation",
        "compliance", "contract", "litigation", "statute", "rights"
    ],
    "history": [
        "century", "war", "ancient", "medieval", "revolution", "empire",
        "civilization", "historical", "era", "dynasty", "king", "queen"
    ],
    "finance": [
        "stock", "market", "investment", "bank", "finance", "economy",
        "currency", "trading", "profit", "revenue", "tax", "accounting"
    ],
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConvertedCase:
    """A benchmark case converted from HF dataset."""
    id: int
    domain: str
    difficulty: str
    label: bool  # True = factual, False = hallucination
    text: str
    notes: str
    hallucination_type: str | None = None
    source_dataset: str = ""


# =============================================================================
# Domain Classification
# =============================================================================

def classify_domain(text: str, context: str = "") -> str:
    """
    Classify text into a domain based on keyword matching.
    
    Args:
        text: The claim text to classify.
        context: Optional context for better classification.
    
    Returns:
        Domain string (e.g., 'medical', 'technical', 'general').
    """
    combined = (text + " " + context).lower()
    
    domain_scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            domain_scores[domain] = score
    
    if domain_scores:
        return max(domain_scores, key=domain_scores.get)  # type: ignore
    return "general"


def estimate_difficulty(text: str, context: str = "") -> str:
    """
    Estimate difficulty based on text complexity.
    
    Heuristics:
    - Short, simple claims → easy
    - Medium length, some technical terms → medium
    - Long, complex, multi-clause → hard
    - Medical/legal with specific claims → critical
    """
    word_count = len(text.split())
    domain = classify_domain(text, context)
    
    # Critical domains get higher base difficulty
    if domain in ("medical", "legal"):
        if word_count > 30:
            return "critical"
        return "hard"
    
    # Technical gets medium-hard
    if domain == "technical":
        if word_count > 40:
            return "hard"
        return "medium"
    
    # General complexity estimation
    if word_count <= 15:
        return "easy"
    elif word_count <= 30:
        return "medium"
    else:
        return "hard"


def classify_hallucination_type(answer: str, context: str = "") -> str | None:
    """
    Attempt to classify the type of hallucination.
    
    Returns None if the claim is factual.
    """
    answer_lower = answer.lower()
    
    # Check for temporal claims
    if re.search(r'\b(19|20)\d{2}\b', answer) or re.search(r'(century|decade|year|month|day)', answer_lower):
        return "temporal_error"
    
    # Check for numerical claims
    if re.search(r'\b\d+(\.\d+)?\s*(percent|%|million|billion|thousand)', answer_lower):
        return "numerical_error"
    
    # Check for attribution claims (X said/created/invented Y)
    if re.search(r'(created|invented|developed|founded|said|stated) by', answer_lower):
        return "entity_swap"
    
    # Check for location/attribution
    if re.search(r'(located|based|headquartered|born) (in|at)', answer_lower):
        return "attribute_error"
    
    # Default to fabrication for RAG hallucinations
    return "fabrication"


# =============================================================================
# Dataset Converters
# =============================================================================

def convert_aporia_rag_hallucinations(
    start_id: int = 1,
    limit: int | None = None,
) -> list[ConvertedCase]:
    """
    Convert aporia-ai/rag_hallucinations dataset.
    
    Dataset structure:
        - context: Source document text
        - question: User question
        - answer: LLM response
        - hallucination: "hallucination" or "faithful"
    
    Args:
        start_id: Starting ID for new cases.
        limit: Maximum number of cases to convert.
    
    Returns:
        List of ConvertedCase objects.
    """
    console.print("[cyan]Loading aporia-ai/rag_hallucinations from Hugging Face...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading dataset...", total=None)
        dataset = load_dataset("aporia-ai/rag_hallucinations", split="train")
        progress.update(task, description="Dataset loaded!")
    
    cases: list[ConvertedCase] = []
    current_id = start_id
    
    total = len(dataset)
    if limit:
        total = min(total, limit)
    
    console.print(f"[green]Converting {total} entries...[/green]")
    
    for i, entry in enumerate(dataset):
        if limit and i >= limit:
            break
        
        context = entry.get("context", "")
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        label_str = entry.get("hallucination", "").lower()
        
        # Skip empty entries
        if not answer.strip():
            continue
        
        # Determine if factual (True) or hallucination (False)
        is_factual = label_str == "faithful"
        
        # Classify domain and difficulty
        domain = classify_domain(answer, context)
        difficulty = estimate_difficulty(answer, context)
        
        # Determine hallucination type (only for hallucinations)
        hallucination_type = None
        if not is_factual:
            hallucination_type = classify_hallucination_type(answer, context)
        
        # Create notes
        notes_parts = ["From aporia-ai/rag_hallucinations"]
        if question:
            # Truncate long questions
            q_short = question[:80] + "..." if len(question) > 80 else question
            notes_parts.append(f"Q: {q_short}")
        
        case = ConvertedCase(
            id=current_id,
            domain=domain,
            difficulty=difficulty,
            label=is_factual,
            text=answer.strip(),
            notes=" | ".join(notes_parts),
            hallucination_type=hallucination_type,
            source_dataset="aporia-ai/rag_hallucinations",
        )
        cases.append(case)
        current_id += 1
    
    return cases


# =============================================================================
# IO Functions
# =============================================================================

def load_existing_dataset(path: Path) -> tuple[list[dict[str, Any]], int]:
    """
    Load existing benchmark dataset and return max ID.
    
    Returns:
        Tuple of (existing rows, max_id).
    """
    if not path.exists():
        return [], 0
    
    rows: list[dict[str, Any]] = []
    max_id = 0
    
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            try:
                row_id = int(row.get("id", 0))
                max_id = max(max_id, row_id)
            except ValueError:
                pass
    
    return rows, max_id


def get_text_hash(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save_cases_to_csv(
    cases: list[ConvertedCase],
    output_path: Path,
    existing_rows: list[dict[str, Any]] | None = None,
) -> int:
    """
    Save converted cases to CSV file.
    
    Args:
        cases: List of ConvertedCase objects to save.
        output_path: Path to output CSV file.
        existing_rows: Optional existing rows to merge with.
    
    Returns:
        Number of cases written.
    """
    fieldnames = ["id", "domain", "difficulty", "label", "text", "notes", "hallucination_type"]
    
    # Build set of existing text hashes for deduplication
    existing_hashes: set[str] = set()
    if existing_rows:
        for row in existing_rows:
            text = row.get("text", "")
            existing_hashes.add(get_text_hash(text))
    
    # Filter duplicates
    unique_cases = []
    for case in cases:
        text_hash = get_text_hash(case.text)
        if text_hash not in existing_hashes:
            unique_cases.append(case)
            existing_hashes.add(text_hash)
    
    duplicates_skipped = len(cases) - len(unique_cases)
    if duplicates_skipped > 0:
        console.print(f"[yellow]Skipped {duplicates_skipped} duplicate entries[/yellow]")
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write existing rows first (if merging)
        if existing_rows:
            for row in existing_rows:
                # Ensure all fields exist
                clean_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(clean_row)
        
        # Write new cases
        for case in unique_cases:
            writer.writerow({
                "id": case.id,
                "domain": case.domain,
                "difficulty": case.difficulty,
                "label": str(case.label),
                "text": case.text,
                "notes": case.notes,
                "hallucination_type": case.hallucination_type or "",
            })
    
    total_rows = len(existing_rows or []) + len(unique_cases)
    return total_rows


# =============================================================================
# CLI
# =============================================================================

def print_stats(cases: list[ConvertedCase], title: str = "Conversion Statistics") -> None:
    """Print statistics table for converted cases."""
    # Domain distribution
    domains: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    labels = {"factual": 0, "hallucination": 0}
    
    for case in cases:
        domains[case.domain] = domains.get(case.domain, 0) + 1
        difficulties[case.difficulty] = difficulties.get(case.difficulty, 0) + 1
        if case.label:
            labels["factual"] += 1
        else:
            labels["hallucination"] += 1
    
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Category", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Total", "", str(len(cases)))
    table.add_row("", "", "")
    
    table.add_row("Labels", "Factual", str(labels["factual"]))
    table.add_row("", "Hallucination", str(labels["hallucination"]))
    table.add_row("", "", "")
    
    for domain, count in sorted(domains.items()):
        table.add_row("Domains", domain, str(count))
    table.add_row("", "", "")
    
    for diff, count in sorted(difficulties.items()):
        table.add_row("Difficulty", diff, str(count))
    
    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face hallucination datasets to OHI benchmark format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert Aporia dataset to new file
    python -m benchmark.hf_dataset_converter --dataset aporia

    # Merge with existing benchmark dataset
    python -m benchmark.hf_dataset_converter --dataset aporia --merge

    # Limit to 500 entries
    python -m benchmark.hf_dataset_converter --dataset aporia --limit 500
        """,
    )
    
    parser.add_argument(
        "--dataset", "-d",
        choices=["aporia", "aporia-ai/rag_hallucinations"],
        default="aporia",
        help="Dataset to convert (default: aporia)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: benchmark_dataset_extended.csv or merge target)",
    )
    
    parser.add_argument(
        "--merge", "-m",
        action="store_true",
        help="Merge with existing benchmark_dataset.csv",
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of entries to convert",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without saving",
    )
    
    args = parser.parse_args()
    
    console.print(Panel(
        "[bold cyan]OHI Hugging Face Dataset Converter[/bold cyan]\n"
        "Converting external datasets to benchmark format",
        border_style="cyan",
    ))
    
    # Determine start ID and existing rows
    existing_rows: list[dict[str, Any]] = []
    start_id = 1
    
    if args.merge:
        if EXISTING_DATASET.exists():
            existing_rows, max_id = load_existing_dataset(EXISTING_DATASET)
            start_id = max_id + 1
            console.print(f"[green]Loaded {len(existing_rows)} existing cases (max ID: {max_id})[/green]")
        else:
            console.print(f"[yellow]Warning: {EXISTING_DATASET} not found, creating new file[/yellow]")
    
    # Convert dataset
    if args.dataset in ("aporia", "aporia-ai/rag_hallucinations"):
        cases = convert_aporia_rag_hallucinations(start_id=start_id, limit=args.limit)
    else:
        console.print(f"[red]Unknown dataset: {args.dataset}[/red]")
        sys.exit(1)
    
    if not cases:
        console.print("[red]No cases converted![/red]")
        sys.exit(1)
    
    # Print statistics
    print_stats(cases, title="Converted Dataset Statistics")
    
    if args.dry_run:
        console.print("[yellow]Dry run - no files saved[/yellow]")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    elif args.merge:
        output_path = EXISTING_DATASET
    else:
        output_path = DEFAULT_OUTPUT
    
    # Save to CSV
    total_rows = save_cases_to_csv(
        cases=cases,
        output_path=output_path,
        existing_rows=existing_rows if args.merge else None,
    )
    
    console.print(Panel(
        f"[bold green]✔ Conversion Complete[/bold green]\n\n"
        f"Output: [cyan]{output_path}[/cyan]\n"
        f"Total rows: [green]{total_rows}[/green]\n"
        f"New cases: [green]{len(cases)}[/green]",
        title="Success",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
