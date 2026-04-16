"""Phase 0 entrypoint: capture v1 baseline numbers across configured
benchmarks and write them to `benchmark_results/v1_baseline_<date>.jsonl`
plus a summary JSON.

Usage (from repo root):

    python -m scripts.benchmark.run_baseline --datasets factscore --limit 50

The v1 engine adapter is implemented in `scripts.benchmark.v1_engine_adapter`
(Task 0.3); this module is the CLI that drives it. In Phase 0 this file is
intentionally thin — actual v1 integration is done as part of Task 0.3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from scripts.benchmark.datasets import REGISTRY, list_datasets
from scripts.benchmark.metrics import summarize
from scripts.benchmark.runner import run_benchmark
from scripts.benchmark.types import (
    BenchmarkExample,
    BenchmarkResult,
    VerificationEngine,
)


class _Placeholder_V1Engine(VerificationEngine):
    """Placeholder v1 engine for Phase 0 wiring tests.

    Always predicts "true" with p_true=0.7. Replaced by
    V1EngineAdapter in Task 0.3 once we're ready to capture real
    baselines from the existing oracle.py + scorer.py stack.
    """

    name = "v1-placeholder"

    async def verify(self, example: BenchmarkExample) -> BenchmarkResult:
        return BenchmarkResult(
            example_id=example.id,
            predicted_label="true",
            p_true=0.7,
            interval=None,
            raw_response={"placeholder": True},
            latency_ms=0.0,
            error=None,
        )


async def _collect_examples(
    dataset_name: str, *, limit: int | None
) -> list[BenchmarkExample]:
    loader = REGISTRY[dataset_name]
    examples: list[BenchmarkExample] = []
    async for ex in loader(limit=limit):
        examples.append(ex)
    return examples


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OHI v1 baseline capture.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["factscore"],
        help=f"Dataset names (available: {', '.join(list_datasets())})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Per-dataset example limit (useful for smoke tests).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max in-flight engine calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Where to write the JSONL + summary.",
    )
    args = parser.parse_args(argv)

    engine = _Placeholder_V1Engine()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    summary_per_dataset: dict[str, dict[str, float | int]] = {}

    for ds in args.datasets:
        if ds not in REGISTRY:
            print(
                f"error: unknown dataset {ds!r}. known: {list_datasets()}",
                file=sys.stderr,
            )
            return 2

        examples = await _collect_examples(ds, limit=args.limit)
        if not examples:
            print(f"warning: dataset {ds!r} produced 0 examples (loader may be a stub)")
            continue

        jsonl_path = output_dir / f"v1_baseline_{today}.{ds}.jsonl"
        await run_benchmark(
            engine,
            examples,
            concurrency=args.concurrency,
            output_path=jsonl_path,
        )

        # Read back results for summary
        results: list[BenchmarkResult] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                results.append(
                    BenchmarkResult(
                        example_id=d["example_id"],
                        predicted_label=d["predicted_label"],
                        p_true=d["p_true"],
                        interval=tuple(d["interval"]) if d["interval"] else None,
                        raw_response=d["raw_response"],
                        latency_ms=d["latency_ms"],
                        error=d["error"],
                    )
                )

        truth_map = {ex.id: ex.expected_label for ex in examples}
        summary = summarize(
            results,
            predicted_label_fn=lambda r: r.predicted_label,
            true_label_fn=truth_map.__getitem__,
            is_true_fn=lambda eid: truth_map[eid] == "true",
        )
        summary_per_dataset[ds] = summary
        print(f"{ds}: {summary}")

    summary_path = output_dir / f"v1_baseline_{today}.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "engine": engine.name,
                "date": today,
                "per_dataset": summary_per_dataset,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote summary: {summary_path}")
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
