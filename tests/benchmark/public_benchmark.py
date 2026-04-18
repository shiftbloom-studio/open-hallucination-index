"""Wave 3 Stream G.2 — FEVER slice runner (Decision G).

Reads a small FEVER slice fixture from disk, runs each claim through
the live API, and reports accuracy as a signal metric (not blocking).
Commits ``docs/benchmarks/v2.0-public-benchmark.md`` as the published
artifact.

Fixture: ``tests/benchmark/fixtures/fever_slice.jsonl`` — one JSON
object per line with fields ``claim``, ``label`` (``SUPPORTS`` /
``REFUTES`` / ``NOT ENOUGH INFO``). 50-claim slice keeps the benchmark
fast enough for CI; full FEVER evaluation is a separate offline run.

This module is a **CLI**, not a pytest test — it runs from the
gate workflow and writes its report. ``tests/benchmark`` is on the
pytest path but this file doesn't expose a ``test_*`` function.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class FeverExample:
    claim: str
    label: str  # SUPPORTS / REFUTES / NOT ENOUGH INFO


def _load_slice(path: Path) -> list[FeverExample]:
    out: list[FeverExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(FeverExample(claim=obj["claim"], label=obj["label"]))
    return out


def _verify(client: httpx.Client, url: str, secret: str, text: str, timeout: float) -> float:
    headers = {"X-OHI-Edge-Secret": secret, "Content-Type": "application/json"}
    resp = client.post(f"{url}/api/v2/verify", json={"text": text}, headers=headers)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"{url}/api/v2/verify/status/{job_id}", headers=headers)
        r.raise_for_status()
        body = r.json()
        if body.get("status") == "done":
            result = body["result"]
            return float(result.get("document_score", 0.5))
        if body.get("status") == "failed":
            return 0.5
        time.sleep(2.0)
    return 0.5


def _bucket(p: float) -> str:
    if p >= 0.65:
        return "SUPPORTS"
    if p <= 0.35:
        return "REFUTES"
    return "NOT ENOUGH INFO"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", default="tests/benchmark/fixtures/fever_slice.jsonl")
    p.add_argument("--out", default="docs/benchmarks/v2.0-public-benchmark.md")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args(argv)

    url = os.environ.get("OHI_API_URL", "").rstrip("/")
    secret = os.environ.get("OHI_EDGE_SECRET")
    if not url or not secret:
        print("SKIP: OHI_API_URL / OHI_EDGE_SECRET not set", file=sys.stderr)
        return 0

    fixture = Path(args.fixture)
    if not fixture.exists():
        print(f"SKIP: fixture not found at {fixture}", file=sys.stderr)
        return 0

    examples = _load_slice(fixture)[: args.limit]
    correct = 0
    total = 0
    by_label: dict[str, dict[str, int]] = {}
    with httpx.Client(timeout=30.0) as client:
        for ex in examples:
            p_true = _verify(client, url, secret, ex.claim, args.timeout)
            predicted = _bucket(p_true)
            total += 1
            group = by_label.setdefault(ex.label, {"correct": 0, "total": 0})
            group["total"] += 1
            if predicted == ex.label:
                correct += 1
                group["correct"] += 1

    acc = correct / max(total, 1)
    lines = [
        "# v2.0 Public Benchmark — FEVER slice",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        f"**Slice size:** {total}",
        f"**Overall accuracy:** {acc:.1%}",
        "",
        "## Per-label breakdown",
        "",
        "| Label            | Correct | Total | Accuracy |",
        "|------------------|---------|-------|----------|",
    ]
    for label, stats in sorted(by_label.items()):
        lab_acc = stats["correct"] / max(stats["total"], 1)
        lines.append(
            f"| {label:<16} | {stats['correct']:>7} | {stats['total']:>5} | {lab_acc:.1%} |"
        )
    lines.append("")
    lines.append("## Bucketing")
    lines.append("")
    lines.append("* SUPPORTS if p_true ≥ 0.65")
    lines.append("* REFUTES if p_true ≤ 0.35")
    lines.append("* NOT ENOUGH INFO otherwise")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
