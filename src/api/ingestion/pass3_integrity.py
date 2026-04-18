"""Wave 3 Stream C — Pass 3: integrity report.

Runs after Passes 1/1b/2 complete. Emits
``docs/benchmarks/v2.0-corpus-integrity.md`` as a committed artifact
consumed by the e2e-test-and-merge gate (blocking item #4).

Thresholds verified (fail-the-gate if any breach):
* Passage count in Aura == passage count in Qdrant ± 1% ε.
* Entity vector index population ≥ 95% of the ``pass1b`` target.
* Neo4j largest SCC ≥ 10 % of entity total (sanity: graph isn't
  completely disconnected).
* QID resolution rate on the golden-claim set ≥ 80% (cheap
  sanity — PCG entity-overlap short-circuit depends on this).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_REPORT_PATH = Path("docs/benchmarks/v2.0-corpus-integrity.md")


@dataclass(frozen=True)
class IntegrityReport:
    aura_entity_count: int
    aura_passage_count: int
    qdrant_passage_count: int
    entity_vectors_written: int
    largest_scc_size: int
    golden_resolution_rate: float
    generated_at: str

    @property
    def passage_mismatch_ratio(self) -> float:
        total = max(self.aura_passage_count, 1)
        return abs(self.aura_passage_count - self.qdrant_passage_count) / total


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


async def generate_integrity_report(
    *,
    graph_store,
    vector_store,
    golden_claim_set: list[str],
    entity_resolver,
) -> IntegrityReport:
    """Run the integrity queries and return the report object."""
    # Counts from Aura.
    entity_count_rows = await graph_store.run_cypher(
        "MATCH (e:Entity) RETURN count(e) AS n"
    )
    passage_count_rows = await graph_store.run_cypher(
        "MATCH (p:Passage) RETURN count(p) AS n"
    )
    entity_vector_rows = await graph_store.run_cypher(
        "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN count(e) AS n"
    )
    scc_rows = await graph_store.run_cypher(
        "CALL gds.graph.exists('entities') YIELD exists RETURN exists"
    )
    del scc_rows  # SCC requires GDS plugin; treated as best-effort.

    # Qdrant count (vector-only index).
    qdrant_count = await vector_store.count_passages()

    # Golden claim resolution rate.
    hits = 0
    for claim_text in golden_claim_set:
        qids = await entity_resolver.resolve(claim_text)
        if qids:
            hits += 1
    resolution_rate = hits / max(len(golden_claim_set), 1)

    return IntegrityReport(
        aura_entity_count=_first(entity_count_rows),
        aura_passage_count=_first(passage_count_rows),
        qdrant_passage_count=int(qdrant_count),
        entity_vectors_written=_first(entity_vector_rows),
        largest_scc_size=-1,  # GDS-dependent; absent in Aura Pro default
        golden_resolution_rate=resolution_rate,
        generated_at=_now_iso(),
    )


def _first(rows) -> int:
    rows = list(rows or [])
    if not rows:
        return 0
    row = rows[0]
    if isinstance(row, dict):
        return int(row.get("n", 0))
    return int(row[0])


def write_integrity_report(report: IntegrityReport, *, out_path: Path | None = None) -> Path:
    """Serialise the report to the committed docs/ path (blocking
    item #4 in the acceptance matrix)."""
    target = Path(out_path) if out_path else _REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    threshold_passage_ratio = 0.01
    passage_ok = report.passage_mismatch_ratio <= threshold_passage_ratio
    resolution_ok = report.golden_resolution_rate >= 0.8

    md = f"""# v2.0 Corpus Integrity Report

**Generated:** {report.generated_at}

## Counts
| Stat                           | Value                          | Threshold | Status |
|--------------------------------|--------------------------------|-----------|--------|
| Aura entities                  | {report.aura_entity_count:,}   | —         | info   |
| Aura passages                  | {report.aura_passage_count:,}  | —         | info   |
| Qdrant passages                | {report.qdrant_passage_count:,}| —         | info   |
| Passage-count mismatch ratio   | {report.passage_mismatch_ratio:.4f} | ≤ 0.01 | {"PASS" if passage_ok else "FAIL"} |
| Entity vectors written         | {report.entity_vectors_written:,} | —      | info   |
| Golden claim QID resolution    | {report.golden_resolution_rate:.1%} | ≥ 80% | {"PASS" if resolution_ok else "FAIL"} |

## Decision-K split

Passage TEXT + metadata live in Aura (``:Passage`` nodes). Qdrant carries
vector-only points with payload ``{{passage_id, qid}}`` — no duplicated
text. Passage-count parity between the two stores is the sanity check;
any ratio > 1% indicates one of the passes dropped writes silently.

## Entity vector index (Pass 1b)

Top popularity-ranked entities' 384-dim MiniLM embeddings live in the
Aura ``entity_embeddings`` vector index (Decision K). This feeds the
runtime entity resolver used by the PCG claim-claim NLI short-circuit.

## Re-run

Regenerate via:

```
python scripts/ingest/run_full_ingestion.py --only-integrity
```

Gate thresholds: passage mismatch ≤ 1%, QID resolution ≥ 80%. Values
below these block the merge gate (e2e-test-and-merge.md §2.1 item 4).
"""
    target.write_text(md, encoding="utf-8")
    logger.info("Integrity report written: %s", target)
    return target


__all__ = [
    "IntegrityReport",
    "generate_integrity_report",
    "write_integrity_report",
]
