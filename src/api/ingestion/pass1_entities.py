"""Wave 3 Stream C — Pass 1: Wikidata entity graph into Aura.

Streams the Wikidata JSON dump, projects each entity with an English
Wikipedia sitelink into a ``(:Entity {...})`` Neo4j node, and writes
Wikidata claim edges filtered by :data:`WIKIDATA_PROPERTY_EDGES` (the
default Wave 3 baseline, tunable via the ``CORPUS_WIKIDATA_PROPERTIES``
env var).

Batched Cypher writes via ``UNWIND`` for throughput. Default batch
size of 1,000 entities balances Aura's Bolt-channel throughput
against the per-record retry overhead when a single record fails
serialisation.
"""

from __future__ import annotations

import logging
import os

from ingestion.checkpoint_store import CheckpointStore
from ingestion.dump_parsers import WikidataEntity, WikidataJsonDumpParser
from ingestion.lifecycle import (
    DLQWriter,
    PauseController,
    ProgressReporter,
    retry_record,
)

logger = logging.getLogger(__name__)


# Wave 3 default Wikidata property filter (spec §2.1).
WIKIDATA_PROPERTY_EDGES: dict[str, str] = {
    "P31": "INSTANCE_OF",
    "P279": "SUBCLASS_OF",
    "P361": "PART_OF",
    "P17": "COUNTRY",
    "P131": "ADM_LOCATION",
    "P527": "HAS_PART",
    "P569": "DATE_OF_BIRTH",
    "P570": "DATE_OF_DEATH",
    "P19": "PLACE_OF_BIRTH",
    "P20": "PLACE_OF_DEATH",
    "P106": "OCCUPATION",
}


def _override_edges_from_env() -> dict[str, str]:
    """``CORPUS_WIKIDATA_PROPERTIES`` env var override.

    Format: comma-separated ``P31:INSTANCE_OF,P279:SUBCLASS_OF,...``.
    When unset, falls back to :data:`WIKIDATA_PROPERTY_EDGES`.
    """
    raw = os.environ.get("CORPUS_WIKIDATA_PROPERTIES", "").strip()
    if not raw:
        return dict(WIKIDATA_PROPERTY_EDGES)
    result: dict[str, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        pid, edge = item.split(":", 1)
        pid = pid.strip()
        edge = edge.strip().upper()
        if pid and edge:
            result[pid] = edge
    return result or dict(WIKIDATA_PROPERTY_EDGES)


_UPSERT_ENTITY_CYPHER = """
UNWIND $batch AS row
MERGE (e:Entity {qid: row.qid})
SET e.label = row.label,
    e.label_lower = toLower(coalesce(row.label, '')),
    e.description = row.description,
    e.wikipedia_title = row.wikipedia_title,
    e.inbound_link_count = coalesce(e.inbound_link_count, 0)
RETURN count(e) AS n
"""


_UPSERT_EDGE_CYPHER_TEMPLATE = """
UNWIND $batch AS row
MATCH (src:Entity {qid: row.src_qid})
MERGE (tgt:Entity {qid: row.tgt_qid})
ON CREATE SET tgt.label = row.tgt_label
MERGE (src)-[r:{edge_type}]->(tgt)
RETURN count(r) AS n
"""


_RECOMPUTE_INBOUND_LINK_COUNT_CYPHER = """
MATCH (e:Entity)
OPTIONAL MATCH (src:Entity)-[r]->(e)
WITH e, count(r) AS inbound_link_count
SET e.inbound_link_count = inbound_link_count
RETURN count(e) AS n
"""


class Pass1EntityWriter:
    """Run Pass 1 — Wikidata → Aura entities + property edges."""

    def __init__(
        self,
        *,
        graph_store,
        checkpoint_store: CheckpointStore,
        dlq: DLQWriter,
        pause_controller: PauseController,
        progress: ProgressReporter,
        batch_size: int = 1000,
        total_estimate: int | None = None,
    ) -> None:
        self._graph = graph_store
        self._ckpt = checkpoint_store
        self._dlq = dlq
        self._pause = pause_controller
        self._progress = progress
        self._batch_size = batch_size
        self._edge_filter = _override_edges_from_env()
        self._total_estimate = total_estimate

    async def run(self, dump_path: str, *, start_from: str | None = None) -> None:
        """Drain the Wikidata dump into Aura. ``start_from`` is the
        previously committed eligible-entity count from a checkpoint."""
        self._ckpt.start_pass("pass1")
        logger.info(
            "Pass 1 starting (batch_size=%d, edge_types=%d, start_from=%s)",
            self._batch_size,
            len(self._edge_filter),
            start_from,
        )
        resume_offset = _resume_offset(start_from, pass_name="pass1")
        parser = WikidataJsonDumpParser(dump_path)
        batch: list[WikidataEntity] = []
        seen = 0
        committed = resume_offset
        processed = 0
        for entity in parser:
            seen += 1
            if seen <= resume_offset:
                continue
            batch.append(entity)
            if len(batch) >= self._batch_size:
                if not await self._flush_batch(batch):
                    self._ckpt.set_status(
                        "pass1",
                        "aborted",
                        notes=f"batch failed before checkpoint at entity_offset={seen}",
                    )
                    raise RuntimeError("Pass 1 aborted after an unrecoverable batch failure")
                processed += len(batch)
                committed += len(batch)
                self._ckpt.advance(
                    "pass1",
                    last_committed_id=str(committed),
                    records_in_batch=len(batch),
                )
                self._progress.tick(len(batch))
                batch = []
                if self._pause.should_pause():
                    self._ckpt.set_status("pass1", "paused", notes="SIGUSR1 / flag-file")
                    logger.info("Pass 1 paused at qid=%s", entity.qid)
                    return
        if batch:
            if not await self._flush_batch(batch):
                self._ckpt.set_status(
                    "pass1",
                    "aborted",
                    notes=f"final batch failed before checkpoint at entity_offset={seen}",
                )
                raise RuntimeError("Pass 1 aborted after an unrecoverable final-batch failure")
            committed += len(batch)
            self._ckpt.advance(
                "pass1",
                last_committed_id=str(committed),
                records_in_batch=len(batch),
            )
            self._progress.tick(len(batch))
            processed += len(batch)
        ok = await retry_record(
            fn=lambda: self._graph.run_cypher(_RECOMPUTE_INBOUND_LINK_COUNT_CYPHER, {}),
            record_id="pass1:recompute_inbound_link_count",
        )
        if not ok:
            self._dlq.write(
                pass_name="pass1",
                record_id="inbound_link_count",
                error_class="InboundLinkCountRecomputeFailed",
                error_detail="failed after retries",
            )
            self._ckpt.increment_dlq("pass1")
            self._ckpt.set_status(
                "pass1",
                "aborted",
                notes="failed to recompute inbound_link_count",
            )
            raise RuntimeError("Pass 1 failed to recompute inbound_link_count")
        self._ckpt.set_status("pass1", "complete", notes=f"{processed} entities ingested")
        logger.info("Pass 1 complete (%d entities)", processed)

    async def _flush_batch(self, batch: list[WikidataEntity]) -> bool:
        entity_rows = [
            {
                "qid": e.qid,
                "label": e.label,
                "description": e.description,
                "wikipedia_title": e.wikipedia_title,
            }
            for e in batch
        ]
        ok = await retry_record(
            fn=lambda: self._graph.run_cypher(_UPSERT_ENTITY_CYPHER, {"batch": entity_rows}),
            record_id=f"entities:{batch[0].qid}..{batch[-1].qid}",
        )
        if not ok:
            for e in batch:
                self._dlq.write(
                    pass_name="pass1",
                    record_id=e.qid,
                    error_class="BatchUpsertFailed",
                    error_detail="entity batch failed after retries",
                )
            self._ckpt.increment_dlq("pass1", n=len(batch))
            return False

        # Edge writes — one batch per relationship type.
        by_edge_type: dict[str, list[dict[str, str]]] = {}
        for e in batch:
            for pid, edge_type in self._edge_filter.items():
                for claim in e.claims.get(pid, []):
                    mainsnak = claim.get("mainsnak", {})
                    if mainsnak.get("snaktype") != "value":
                        continue
                    dv = mainsnak.get("datavalue", {}).get("value")
                    target_qid = None
                    if isinstance(dv, dict) and "id" in dv:
                        target_qid = dv["id"]
                    elif isinstance(dv, str):
                        target_qid = dv
                    if not target_qid or not isinstance(target_qid, str):
                        continue
                    by_edge_type.setdefault(edge_type, []).append(
                        {"src_qid": e.qid, "tgt_qid": target_qid, "tgt_label": target_qid}
                    )

        for edge_type, rows in by_edge_type.items():
            cypher = _UPSERT_EDGE_CYPHER_TEMPLATE.replace("{edge_type}", edge_type)
            batch_ok = await retry_record(
                fn=lambda c=cypher, r=rows: self._graph.run_cypher(c, {"batch": r}),
                record_id=f"edges:{edge_type}:{batch[0].qid}..{batch[-1].qid}",
            )
            if not batch_ok:
                self._dlq.write(
                    pass_name="pass1",
                    record_id=f"edges:{edge_type}:{batch[0].qid}..{batch[-1].qid}",
                    error_class="EdgeBatchUpsertFailed",
                    error_detail=f"{edge_type} batch failed after retries",
                    extra={"edge_type": edge_type, "batch_len": len(rows)},
                )
                self._ckpt.increment_dlq("pass1")
                return False

        return True


def _resume_offset(start_from: str | None, *, pass_name: str) -> int:
    if start_from is None:
        return 0
    if start_from.isdigit():
        return int(start_from)
    raise ValueError(
        f"{pass_name} resume marker must be a numeric offset; "
        "legacy checkpoints require --force-restart"
    )


__all__ = [
    "Pass1EntityWriter",
    "WIKIDATA_PROPERTY_EDGES",
]
