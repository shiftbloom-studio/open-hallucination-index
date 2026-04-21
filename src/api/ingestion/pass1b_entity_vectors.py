"""Wave 3 Stream C — Pass 1b: entity vector index (Decision K).

After Pass 1 lands the full entity graph, Pass 1b:

1. Queries Aura for the top-K entities by ``inbound_link_count`` (the
   popularity prior). K defaults to 1,800,000 — the Aura Pro 2.8 GiB
   vector-optimisation carve-out can hold ~1.8M 384-dim vectors.
2. Embeds each entity's ``label`` via the pc-embed service.
3. Writes the embedding to Aura's vector index
   ``entity_embeddings`` via Cypher.

The index must exist before writing; first invocation runs the
``db.index.vector.createNodeIndex`` call. Lives on the Aura side
(single source of truth, Decision K) — no Qdrant hop for entity
resolution.
"""

from __future__ import annotations

import logging

from ingestion.checkpoint_store import CheckpointStore
from ingestion.lifecycle import (
    DLQWriter,
    PauseController,
    ProgressReporter,
    retry_record,
)

logger = logging.getLogger(__name__)


_CREATE_VECTOR_INDEX_CYPHER = """
CALL db.index.vector.createNodeIndex(
    'entity_embeddings',
    'Entity',
    'embedding',
    $dim,
    'cosine'
) YIELD name
RETURN name
"""


_SELECT_TOP_K_CYPHER = """
MATCH (e:Entity)
WHERE e.label IS NOT NULL
WITH e
ORDER BY coalesce(e.inbound_link_count, 0) DESC, e.qid ASC
LIMIT $top_k
WITH e
WHERE e.embedding IS NULL
RETURN e.qid AS qid, e.label AS label, e.description AS description
ORDER BY coalesce(e.inbound_link_count, 0) DESC, e.qid ASC
LIMIT $batch_size
"""


_WRITE_EMBEDDING_CYPHER = """
UNWIND $batch AS row
MATCH (e:Entity {qid: row.qid})
CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
RETURN count(e) AS n
"""


def _entity_embedding_text(label: str, description: str | None) -> str:
    description_text = (description or "").strip()
    if not description_text:
        return label
    if description_text.casefold() == label.casefold():
        return label
    return f"{label}\n{description_text}"


class Pass1bEntityVectorWriter:
    """Embed top-K entities and write to Aura vector index."""

    def __init__(
        self,
        *,
        graph_store,
        embedding_adapter,
        checkpoint_store: CheckpointStore,
        dlq: DLQWriter,
        pause_controller: PauseController,
        progress: ProgressReporter,
        top_k: int = 1_800_000,
        batch_size: int = 256,
        dim: int = 384,
    ) -> None:
        self._graph = graph_store
        self._embed = embedding_adapter
        self._ckpt = checkpoint_store
        self._dlq = dlq
        self._pause = pause_controller
        self._progress = progress
        self._top_k = top_k
        self._batch_size = batch_size
        self._dim = dim

    async def run(self, *, start_from: str | None = None) -> None:
        """Runs the pass from ``start_from`` (or fresh if None)."""
        self._ckpt.start_pass("pass1b")
        logger.info(
            "Pass 1b starting (top_k=%d, batch_size=%d, dim=%d)",
            self._top_k,
            self._batch_size,
            self._dim,
        )
        await self._ensure_index()

        committed = int(start_from) if start_from and start_from.isdigit() else 0
        processed = 0
        while True:
            rows = await self._graph.run_cypher(
                _SELECT_TOP_K_CYPHER,
                {"top_k": self._top_k, "batch_size": self._batch_size},
            )
            rows = list(rows or [])
            if not rows:
                break
            labels = [_entity_embedding_text(r["label"], r.get("description")) for r in rows]
            try:
                vectors = await self._embed.generate_embeddings_batch(labels)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pass 1b batch embed failed: %s", exc)
                for r in rows:
                    self._dlq.write(
                        pass_name="pass1b",
                        record_id=str(r["qid"]),
                        error_class=type(exc).__name__,
                        error_detail=str(exc),
                    )
                self._ckpt.increment_dlq("pass1b", n=len(rows))
                self._ckpt.set_status(
                    "pass1b",
                    "aborted",
                    notes=f"embedding batch failed at qid={rows[0]['qid']}",
                )
                raise RuntimeError("Pass 1b aborted after an unrecoverable embed failure") from exc

            payload = [
                {"qid": r["qid"], "embedding": list(vec)}
                for r, vec in zip(rows, vectors, strict=True)
            ]
            ok = await retry_record(
                fn=lambda p=payload: self._graph.run_cypher(_WRITE_EMBEDDING_CYPHER, {"batch": p}),
                record_id=f"entvec:{payload[0]['qid']}..{payload[-1]['qid']}",
            )
            if not ok:
                for row in payload:
                    self._dlq.write(
                        pass_name="pass1b",
                        record_id=str(row["qid"]),
                        error_class="VectorWriteFailed",
                        error_detail="batch failed after retries",
                    )
                self._ckpt.increment_dlq("pass1b", n=len(payload))
                self._ckpt.set_status(
                    "pass1b",
                    "aborted",
                    notes=f"vector write failed at qid={payload[0]['qid']}",
                )
                raise RuntimeError("Pass 1b aborted after an unrecoverable write failure")
            else:
                processed += len(payload)
                committed += len(payload)
                self._ckpt.advance(
                    "pass1b",
                    last_committed_id=str(committed),
                    records_in_batch=len(payload),
                )
                self._progress.tick(len(payload))

            if self._pause.should_pause():
                self._ckpt.set_status("pass1b", "paused", notes="SIGUSR1 / flag-file")
                logger.info("Pass 1b paused after %d committed vectors", committed)
                return

        self._ckpt.set_status("pass1b", "complete", notes=f"{processed} entity vectors written")
        logger.info("Pass 1b complete (%d entity vectors)", processed)

    async def _ensure_index(self) -> None:
        try:
            await self._graph.run_cypher(_CREATE_VECTOR_INDEX_CYPHER, {"dim": self._dim})
        except Exception as exc:  # noqa: BLE001
            # Index likely already exists — log at info, keep going.
            logger.info("Aura vector index create: %s (ok if already exists)", exc)


__all__ = ["Pass1bEntityVectorWriter"]
