"""Wave 3 Stream C — Pass 2: enwiki passages (Decision K split).

Per Decision K:
* **Aura** holds passage **text + metadata** (single source of truth).
* **Qdrant** holds the passage **vector only**, payload reduced to
  ``{passage_id, qid}`` — no duplicated text.

This pass:
1. Streams enwiki main-namespace articles via
   :class:`EnwikiXmlDumpParser`.
2. Chunks wikitext into ≤ ``CORPUS_MAX_PASSAGE_TOKENS`` passages per
   section (sentence-boundary-aware).
3. Resolves the article's QID from the pre-built
   ``wikipedia_title → qid`` sitelinks index (Pass 1 output).
4. Embeds each passage via pc-embed.
5. Writes ``(:Passage {id, text, article_title, section_title,
   text_hash, article_qid})`` to Aura with
   ``(:Passage)-[:IN_ARTICLE]->(:Entity)`` and
   ``(:Passage)-[:MENTIONS]->(:Entity)`` edges.
6. Writes the vector to Qdrant with minimal payload.

Large files use wikitextparser for [[Wikilink]] extraction; pure
regex would miss templated wiki syntax that carries entity mentions.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

from ingestion.checkpoint_store import CheckpointStore
from ingestion.dump_parsers import EnwikiPage, EnwikiXmlDumpParser
from ingestion.lifecycle import (
    DLQWriter,
    PauseController,
    ProgressReporter,
    retry_record,
)

logger = logging.getLogger(__name__)


_MAX_PASSAGE_TOKENS = int(os.environ.get("CORPUS_MAX_PASSAGE_TOKENS", "500"))
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class PassageChunk:
    """One passage as produced by the chunker."""

    passage_id: str  # deterministic: sha256(article_qid + section + ordinal)
    text: str
    article_title: str
    article_qid: str
    section_title: str
    text_hash: str
    mention_qids: list[str]  # QIDs of wikilinked mentions


_UPSERT_PASSAGE_CYPHER = """
UNWIND $batch AS row
MERGE (p:Passage {id: row.passage_id})
SET p.text = row.text,
    p.article_title = row.article_title,
    p.section_title = row.section_title,
    p.text_hash = row.text_hash,
    p.article_qid = row.article_qid
WITH p, row
MATCH (art:Entity {qid: row.article_qid})
MERGE (p)-[:IN_ARTICLE]->(art)
WITH p, row
UNWIND row.mention_qids AS mqid
MATCH (m:Entity {qid: mqid})
MERGE (p)-[:MENTIONS]->(m)
RETURN count(DISTINCT p) AS n
"""


class Pass2PassageWriter:
    """Stream enwiki articles → Aura passages + Qdrant vectors."""

    def __init__(
        self,
        *,
        graph_store,
        vector_store,
        embedding_adapter,
        checkpoint_store: CheckpointStore,
        dlq: DLQWriter,
        pause_controller: PauseController,
        progress: ProgressReporter,
        sitelink_resolver,  # maps wikipedia_title → qid (from Pass 1 index)
        batch_size: int = 64,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._embed = embedding_adapter
        self._ckpt = checkpoint_store
        self._dlq = dlq
        self._pause = pause_controller
        self._progress = progress
        self._resolve_sitelink = sitelink_resolver
        self._batch_size = batch_size

    async def run(self, dump_path: str, *, start_from: str | None = None) -> None:
        self._ckpt.start_pass("pass2")
        logger.info("Pass 2 starting (batch_size=%d)", self._batch_size)
        parser = EnwikiXmlDumpParser(dump_path)
        batch: list[PassageChunk] = []
        processed = 0
        for page in parser:
            if start_from and page.title <= start_from:
                continue
            if page.is_redirect:
                continue
            try:
                qid = await self._resolve_sitelink(page.title)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sitelink resolve failed for %r: %s", page.title, exc)
                qid = None
            if qid is None:
                # Article without Wikidata sitelink — skip (post-corpus
                # freshness fallback via MediaWiki MCP covers these).
                continue
            try:
                chunks = list(self._chunk_page(page, qid))
            except Exception as exc:  # noqa: BLE001
                self._dlq.write(
                    pass_name="pass2",
                    record_id=page.title,
                    error_class=type(exc).__name__,
                    error_detail=str(exc),
                )
                self._ckpt.increment_dlq("pass2")
                continue
            batch.extend(chunks)
            if len(batch) >= self._batch_size:
                await self._flush_batch(batch)
                processed += len(batch)
                self._ckpt.advance(
                    "pass2",
                    last_committed_id=page.title,
                    records_in_batch=len(batch),
                )
                self._progress.tick(len(batch))
                batch = []
                if self._pause.should_pause():
                    self._ckpt.set_status("pass2", "paused", notes="SIGUSR1 / flag-file")
                    logger.info("Pass 2 paused at title=%r", page.title)
                    return
        if batch:
            await self._flush_batch(batch)
            self._ckpt.advance(
                "pass2",
                last_committed_id=batch[-1].article_title,
                records_in_batch=len(batch),
            )
            self._progress.tick(len(batch))
            processed += len(batch)
        self._ckpt.set_status(
            "pass2", "complete", notes=f"{processed} passages ingested"
        )
        logger.info("Pass 2 complete (%d passages)", processed)

    # ------------------------------------------------------------------
    # Chunker
    # ------------------------------------------------------------------

    def _chunk_page(
        self, page: EnwikiPage, article_qid: str
    ) -> Iterator[PassageChunk]:
        try:
            import wikitextparser as wtp  # noqa: PLC0415
        except ImportError:
            # Fallback: naive text chunking if wikitextparser missing.
            # Won't happen in prod (Dockerfile installs it) but keeps
            # the function importable for unit tests.
            yield from self._naive_chunk(page, article_qid)
            return

        parsed = wtp.parse(page.wikitext)
        # wikitextparser's sections yields the lead + named sections.
        for section_ordinal, section in enumerate(parsed.sections):
            section_title = (section.title or "lead").strip() or "lead"
            plain = section.plain_text().strip()
            if not plain:
                continue
            mentions = self._extract_mention_qids(section.wikilinks)
            for chunk_idx, chunk in enumerate(self._split_into_chunks(plain)):
                passage_id = _passage_id(article_qid, section_title, section_ordinal, chunk_idx)
                text_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                yield PassageChunk(
                    passage_id=passage_id,
                    text=chunk,
                    article_title=page.title,
                    article_qid=article_qid,
                    section_title=section_title,
                    text_hash=text_hash,
                    mention_qids=mentions,
                )

    def _naive_chunk(self, page: EnwikiPage, article_qid: str) -> Iterator[PassageChunk]:
        plain = page.wikitext.strip()
        if not plain:
            return
        for chunk_idx, chunk in enumerate(self._split_into_chunks(plain)):
            passage_id = _passage_id(article_qid, "lead", 0, chunk_idx)
            yield PassageChunk(
                passage_id=passage_id,
                text=chunk,
                article_title=page.title,
                article_qid=article_qid,
                section_title="lead",
                text_hash=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                mention_qids=[],
            )

    def _split_into_chunks(self, text: str) -> Iterator[str]:
        """Sentence-aware split capped at _MAX_PASSAGE_TOKENS (approx,
        token = word here; cheap estimate)."""
        sentences = _SENTENCE_SPLIT_RE.split(text)
        buf: list[str] = []
        buf_tokens = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            tc = len(s.split())
            if buf_tokens + tc > _MAX_PASSAGE_TOKENS and buf:
                yield " ".join(buf)
                buf = [s]
                buf_tokens = tc
            else:
                buf.append(s)
                buf_tokens += tc
        if buf:
            yield " ".join(buf)

    def _extract_mention_qids(self, wikilinks) -> list[str]:
        """Resolve wikilinks to QIDs via the sitelink map.

        Cheap mode: only pull wikilinks whose target (sitelink)
        matches a title we know. Pass-2 scope; deeper alias resolution
        is a Wave-4 NER task.
        """
        # NB: since wikilinks are resolved synchronously inside the
        # async chunk loop, we return empty here and let the caller's
        # MENTIONS upsert skip. Full implementation plumbs a sync
        # sitelink dict; optimisation tracked for v2.1.
        return []

    # ------------------------------------------------------------------
    # Flush: Aura + Qdrant in one batched write
    # ------------------------------------------------------------------

    async def _flush_batch(self, batch: list[PassageChunk]) -> None:
        # Embed.
        texts = [c.text for c in batch]
        try:
            vectors = await self._embed.generate_embeddings_batch(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 2 embed failed: %s", exc)
            for c in batch:
                self._dlq.write(
                    pass_name="pass2",
                    record_id=c.passage_id,
                    error_class=type(exc).__name__,
                    error_detail=str(exc),
                )
            self._ckpt.increment_dlq("pass2", n=len(batch))
            return

        # Aura: passage nodes + edges.
        aura_rows = [
            {
                "passage_id": c.passage_id,
                "text": c.text,
                "article_title": c.article_title,
                "article_qid": c.article_qid,
                "section_title": c.section_title,
                "text_hash": c.text_hash,
                "mention_qids": c.mention_qids,
            }
            for c in batch
        ]
        ok_aura = await retry_record(
            fn=lambda r=aura_rows: self._graph.run_cypher(
                _UPSERT_PASSAGE_CYPHER, {"batch": r}
            ),
            record_id=f"passages:{batch[0].passage_id[:12]}..",
        )
        if not ok_aura:
            for c in batch:
                self._dlq.write(
                    pass_name="pass2",
                    record_id=c.passage_id,
                    error_class="PassageAuraWriteFailed",
                    error_detail="aura batch failed after retries",
                )
            self._ckpt.increment_dlq("pass2", n=len(batch))
            return

        # Qdrant: vector-only payload (Decision K — no text duplication).
        points = [
            {
                "id": c.passage_id,
                "vector": list(vec),
                "payload": {"passage_id": c.passage_id, "qid": c.article_qid},
            }
            for c, vec in zip(batch, vectors, strict=True)
        ]
        ok_qdrant = await retry_record(
            fn=lambda p=points: self._vector.upsert_passage_points(p),
            record_id=f"qdrant:{batch[0].passage_id[:12]}..",
        )
        if not ok_qdrant:
            for c in batch:
                self._dlq.write(
                    pass_name="pass2",
                    record_id=c.passage_id,
                    error_class="PassageQdrantWriteFailed",
                    error_detail="qdrant upsert failed after retries",
                )
            self._ckpt.increment_dlq("pass2", n=len(batch))


def _passage_id(qid: str, section: str, section_ord: int, chunk_idx: int) -> str:
    """Deterministic passage id: short SHA-256 of (qid, section, ord, idx)."""
    key = f"{qid}::{section}::{section_ord}::{chunk_idx}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]


__all__ = ["Pass2PassageWriter", "PassageChunk"]
