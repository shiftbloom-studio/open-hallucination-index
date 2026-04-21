"""Wave 3 Stream C — Pass 2: enwiki passages (Decision K split).

Per Decision K:
* **Aura** holds passage **text + metadata** (single source of truth).
* **Qdrant** holds the passage **vector only**, payload reduced to
  ``{passage_id, qid}`` — no duplicated text.

This pass:
1. Streams enwiki main-namespace articles via
   :class:`EnwikiXmlDumpParser`.
2. Chunks wikitext into retrieval-optimized passages using
   ``CORPUS_CHUNK_*`` token budget controls.
3. Resolves the article's QID from the pre-built
   ``wikipedia_title → qid`` sitelinks index (Pass 1 output).
4. Embeds each passage via the configured embedding adapter.
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


_CHARS_PER_TOKEN = 4.7
_CHUNK_TARGET_TOKENS = int(os.environ.get("CORPUS_CHUNK_TARGET_TOKENS", "320"))
_CHUNK_MAX_TOKENS = int(os.environ.get("CORPUS_CHUNK_MAX_TOKENS", "420"))
_CHUNK_OVERLAP_TOKENS = int(os.environ.get("CORPUS_CHUNK_OVERLAP_TOKENS", "64"))
_CHUNK_MIN_TOKENS = int(os.environ.get("CORPUS_CHUNK_MIN_TOKENS", "96"))
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_LOW_SIGNAL_SECTION_TITLES = {
    "bibliography",
    "citations",
    "external links",
    "further reading",
    "notes",
    "references",
    "see also",
    "sources",
}
_SKIP_WIKILINK_NAMESPACES = (
    "category:",
    "draft:",
    "file:",
    "help:",
    "image:",
    "media:",
    "module:",
    "portal:",
    "special:",
    "talk:",
    "template:",
    "user:",
    "wikipedia:",
)


@dataclass(frozen=True)
class PassageChunk:
    """One passage as produced by the chunker."""

    passage_id: str  # deterministic: sha256(article_qid + section + ordinal)
    text: str
    article_title: str
    article_qid: str
    section_title: str
    section_ordinal: int
    chunk_ordinal: int
    token_estimate: int
    text_hash: str
    mention_titles: list[str]  # Raw Wikipedia titles extracted from wikilinks


_UPSERT_PASSAGE_CYPHER = """
UNWIND $batch AS row
MERGE (p:Passage {id: row.passage_id})
SET p.text = row.text,
    p.article_title = row.article_title,
    p.section_title = row.section_title,
    p.section_ordinal = row.section_ordinal,
    p.chunk_ordinal = row.chunk_ordinal,
    p.token_estimate = row.token_estimate,
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


_RESOLVE_MENTION_TITLES_CYPHER = """
UNWIND $titles AS title
MATCH (e:Entity {wikipedia_title: title})
RETURN e.wikipedia_title AS title, e.qid AS qid
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
        resume_offset = _resume_offset(start_from, pass_name="pass2")
        parser = EnwikiXmlDumpParser(dump_path)
        batch: list[PassageChunk] = []
        batch_last_page_ordinal: int | None = None
        page_ordinal = 0
        processed = 0
        for page in parser:
            page_ordinal += 1
            if page_ordinal <= resume_offset:
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
            if chunks:
                batch_last_page_ordinal = page_ordinal
            if len(batch) >= self._batch_size:
                if not await self._flush_batch(batch):
                    self._ckpt.set_status(
                        "pass2",
                        "aborted",
                        notes=(
                            "batch failed before checkpoint at page_offset="
                            f"{batch_last_page_ordinal}"
                        ),
                    )
                    raise RuntimeError("Pass 2 aborted after an unrecoverable batch failure")
                processed += len(batch)
                self._ckpt.advance(
                    "pass2",
                    last_committed_id=str(batch_last_page_ordinal),
                    records_in_batch=len(batch),
                )
                self._progress.tick(len(batch))
                batch = []
                batch_last_page_ordinal = None
                if self._pause.should_pause():
                    self._ckpt.set_status("pass2", "paused", notes="SIGUSR1 / flag-file")
                    logger.info("Pass 2 paused at title=%r", page.title)
                    return
        if batch:
            if not await self._flush_batch(batch):
                self._ckpt.set_status(
                    "pass2",
                    "aborted",
                    notes=(
                        "final batch failed before checkpoint at page_offset="
                        f"{batch_last_page_ordinal}"
                    ),
                )
                raise RuntimeError("Pass 2 aborted after an unrecoverable final-batch failure")
            self._ckpt.advance(
                "pass2",
                last_committed_id=str(batch_last_page_ordinal),
                records_in_batch=len(batch),
            )
            self._progress.tick(len(batch))
            processed += len(batch)
        self._ckpt.set_status("pass2", "complete", notes=f"{processed} passages ingested")
        logger.info("Pass 2 complete (%d passages)", processed)

    # ------------------------------------------------------------------
    # Chunker
    # ------------------------------------------------------------------

    def _chunk_page(self, page: EnwikiPage, article_qid: str) -> Iterator[PassageChunk]:
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
            if self._is_low_signal_section(section_title):
                continue
            plain = section.plain_text().strip()
            if not plain:
                continue
            mentions = self._extract_mention_titles(section.wikilinks)
            for chunk_idx, (chunk, token_estimate) in enumerate(
                self._split_into_chunks(plain)
            ):
                passage_id = _passage_id(article_qid, section_title, section_ordinal, chunk_idx)
                text_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                yield PassageChunk(
                    passage_id=passage_id,
                    text=chunk,
                    article_title=page.title,
                    article_qid=article_qid,
                    section_title=section_title,
                    section_ordinal=section_ordinal,
                    chunk_ordinal=chunk_idx,
                    token_estimate=token_estimate,
                    text_hash=text_hash,
                    mention_titles=mentions,
                )

    def _naive_chunk(self, page: EnwikiPage, article_qid: str) -> Iterator[PassageChunk]:
        plain = page.wikitext.strip()
        if not plain:
            return
        for chunk_idx, (chunk, token_estimate) in enumerate(self._split_into_chunks(plain)):
            passage_id = _passage_id(article_qid, "lead", 0, chunk_idx)
            yield PassageChunk(
                passage_id=passage_id,
                text=chunk,
                article_title=page.title,
                article_qid=article_qid,
                section_title="lead",
                section_ordinal=0,
                chunk_ordinal=chunk_idx,
                token_estimate=token_estimate,
                text_hash=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                mention_titles=[],
            )

    def _split_into_chunks(self, text: str) -> Iterator[tuple[str, int]]:
        """Retrieval-first chunking with overlap and approximate token budgeting."""
        if not text.strip():
            return

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        sentence_stream: list[str] = []
        for paragraph in paragraphs:
            raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
            if not raw_sentences:
                continue
            for sentence in raw_sentences:
                sentence_stream.extend(self._split_long_sentence(sentence))

        if not sentence_stream:
            return

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentence_stream:
            sentence_tokens = _estimate_tokens(sentence)
            if current and (
                current_tokens + sentence_tokens > _CHUNK_MAX_TOKENS
                or (
                    current_tokens >= _CHUNK_TARGET_TOKENS
                    and current_tokens + sentence_tokens > _CHUNK_TARGET_TOKENS
                )
            ):
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                overlap = self._tail_for_overlap(current)
                current = list(overlap)
                current_tokens = sum(_estimate_tokens(s) for s in current)

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        chunks = self._merge_small_chunks(chunks)
        for chunk in chunks:
            token_estimate = _estimate_tokens(chunk)
            if token_estimate <= 0:
                continue
            yield chunk, token_estimate

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Split pathological long sentences so hard-max constraints hold."""
        if _estimate_tokens(sentence) <= _CHUNK_MAX_TOKENS:
            return [sentence]
        words = sentence.split()
        out: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for word in words:
            word_tokens = _estimate_tokens(word)
            if buf and buf_tokens + word_tokens > _CHUNK_MAX_TOKENS:
                out.append(" ".join(buf))
                buf = [word]
                buf_tokens = word_tokens
            else:
                buf.append(word)
                buf_tokens += word_tokens
        if buf:
            out.append(" ".join(buf))
        return out

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge tiny tail chunks into neighbors to avoid low-signal fragments."""
        if len(chunks) <= 1:
            return chunks
        merged: list[str] = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            if _estimate_tokens(chunk) < _CHUNK_MIN_TOKENS:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)
        if len(merged) > 1 and _estimate_tokens(merged[0]) < _CHUNK_MIN_TOKENS:
            merged[1] = f"{merged[0]} {merged[1]}".strip()
            return merged[1:]
        return merged

    def _tail_for_overlap(self, sentences: list[str]) -> list[str]:
        """Return a tail sentence window with approx overlap token budget."""
        if _CHUNK_OVERLAP_TOKENS <= 0 or not sentences:
            return []
        tail: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            tokens = _estimate_tokens(sentence)
            if tail and total + tokens > _CHUNK_OVERLAP_TOKENS:
                break
            tail.insert(0, sentence)
            total += tokens
        return tail

    def _extract_mention_titles(self, wikilinks) -> list[str]:
        """Extract normalized Wikipedia article titles from section wikilinks."""
        titles: list[str] = []
        seen: set[str] = set()
        for wikilink in wikilinks or []:
            raw_title = getattr(wikilink, "title", None)
            if raw_title is None:
                continue
            title = self._normalize_wikilink_title(str(raw_title))
            if title is None or title in seen:
                continue
            seen.add(title)
            titles.append(title)
        return titles

    def _normalize_wikilink_title(self, raw_title: str) -> str | None:
        title = raw_title.strip().replace("_", " ")
        if not title:
            return None
        if title.startswith(":"):
            title = title[1:].strip()
        title = title.split("#", 1)[0].strip()
        title = " ".join(title.split())
        if not title:
            return None
        lowered = title.casefold()
        if any(lowered.startswith(prefix) for prefix in _SKIP_WIKILINK_NAMESPACES):
            return None
        return title

    def _is_low_signal_section(self, section_title: str) -> bool:
        return section_title.strip().casefold() in _LOW_SIGNAL_SECTION_TITLES

    def _embedding_text(self, chunk: PassageChunk) -> str:
        lines = [f"Article: {chunk.article_title}"]
        if chunk.section_title.casefold() != "lead":
            lines.append(f"Section: {chunk.section_title}")
        lines.append(chunk.text)
        return "\n".join(lines)

    async def _resolve_mention_title_map(self, batch: list[PassageChunk]) -> dict[str, str]:
        titles: list[str] = []
        seen: set[str] = set()
        for chunk in batch:
            for title in chunk.mention_titles:
                if title in seen:
                    continue
                seen.add(title)
                titles.append(title)
        if not titles:
            return {}

        rows = await self._graph.run_cypher(_RESOLVE_MENTION_TITLES_CYPHER, {"titles": titles})
        resolved: dict[str, str] = {}
        for row in rows or []:
            if isinstance(row, dict):
                title = row.get("title")
                qid = row.get("qid")
            else:
                title = row[0]
                qid = row[1]
            if isinstance(title, str) and isinstance(qid, str):
                resolved.setdefault(title, qid)
        return resolved

    def _mention_qids_for_chunk(
        self, chunk: PassageChunk, mention_title_map: dict[str, str]
    ) -> list[str]:
        qids: list[str] = []
        seen: set[str] = set()
        for title in chunk.mention_titles:
            qid = mention_title_map.get(title)
            if qid is None or qid == chunk.article_qid or qid in seen:
                continue
            seen.add(qid)
            qids.append(qid)
        return qids

    # ------------------------------------------------------------------
    # Flush: Aura + Qdrant in one batched write
    # ------------------------------------------------------------------

    async def _flush_batch(self, batch: list[PassageChunk]) -> bool:
        try:
            mention_title_map = await self._resolve_mention_title_map(batch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 2 mention resolution failed: %s", exc)
            for c in batch:
                self._dlq.write(
                    pass_name="pass2",
                    record_id=c.passage_id,
                    error_class=type(exc).__name__,
                    error_detail=str(exc),
                )
            self._ckpt.increment_dlq("pass2", n=len(batch))
            return False

        # Embed.
        texts = [self._embedding_text(c) for c in batch]
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
            return False

        # Aura: passage nodes + edges.
        aura_rows = [
            {
                "passage_id": c.passage_id,
                "text": c.text,
                "article_title": c.article_title,
                "article_qid": c.article_qid,
                "section_title": c.section_title,
                "section_ordinal": c.section_ordinal,
                "chunk_ordinal": c.chunk_ordinal,
                "token_estimate": c.token_estimate,
                "text_hash": c.text_hash,
                "mention_qids": self._mention_qids_for_chunk(c, mention_title_map),
            }
            for c in batch
        ]
        ok_aura = await retry_record(
            fn=lambda r=aura_rows: self._graph.run_cypher(_UPSERT_PASSAGE_CYPHER, {"batch": r}),
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
            return False

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


def _passage_id(qid: str, section: str, section_ord: int, chunk_idx: int) -> str:
    """Deterministic passage id: short SHA-256 of (qid, section, ord, idx)."""
    key = f"{qid}::{section}::{section_ord}::{chunk_idx}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]


def _estimate_tokens(text: str) -> int:
    """Cheap English token estimate used for chunk budgeting."""
    if not text:
        return 0
    return max(1, int(round(len(text) / _CHARS_PER_TOKEN)))


__all__ = ["Pass2PassageWriter", "PassageChunk"]
