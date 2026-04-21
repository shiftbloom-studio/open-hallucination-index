"""Wave 3 Stream C — graph retriever orchestrating the
Qdrant → Aura → rerank path (Decision K).

Flow:

1. :class:`QdrantPassageSearch` returns top-K passage_ids by
   vector similarity on the query text.
2. :class:`AuraPassageFetch` fetches the passages' text + article
   context in a single batch ``MATCH``.

3. Optional Bedrock reranking reorders the fetched passages by
   query relevance before returning final top-K evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from adapters.bedrock_rerank import BedrockRerankAdapter, RerankScore
from models.entities import Evidence, EvidenceSource
from pipeline.retrieval.aura_passage_fetch import AuraPassageFetch
from pipeline.retrieval.qdrant_passage_search import (
    PassageHit,
    QdrantPassageSearch,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphRetriever:
    """Compose Qdrant search + Aura fetch into a flat ``list[Evidence]``."""

    qdrant: QdrantPassageSearch
    aura: AuraPassageFetch
    reranker: BedrockRerankAdapter | None = None
    candidate_top_k: int = 40
    final_top_k: int = 12

    async def retrieve(
        self, query_text: str, *, top_k: int | None = None
    ) -> list[Evidence]:
        final_top_k = top_k or self.final_top_k
        candidate_top_k = max(final_top_k, self.candidate_top_k)

        hits = await self.qdrant.search(query_text, top_k=candidate_top_k)
        if not hits:
            return []
        by_id = {h.passage_id: h for h in hits}
        fetched = await self.aura.fetch_many([h.passage_id for h in hits])
        if not fetched:
            return []

        rerank_scores: dict[int, float] = {}
        ordered_indices = list(range(len(fetched)))
        if self.reranker is not None and self.reranker.enabled:
            try:
                scores = await self.reranker.rerank(
                    query=query_text,
                    documents=[p.text for p in fetched],
                    top_k=final_top_k,
                )
                rerank_scores = {row.index: row.relevance_score for row in scores}
                ordered_indices = [
                    row.index
                    for row in sorted(
                        scores,
                        key=lambda row: row.relevance_score,
                        reverse=True,
                    )
                    if 0 <= row.index < len(fetched)
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("GraphRetriever rerank failed, fallback to ANN order: %s", exc)

        if not rerank_scores:
            ordered_indices = sorted(
                range(len(fetched)),
                key=lambda idx: by_id.get(fetched[idx].passage_id, PassageHit("", "", 0.0)).similarity,
                reverse=True,
            )

        evidence: list[Evidence] = []
        for idx in ordered_indices[:final_top_k]:
            p = fetched[idx]
            hit = by_id.get(p.passage_id)
            similarity = hit.similarity if hit else 0.0
            wiki_title = p.article_title.replace(" ", "_") if p.article_title else ""
            evidence.append(
                Evidence(
                    source=EvidenceSource.VECTOR_SEMANTIC,
                    content=p.text,
                    source_uri=f"https://en.wikipedia.org/wiki/{wiki_title}" if wiki_title else None,
                    similarity_score=similarity,
                    structured_data={
                        "article_title": p.article_title,
                        "article_qid": p.article_qid,
                        "section_title": p.section_title,
                        "section_ordinal": p.section_ordinal,
                        "chunk_ordinal": p.chunk_ordinal,
                        "token_estimate": p.token_estimate,
                        "passage_id": p.passage_id,
                        "ann_score": similarity,
                        "rerank_score": rerank_scores.get(idx),
                    },
                )
            )
        return evidence


__all__ = ["GraphRetriever"]
