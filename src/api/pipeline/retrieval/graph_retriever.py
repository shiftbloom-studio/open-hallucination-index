"""Wave 3 Stream C — 2-step graph retriever orchestrating the
Qdrant → Aura hop (Decision K).

Flow:

1. :class:`QdrantPassageSearch` returns top-K passage_ids by
   vector similarity on the query text.
2. :class:`AuraPassageFetch` fetches the passages' text + article
   context in a single batch ``MATCH``.

Returned :class:`Evidence` objects match the existing retrieval
contract so the pipeline's claim-evidence NLI path consumes them
unchanged. The collector (:class:`AdaptiveEvidenceCollector`) will
pick this retriever up through DI once Pass 2 has produced enough
passages to beat MediaWiki MCP on recall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

    async def retrieve(
        self, query_text: str, *, top_k: int = 10
    ) -> list[Evidence]:
        hits = await self.qdrant.search(query_text, top_k=top_k)
        if not hits:
            return []
        by_id = {h.passage_id: h for h in hits}
        fetched = await self.aura.fetch_many([h.passage_id for h in hits])
        evidence: list[Evidence] = []
        for p in fetched:
            hit = by_id.get(p.passage_id)
            similarity = hit.similarity if hit else 0.0
            evidence.append(
                Evidence(
                    source=EvidenceSource.NEO4J,
                    content=p.text,
                    source_uri=f"https://en.wikipedia.org/wiki/{p.article_title.replace(' ', '_')}",
                    similarity_score=similarity,
                    structured_data={
                        "article_title": p.article_title,
                        "article_qid": p.article_qid,
                        "section_title": p.section_title,
                        "passage_id": p.passage_id,
                    },
                )
            )
        return evidence


__all__ = ["GraphRetriever"]
