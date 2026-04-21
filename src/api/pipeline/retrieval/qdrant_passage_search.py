"""Wave 3 Stream C — vector-only Qdrant passage search (Decision K).

Qdrant stores payload ``{passage_id, qid}`` only. Search returns
passage IDs; callers hop to Aura to fetch text + article context.

Thin wrapper around :class:`~adapters.qdrant.QdrantVectorAdapter`
so the retrieval layer has a stable interface even when the
underlying Qdrant client evolves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.embeddings import LocalEmbeddingAdapter
    from adapters.qdrant import QdrantVectorAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassageHit:
    passage_id: str
    qid: str
    similarity: float


class QdrantPassageSearch:
    """Top-K passage id lookup by query-vector similarity."""

    def __init__(
        self,
        *,
        vector_store: "QdrantVectorAdapter",
        embedding_adapter: "LocalEmbeddingAdapter",
        collection_name: str = "ohi_passages_titan1024",
        score_threshold: float = 0.4,
    ) -> None:
        self._vector = vector_store
        self._embed = embedding_adapter
        self._collection = collection_name
        self._score_threshold = score_threshold

    async def search(
        self, query_text: str, *, top_k: int = 10
    ) -> list[PassageHit]:
        try:
            vec = await self._embed.generate_embedding(query_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QdrantPassageSearch embed failed: %s", exc)
            return []
        try:
            results = await self._vector.search_passages(
                vector=list(vec),
                top_k=top_k,
                collection_name=self._collection,
                score_threshold=self._score_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("QdrantPassageSearch search failed: %s", exc)
            return []
        hits: list[PassageHit] = []
        for r in results or []:
            payload = r.get("payload") or {}
            pid = payload.get("passage_id")
            qid = payload.get("qid")
            if not pid:
                continue
            hits.append(
                PassageHit(
                    passage_id=str(pid),
                    qid=str(qid) if qid else "",
                    similarity=float(r.get("score", 0.0)),
                )
            )
        return hits


__all__ = ["QdrantPassageSearch", "PassageHit"]
