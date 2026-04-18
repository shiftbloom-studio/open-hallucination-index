"""Wave 3 Stream C — Aura passage fetch (Decision K).

After :class:`QdrantPassageSearch` returns passage IDs, this fetcher
runs one Cypher ``MATCH`` to pull the full passage text + article
context out of Aura. Batched to minimise Bolt-channel round-trips.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassageFetchResult:
    passage_id: str
    text: str
    article_title: str
    article_qid: str
    section_title: str


_BATCH_FETCH_CYPHER = """
UNWIND $ids AS pid
MATCH (p:Passage {id: pid})
RETURN p.id AS id, p.text AS text, p.article_title AS article_title,
       p.article_qid AS article_qid, p.section_title AS section_title
"""


class AuraPassageFetch:
    """Batched ``MATCH`` for passage text + metadata by id."""

    def __init__(self, *, graph_store) -> None:
        self._graph = graph_store

    async def fetch_many(self, passage_ids: list[str]) -> list[PassageFetchResult]:
        if not passage_ids:
            return []
        try:
            rows = await self._graph.run_cypher(
                _BATCH_FETCH_CYPHER, {"ids": passage_ids}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AuraPassageFetch batch failed: %s", exc)
            return []
        out: list[PassageFetchResult] = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append(
                    PassageFetchResult(
                        passage_id=str(r["id"]),
                        text=str(r.get("text") or ""),
                        article_title=str(r.get("article_title") or ""),
                        article_qid=str(r.get("article_qid") or ""),
                        section_title=str(r.get("section_title") or ""),
                    )
                )
        # Preserve input order (Aura doesn't guarantee UNWIND ordering).
        order = {pid: i for i, pid in enumerate(passage_ids)}
        out.sort(key=lambda p: order.get(p.passage_id, 10**9))
        return out


__all__ = ["AuraPassageFetch", "PassageFetchResult"]
