"""Wave 3 Stream C — runtime entity resolver.

Maps claim subjects/objects to Wikidata QIDs at verify time. Used by
the ClaimClaimNliDispatcher's entity-overlap short-circuit
(:mod:`adapters.nli_claim_claim`) and consumed transitively by the PCG
factor graph construction (:mod:`adapters._pcg_graph`).

Resolution strategy (stage order):

1. **Exact-label lookup** via Cypher:
   ``MATCH (e:Entity {label_lower: $t}) RETURN e.qid``. Fast,
   unambiguous when the claim text literally matches a Wikidata label.
2. **Vector lookup** on the Aura entity vector index (Decision K —
   no Qdrant hop). Embeds the claim text, runs
   ``CALL db.index.vector.queryNodes('entity_embeddings', $k, $vec)``,
   returns top-K QIDs above the cosine-similarity threshold.

Two stages so the cheap exact-match short-circuits before the embed
call; saves pc-embed round-trips on claims with well-known subjects.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
        "did", "do", "does", "has", "have", "had", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "and", "or", "but",
    }
)


@dataclass(frozen=True)
class ResolvedEntity:
    qid: str
    label: str
    similarity: float
    source: str  # 'exact' | 'vector'


_EXACT_CYPHER = """
MATCH (e:Entity)
WHERE toLower(e.label) IN $labels
RETURN DISTINCT e.qid AS qid, e.label AS label
"""


_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes('entity_embeddings', $k, $vec) YIELD node, score
WHERE score >= $threshold
RETURN node.qid AS qid, node.label AS label, score AS similarity
"""


class EntityResolver:
    """Resolve free-text claim spans to QIDs via exact + vector hops."""

    def __init__(
        self,
        *,
        graph_store,
        embedding_adapter,
        vector_top_k: int = 5,
        similarity_threshold: float = 0.55,
    ) -> None:
        self._graph = graph_store
        self._embed = embedding_adapter
        self._top_k = vector_top_k
        self._threshold = similarity_threshold

    async def resolve(self, claim_text: str) -> list[ResolvedEntity]:
        """Resolve candidate QIDs for a claim. Returns ordered by
        similarity; an empty list means no confident resolution."""
        if not claim_text or not claim_text.strip():
            return []
        # Stage 1: exact-label lookup.
        candidates = [t for t in _extract_candidate_spans(claim_text)]
        exact = await self._exact_lookup(candidates) if candidates else []
        if exact:
            return exact

        # Stage 2: vector lookup on the full claim text.
        try:
            vec = await self._embed.generate_embedding(claim_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("entity_resolver: embed failed on %r: %s", claim_text[:80], exc)
            return []
        return await self._vector_lookup(vec)

    async def _exact_lookup(self, spans: list[str]) -> list[ResolvedEntity]:
        labels = [s.lower() for s in spans if s]
        if not labels:
            return []
        rows = await self._graph.run_cypher(_EXACT_CYPHER, {"labels": labels})
        out: list[ResolvedEntity] = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append(
                    ResolvedEntity(
                        qid=str(r["qid"]),
                        label=str(r.get("label") or ""),
                        similarity=1.0,
                        source="exact",
                    )
                )
        return out

    async def _vector_lookup(self, vec) -> list[ResolvedEntity]:
        try:
            rows = await self._graph.run_cypher(
                _VECTOR_CYPHER,
                {"vec": list(vec), "k": self._top_k, "threshold": self._threshold},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("entity_resolver: vector index query failed: %s", exc)
            return []
        out: list[ResolvedEntity] = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append(
                    ResolvedEntity(
                        qid=str(r["qid"]),
                        label=str(r.get("label") or ""),
                        similarity=float(r.get("similarity") or 0.0),
                        source="vector",
                    )
                )
        out.sort(key=lambda x: x.similarity, reverse=True)
        return out


def _extract_candidate_spans(text: str) -> list[str]:
    """Pull candidate entity-name spans from a claim.

    Strategy (cheap): capitalised token n-grams, skipping stopwords.
    Precise NER is a Wave-4 enhancement; for the cc-NLI short-circuit
    we just need something that fires on obvious Named Entity
    subjects like "Marie Curie" or "Einstein".
    """
    # Strip punctuation that might split a name across tokens.
    cleaned = re.sub(r"[\".,;:!?\(\)\[\]]", " ", text)
    tokens = cleaned.split()
    spans: list[str] = []
    current: list[str] = []
    for t in tokens:
        if t and t[0].isupper() and t.lower() not in _STOPWORDS:
            current.append(t)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return spans


__all__ = ["EntityResolver", "ResolvedEntity"]
