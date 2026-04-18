"""Claim-claim NLI dispatcher — Wave 3 Stream P, Decision H.

Orchestrates the primary ``NliOpenAIGpt54Adapter`` and fallback
``NliGeminiAdapter`` for the claim-claim NLI channel, applying:

* **Entity-overlap short-circuit** — only pairs sharing
  ≥ ``entity_overlap_threshold`` resolved entity IDs survive. Bounds
  the O(n²) pair cost for large claim-count documents. Entity IDs
  come from the decomposition stage (``Claim.entity_qids``), which in
  Wave 3 is populated by the corpus-ingestion entity resolver
  (Decision K — Aura entity vector index); pre-corpus it's an empty
  dict, which means *every* pair passes the overlap filter (the
  dispatcher uses the conservative "no overlap data → assume
  overlap" policy so Stream P is not blocked by Stream C ingestion
  completing).
* **Hard cap** (``claim_claim_max_pairs``) — even with the overlap
  filter, a malicious long document with many shared entities could
  fan out OpenAI cost. We truncate deterministically (first N pairs
  in ``(claim_a, claim_b)`` sorted order) and log the truncation so
  it's visible in verdict observability.
* **Primary → fallback failover** — when the primary's terminal-
  failure sentinel fires (``reasoning="nli_unavailable"``,
  ``confidence=0.0``), re-run that specific pair through the fallback
  adapter exactly once. Count fallback firings in the returned
  metrics so ``PCGObservability.cc_nli_fallback_fired_count`` can be
  set on each claim verdict.

The dispatcher is async-concurrent — pairs run under a
``Semaphore(max_concurrency)`` so we don't serialise K OpenAI calls
(GPT-5.4 xhigh latency is high; serialising would blow past the
Lambda timeout for graphs with many edges).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from interfaces.nli import NliResult
from models.nli import NLIDistribution

if TYPE_CHECKING:
    from interfaces.nli import NliAdapter
    from models.entities import Claim

logger = logging.getLogger(__name__)


_NLI_UNAVAILABLE_REASON = "nli_unavailable"


@dataclass(frozen=True)
class DispatchResult:
    """Per-verify batch output from :class:`ClaimClaimNliDispatcher`.

    * ``distributions`` — keyed by ``(claim_a.id, claim_b.id)`` with
      ``claim_a.id < claim_b.id`` in lex order. Empty for pairs the
      overlap filter dropped (nothing to classify → no edge).
    * ``fallback_fired_count`` — number of pairs that fell through
      primary → fallback. Surfaces cost + drift concerns on the
      verdict JSON.
    * ``truncated_pair_count`` — pairs dropped by the hard cap. Zero
      under normal load; non-zero indicates a large-document or
      entity-overlap saturation situation worth flagging.
    """

    distributions: dict[tuple[UUID, UUID], NLIDistribution]
    fallback_fired_count: int
    truncated_pair_count: int


class ClaimClaimNliDispatcher:
    """Wired once at DI time with both adapters, reused per verify."""

    def __init__(
        self,
        *,
        primary: "NliAdapter",
        fallback: "NliAdapter",
        entity_overlap_threshold: int = 1,
        claim_claim_max_pairs: int = 200,
        max_concurrency: int = 10,
    ) -> None:
        if entity_overlap_threshold < 0:
            raise ValueError(
                f"entity_overlap_threshold must be >= 0, got {entity_overlap_threshold}"
            )
        if claim_claim_max_pairs <= 0:
            raise ValueError(
                f"claim_claim_max_pairs must be > 0, got {claim_claim_max_pairs}"
            )
        self._primary = primary
        self._fallback = fallback
        self._overlap_threshold = entity_overlap_threshold
        self._max_pairs = claim_claim_max_pairs
        self._sem = asyncio.Semaphore(max_concurrency)

    async def classify_pairs(
        self, claims: list["Claim"]
    ) -> DispatchResult:
        """Build the pair set, fan out primary calls, fall back on
        sentinels, return the dispatch result."""
        if len(claims) < 2:
            return DispatchResult(distributions={}, fallback_fired_count=0, truncated_pair_count=0)

        pairs = self._build_pair_set(claims)
        truncated = 0
        if len(pairs) > self._max_pairs:
            truncated = len(pairs) - self._max_pairs
            logger.warning(
                "cc-NLI pair hard-cap hit: %d pairs truncated to %d (entity-overlap insufficient "
                "to bound; consider raising CC_NLI threshold or CC_NLI_CLAIM_CLAIM_MAX_PAIRS)",
                len(pairs),
                self._max_pairs,
            )
            pairs = pairs[: self._max_pairs]

        # Primary pass — concurrent fan-out.
        async def _primary_task(
            ca: "Claim", cb: "Claim"
        ) -> tuple[UUID, UUID, NliResult]:
            async with self._sem:
                result = await self._primary.classify(ca.text, cb.text)
            return ca.id, cb.id, result

        primary_results = await asyncio.gather(
            *[_primary_task(ca, cb) for ca, cb in pairs]
        )

        # Fallback pass — only for primary sentinels. One shot, no retries.
        fallback_fired = 0
        final_results: dict[tuple[UUID, UUID], NliResult] = {}
        fallback_pairs: list[tuple[UUID, UUID, "Claim", "Claim"]] = []
        claim_by_id = {c.id: c for c in claims}
        for id_a, id_b, result in primary_results:
            if _is_sentinel(result):
                ca = claim_by_id[id_a]
                cb = claim_by_id[id_b]
                fallback_pairs.append((id_a, id_b, ca, cb))
            else:
                final_results[(id_a, id_b)] = result

        async def _fallback_task(
            id_a: UUID, id_b: UUID, ca: "Claim", cb: "Claim"
        ) -> tuple[UUID, UUID, NliResult]:
            async with self._sem:
                result = await self._fallback.classify(ca.text, cb.text)
            return id_a, id_b, result

        if fallback_pairs:
            fb_results = await asyncio.gather(
                *[_fallback_task(*fp) for fp in fallback_pairs]
            )
            for id_a, id_b, result in fb_results:
                fallback_fired += 1
                final_results[(id_a, id_b)] = result
            logger.info(
                "cc_nli_fallback_fired for %d/%d pairs",
                fallback_fired,
                len(pairs),
            )

        distributions = {
            key: _nli_result_to_distribution(result, nli_model_id="cc-nli-dispatcher")
            for key, result in final_results.items()
        }
        return DispatchResult(
            distributions=distributions,
            fallback_fired_count=fallback_fired,
            truncated_pair_count=truncated,
        )

    def _build_pair_set(
        self, claims: list["Claim"]
    ) -> list[tuple["Claim", "Claim"]]:
        """Generate all unique pairs with ``id_a < id_b`` that pass
        the entity-overlap short-circuit.

        When both claims have empty ``entity_qids``, the pair passes
        (conservative pre-corpus policy). When at least one has
        non-empty qids, the intersection must have ≥ threshold
        elements.
        """
        pairs: list[tuple["Claim", "Claim"]] = []
        n = len(claims)
        for i in range(n):
            for j in range(i + 1, n):
                ca, cb = claims[i], claims[j]
                # Canonical ordering (lexical by UUID) matches the
                # NLIService.claim_claim port contract.
                if ca.id > cb.id:
                    ca, cb = cb, ca
                if self._pair_passes_overlap(ca, cb):
                    pairs.append((ca, cb))
        # Deterministic ordering for the hard cap — sort by pair UUIDs.
        pairs.sort(key=lambda pair: (str(pair[0].id), str(pair[1].id)))
        return pairs

    def _pair_passes_overlap(
        self, ca: "Claim", cb: "Claim"
    ) -> bool:
        """Entity-overlap check. Threshold is inclusive."""
        qa = set(getattr(ca, "entity_qids", {}).values() if ca.entity_qids else [])
        qb = set(getattr(cb, "entity_qids", {}).values() if cb.entity_qids else [])
        if not qa or not qb:
            # No qid data on at least one side → pre-corpus regime.
            # Pass the pair conservatively; the hard cap is the final
            # backstop. Once Stream C lands and populates entity_qids,
            # this case will be rare.
            return True
        return len(qa & qb) >= self._overlap_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_sentinel(result: NliResult) -> bool:
    return result.reasoning == _NLI_UNAVAILABLE_REASON and result.confidence == 0.0


def _nli_result_to_distribution(
    result: NliResult,
    *,
    nli_model_id: str,
) -> NLIDistribution:
    """Convert an ``NliResult`` (support/refute/neutral scores, from
    LLM adapter) into an ``NLIDistribution`` (entail/contradict/neutral,
    consumed by the PCG layer).

    The field rename is deliberate: the LLM port reuses support/refute
    language because it originated in D1's claim-evidence channel,
    while the PCG layer speaks classical NLI vocabulary (entail /
    contradict). Semantics are identical — entail = support,
    contradict = refute.

    Variance is zero for single-shot (K=1) adapters; a future
    self-consistency wrapper could populate it.
    """
    total = result.supporting_score + result.refuting_score + result.neutral_score
    if total <= 0:
        # Degenerate — use uniform. NLIDistribution's model validator
        # requires sum ≈ 1.
        return NLIDistribution(
            entail=1.0 / 3.0,
            contradict=1.0 / 3.0,
            neutral=1.0 / 3.0,
            variance=0.0,
            nli_model_id=nli_model_id,
        )
    return NLIDistribution(
        entail=result.supporting_score / total,
        contradict=result.refuting_score / total,
        neutral=result.neutral_score / total,
        variance=0.0,
        nli_model_id=nli_model_id,
    )


__all__ = [
    "ClaimClaimNliDispatcher",
    "DispatchResult",
]
