"""Gemini 3 Pro NLI adapter (Phase 2 Task 2.1 / plan §4.2).

Composes over an existing :class:`~interfaces.llm.LLMProvider` — typically
the native :class:`~adapters.gemini.GeminiLLMAdapter` running Gemini 3 Pro
with ``thinkingLevel=HIGH`` and ``safetySettings=BLOCK_NONE`` — and layers
three NLI-specific concerns on top:

1. **Prompt** — the strict-JSON NLI prompt from plan §4.2.
2. **Retries + fallback** — transport and JSON-decode errors are retried
   up to ``max_retries`` times; a terminal failure returns a neutral
   ``NliResult`` rather than bubbling up (the pipeline's posterior update
   must keep going even when one evidence passage can't be classified).
3. **Self-consistency** — when ``self_consistency_k > 1``, the adapter
   draws K samples at ``temperature=0.2``, majority-votes the label, and
   averages scores within the winning label before renormalising.

All three concerns are deliberately kept in this module (not pushed
upstream into :class:`LLMProvider`) because they are NLI-specific: neither
the decomposer nor future embedding callers want silent fallbacks or
majority voting at the LLM layer.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import orjson

from interfaces.llm import LLMMessage
from interfaces.nli import NliLabel, NliResult

if TYPE_CHECKING:
    from interfaces.llm import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt — copied verbatim from docs/superpowers/plans/2026-04-18-phase2-
# orchestration.md §4.2. Any change here must be paired with an update to
# that plan section; the prompt is part of the contract Fabian approved.
# ---------------------------------------------------------------------------

_NLI_PROMPT_TEMPLATE = """You are a fact-checking NLI model. Given a claim and an evidence
passage, classify: does the evidence SUPPORT, REFUTE, or stay NEUTRAL
on the claim? Return strict JSON:
  {{
    "label": "support" | "refute" | "neutral",
    "supporting_score": 0.0-1.0,
    "refuting_score":   0.0-1.0,
    "neutral_score":    0.0-1.0,   (all three sum to 1.0)
    "reasoning": "<one sentence, the key fact>",
    "confidence": 0.0-1.0
  }}
Claim: "{claim_text}"
Evidence: "{evidence_text}"
"""


_VALID_LABELS: frozenset[str] = frozenset({"support", "refute", "neutral"})

# Sentinel returned when every retry is exhausted — the pipeline's posterior
# update must keep going even when one evidence passage can't be classified.
_NLI_UNAVAILABLE_REASON = "nli_unavailable"

_NEUTRAL_FALLBACK = NliResult(
    label="neutral",
    supporting_score=0.0,
    refuting_score=0.0,
    neutral_score=1.0,
    reasoning=_NLI_UNAVAILABLE_REASON,
    confidence=0.0,
)


def _is_fallback(result: NliResult) -> bool:
    return (
        result.reasoning == _NLI_UNAVAILABLE_REASON
        and result.confidence == 0.0
    )


class NliGeminiAdapter:
    """LLM-backed NLI classifier. Satisfies :class:`~interfaces.nli.NliAdapter`."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        self_consistency_k: int = 1,
        max_retries: int = 3,
    ) -> None:
        if self_consistency_k < 1:
            raise ValueError(
                f"self_consistency_k must be >= 1, got {self_consistency_k}"
            )
        if max_retries < 1:
            raise ValueError(
                f"max_retries must be >= 1, got {max_retries}"
            )
        self._llm = llm
        self._k = self_consistency_k
        self._max_retries = max_retries

    async def classify(
        self, claim_text: str, evidence_text: str
    ) -> NliResult:
        prompt = _NLI_PROMPT_TEMPLATE.format(
            claim_text=claim_text, evidence_text=evidence_text
        )
        messages = [LLMMessage(role="user", content=prompt)]

        if self._k == 1:
            return await self._single_pass_with_retry(messages)

        samples = [
            await self._single_pass_with_retry(messages) for _ in range(self._k)
        ]
        successes = [s for s in samples if not _is_fallback(s)]
        if not successes:
            return _NEUTRAL_FALLBACK
        return _aggregate_samples(successes)

    async def _single_pass_with_retry(
        self, messages: list[LLMMessage]
    ) -> NliResult:
        """One NLI classification with up to ``max_retries`` attempts.

        Catches every exception — transport, JSON parse, validation —
        because the pipeline must not crash on a flaky LLM. The terminal
        failure returns the ``_NEUTRAL_FALLBACK`` sentinel.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._llm.complete(
                    messages,
                    temperature=0.2,
                    json_mode=True,
                )
                return _parse_result(response.content)
            except Exception as exc:  # noqa: BLE001 — we really do want "everything"
                last_exc = exc
                logger.warning(
                    "NLI attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
        logger.error(
            "NLI terminal failure after %d retries (last error: %s) — "
            "returning neutral fallback",
            self._max_retries,
            last_exc,
        )
        return _NEUTRAL_FALLBACK

    async def health_check(self) -> bool:
        return await self._llm.health_check()



# ---------------------------------------------------------------------------
# Parsing + score hygiene
# ---------------------------------------------------------------------------


def _parse_result(raw: str) -> NliResult:
    """Turn the LLM's JSON text into an ``NliResult`` with renormalised scores.

    Raises ``ValueError`` on any structural issue (unknown label, non-
    numeric score, missing key); the caller is responsible for catching
    and deciding retry vs fallback.
    """
    data = orjson.loads(raw)
    label = data["label"]
    if label not in _VALID_LABELS:
        raise ValueError(f"NLI JSON has unknown label {label!r}")
    supporting = float(data["supporting_score"])
    refuting = float(data["refuting_score"])
    neutral = float(data["neutral_score"])
    total = supporting + refuting + neutral
    if total <= 0:
        raise ValueError(
            f"NLI JSON scores sum to non-positive value {total!r}"
        )
    supporting /= total
    refuting /= total
    neutral /= total
    return NliResult(
        label=label,  # type: ignore[arg-type]  # narrowed by membership check
        supporting_score=supporting,
        refuting_score=refuting,
        neutral_score=neutral,
        reasoning=str(data.get("reasoning", "")),
        confidence=float(data.get("confidence", 0.0)),
    )


def _aggregate_samples(samples: list[NliResult]) -> NliResult:
    """Majority-vote label; average scores + confidence within the
    winning bucket; renormalise so the 3 scores sum to 1.

    Ties fall back to the ``Counter.most_common`` ordering, which
    preserves first-seen on equal counts — deterministic given stable
    sample order. In practice ties are rare at K=3/5 with a calibrated
    NLI model and the pipeline just consumes whichever label wins.
    """
    label_counts = Counter(s.label for s in samples)
    winner_label: NliLabel = label_counts.most_common(1)[0][0]
    winners = [s for s in samples if s.label == winner_label]
    n = len(winners)
    avg_support = sum(s.supporting_score for s in winners) / n
    avg_refute = sum(s.refuting_score for s in winners) / n
    avg_neutral = sum(s.neutral_score for s in winners) / n
    total = avg_support + avg_refute + avg_neutral
    if total > 0:
        avg_support /= total
        avg_refute /= total
        avg_neutral /= total
    avg_confidence = sum(s.confidence for s in winners) / n
    # Surface the reasoning from the most confident winner — still
    # grounded in the winning label, unlike a free pick across buckets.
    best_winner = max(winners, key=lambda s: s.confidence)
    return NliResult(
        label=winner_label,
        supporting_score=avg_support,
        refuting_score=avg_refute,
        neutral_score=avg_neutral,
        reasoning=best_winner.reasoning,
        confidence=avg_confidence,
    )


__all__ = ["NliGeminiAdapter"]
