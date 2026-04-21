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
# Prompt — v2 decisive variant (post Einstein-Russia tuning). The v1 prompt
# was deliberately symmetric and terse; Gemini 3 Pro hedged to "neutral"
# whenever the passage didn't literally quote the claim, which let
# obviously-refutable claims ("Einstein was born in Russia") settle at
# p_true ≈ 0.33 — too soft, and close to "uncertain" when the truth value
# is not uncertain at all. v2 forces a decisive 3-way choice: passages
# that provide a *contradicting fact* are REFUTE, not NEUTRAL; NEUTRAL is
# reserved for genuinely off-topic text.
#
# Paired with a pipeline-side change: ``label="neutral"`` results are now
# skipped from the Beta-posterior fold (``_beta_update_from_nli`` in
# pipeline.py). The prompt and the fold rule must stay in lockstep — if
# one is loosened, off-topic tilt starts leaking into the posterior again.
# ---------------------------------------------------------------------------

_NLI_PROMPT_TEMPLATE = """You are a strict fact-checking NLI classifier. Given a CLAIM and an
EVIDENCE passage, decide whether the evidence SUPPORTS, REFUTES, or is
truly NEUTRAL with respect to the claim.

Decision rules (apply in order):

1. SUPPORT — the evidence confirms the claim's central assertion,
   either by explicitly stating it or by providing facts from which a
   reasonable reader would conclude the claim is true. Use SUPPORT
   whenever the evidence is on-topic and consistent with the claim.

2. REFUTE — the evidence contradicts the claim, EITHER directly ("X
   did not do Y") OR by providing facts that are **mutually exclusive**
   with the claim. Example: evidence "Albert Einstein was born in Ulm,
   Germany in 1879" REFUTES the claim "Einstein was born in Russia",
   because a person is born in exactly one country. Do NOT downgrade
   such cases to NEUTRAL; providing the correct answer to a factual
   question that the claim gets wrong IS a refutation.
   If a claim assigns a specific invention/discovery/creation to a
   person, and the evidence establishes a clearly different core role
   or biography for that person without support for that invention,
   classify as REFUTE (not NEUTRAL).

3. NEUTRAL — reserved ONLY for passages that do not address the claim's
   subject-predicate-object at all. Example: "Einstein won the Nobel
   Prize in Physics in 1921" is NEUTRAL with respect to the claim
   "Einstein was born in Russia" — the Nobel fact is on a different
   predicate and neither confirms nor contradicts the birthplace claim.
   NEUTRAL is NOT for "I am uncertain" — if you can infer SUPPORT or
   REFUTE from the evidence, you MUST pick one of those.

Score guidance:

- For SUPPORT, ``supporting_score`` should be ≥ 0.70 when the evidence
  clearly confirms the claim; reserve lower support scores for weak or
  partial confirmation.
- For REFUTE, ``refuting_score`` should be ≥ 0.70 when the evidence
  clearly contradicts the claim; reserve lower refute scores for
  partial contradiction.
- For NEUTRAL, ``neutral_score`` should be ≥ 0.70. Off-topic evidence
  has no meaningful support or refute mass; keep ``supporting_score``
  and ``refuting_score`` low (≤ 0.15 each).
- All three scores must sum to 1.0.

Return strict JSON, no prose outside the object:
  {{
    "label": "support" | "refute" | "neutral",
    "supporting_score": 0.0-1.0,
    "refuting_score":   0.0-1.0,
    "neutral_score":    0.0-1.0,
    "reasoning": "<one sentence, the single key fact from the evidence>",
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


# Any averaged self-consistency confidence below this is snapped to 0.0
# so D1's posterior update gets a clean "no usable signal" marker. Picked
# small enough that merely-mediocre results (0.3–0.6) pass through but a
# run of tied, very-unsure samples gets flagged.
_LOW_CONFIDENCE_FLOOR = 0.05


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
    if avg_confidence < _LOW_CONFIDENCE_FLOOR:
        avg_confidence = 0.0
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
