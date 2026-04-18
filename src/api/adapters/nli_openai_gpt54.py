"""OpenAI GPT-5.4 claim-claim NLI adapter (Wave 3 Stream P, Decision H).

Satisfies the :class:`~interfaces.nli.NliAdapter` port so it's
interchangeable with :class:`~adapters.nli_gemini.NliGeminiAdapter` —
the claim-claim dispatcher can wire either as primary or fallback.

Uses the **OpenAI Responses API** (``POST /v1/responses``, SDK method
``client.responses.create``) with
``reasoning={"effort": "xhigh"}`` for deep-reasoning GPT-5.4 pulls.
This is distinct from the Chat Completions API (``/v1/chat/completions``);
the Responses shape returns a single ``output_text`` aggregate rather
than a choices-array with role/content, which the adapter reads
directly.

Retry + fallback discipline mirrors ``NliGeminiAdapter``: transport /
JSON-parse errors retry up to ``max_retries``; terminal failure emits
a neutral ``_NLI_UNAVAILABLE`` sentinel (``reasoning="nli_unavailable",
confidence=0.0``) so the :class:`ClaimClaimNliDispatcher` upstream can
detect and hand off to the Gemini fallback. The pipeline's PCG build
treats the sentinel as a no-edge signal, same as the claim-evidence
path (Hebel A in the Phase 2 posterior fold).

The model + reasoning effort come from a single
``model_with_effort`` constructor argument (e.g. ``gpt-5.4-xhigh``)
parsed at init time:

* suffix ``-xhigh`` / ``-high`` / ``-medium`` / ``-low`` →
  ``reasoning.effort = "xhigh" | "high" | "medium" | "low"``
* bare model name (no suffix) → ``reasoning`` unset (uses the OpenAI
  default, typically ``none``).
"""

from __future__ import annotations

import logging
from typing import Any

import orjson

from interfaces.nli import NliResult

logger = logging.getLogger(__name__)


_NLI_PROMPT_TEMPLATE = """You are a strict fact-checking NLI classifier. Given two CLAIMS
(labelled A and B), decide whether Claim A semantically SUPPORTS,
REFUTES, or is truly NEUTRAL with respect to Claim B.

Decision rules (apply in order):

1. SUPPORT — Claim A confirms Claim B's central assertion, either by
   restating it or by providing facts from which Claim B follows.
2. REFUTE — Claim A contradicts Claim B directly, OR provides facts
   that are **mutually exclusive** with Claim B's central assertion
   (e.g. different birthplace, different year, different outcome).
   Mutually-exclusive facts are a refutation, not a neutral.
3. NEUTRAL — reserved ONLY for cases where A and B do not share a
   testable subject-predicate-object. NEUTRAL is NOT for "uncertain".

Score guidance: when SUPPORT or REFUTE is chosen, the corresponding
score should be at least 0.70. Use ``neutral_score ≥ 0.70`` only for
genuinely off-topic pairs.

Return strict JSON, no prose outside the object:
  {{
    "label": "support" | "refute" | "neutral",
    "supporting_score": 0.0-1.0,
    "refuting_score":   0.0-1.0,
    "neutral_score":    0.0-1.0,
    "reasoning": "<one sentence, the single key fact>",
    "confidence": 0.0-1.0
  }}

Claim A: "{claim_text}"
Claim B: "{evidence_text}"
"""


_VALID_LABELS: frozenset[str] = frozenset({"support", "refute", "neutral"})


_NLI_UNAVAILABLE_REASON = "nli_unavailable"


_NEUTRAL_FALLBACK = NliResult(
    label="neutral",
    supporting_score=0.0,
    refuting_score=0.0,
    neutral_score=1.0,
    reasoning=_NLI_UNAVAILABLE_REASON,
    confidence=0.0,
)


_EFFORT_SUFFIXES = ("-xhigh", "-high", "-medium", "-low")


def _parse_model_with_effort(spec: str) -> tuple[str, str | None]:
    """Split ``gpt-5.4-xhigh`` → ``("gpt-5.4", "xhigh")``.

    Returns ``(model, effort_or_none)``. Bare ``gpt-5.4`` returns
    ``("gpt-5.4", None)`` so callers can omit the ``reasoning`` kwarg
    entirely and use the OpenAI default.
    """
    for suffix in _EFFORT_SUFFIXES:
        if spec.endswith(suffix):
            return spec[: -len(suffix)], suffix.lstrip("-")
    return spec, None


class NliOpenAIGpt54Adapter:
    """OpenAI Responses API adapter for claim-claim NLI.

    Constructed once per app. Holds its own ``openai.AsyncOpenAI``
    client pinned to the API key from ``ohi/openai-api-key`` secret
    (surfaced as the ``OHI_OPENAI_API_KEY`` Lambda env var via
    ``infra/terraform/compute/lambda.tf``).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_with_effort: str = "gpt-5.4-xhigh",
        max_retries: int = 3,
        timeout_s: float = 60.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        # Import inside __init__ so the module stays import-clean even
        # when openai isn't installed (e.g. in a stripped test runner).
        from openai import AsyncOpenAI  # noqa: PLC0415

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s)
        self._model, self._effort = _parse_model_with_effort(model_with_effort)
        self._max_retries = max_retries

    async def classify(
        self, claim_text: str, evidence_text: str
    ) -> NliResult:
        """Run one NLI classification. Never raises — terminal
        failures return the ``_NEUTRAL_FALLBACK`` sentinel.

        For the cc-NLI channel, ``claim_text`` is Claim A and
        ``evidence_text`` is Claim B (port signature reused from D1).
        """
        prompt = _NLI_PROMPT_TEMPLATE.format(
            claim_text=claim_text, evidence_text=evidence_text
        )
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "input": [{"role": "user", "content": prompt}],
                }
                if self._effort is not None:
                    kwargs["reasoning"] = {"effort": self._effort}
                response = await self._client.responses.create(**kwargs)
                raw = _extract_output_text(response)
                return _parse_result(raw)
            except Exception as exc:  # noqa: BLE001 — intentional: keep pipeline alive
                last_exc = exc
                logger.warning(
                    "cc-NLI OpenAI attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
        logger.error(
            "cc-NLI OpenAI terminal failure after %d retries (last: %s) — "
            "returning neutral fallback sentinel",
            self._max_retries,
            last_exc,
        )
        return _NEUTRAL_FALLBACK

    async def health_check(self) -> bool:
        """Minimal connectivity probe — runs a tiny ``responses.create``
        call with ``max_output_tokens=1``. Returns False on any error
        rather than raising so ``/health/deep`` can render degraded."""
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "input": [{"role": "user", "content": "ping"}],
                "max_output_tokens": 16,
            }
            if self._effort is not None:
                kwargs["reasoning"] = {"effort": self._effort}
            await self._client.responses.create(**kwargs)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("cc-NLI OpenAI health_check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_output_text(response: Any) -> str:
    """Pull the aggregated text output from an OpenAI Responses object.

    SDK 2.14+ exposes ``response.output_text`` as a convenience string
    accessor that concatenates all output parts. If the attribute is
    missing (older SDK or response shape drift), fall back to walking
    ``response.output`` and joining text items.
    """
    text = getattr(response, "output_text", None)
    if text is not None:
        return str(text)
    # Fallback walker — keeps us resilient if SDK shape shifts.
    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for c in content:
            t = getattr(c, "text", None)
            if t is not None:
                parts.append(str(t))
    return "".join(parts)


def _parse_result(raw: str) -> NliResult:
    """Parse the strict-JSON response into an ``NliResult``.

    Raises ``ValueError`` on structural issues so the retry loop can
    catch and re-attempt.
    """
    # Strip accidental code fences if GPT wrapped the JSON (rare at
    # "xhigh" reasoning effort, but cheap to handle).
    stripped = raw.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl >= 0:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -3]
        stripped = stripped.strip()
    data = orjson.loads(stripped)
    label = data["label"]
    if label not in _VALID_LABELS:
        raise ValueError(f"cc-NLI JSON has unknown label {label!r}")
    supporting = float(data["supporting_score"])
    refuting = float(data["refuting_score"])
    neutral = float(data["neutral_score"])
    total = supporting + refuting + neutral
    if total <= 0:
        raise ValueError(
            f"cc-NLI JSON scores sum to non-positive value {total!r}"
        )
    supporting /= total
    refuting /= total
    neutral /= total
    return NliResult(
        label=label,  # type: ignore[arg-type]
        supporting_score=supporting,
        refuting_score=refuting,
        neutral_score=neutral,
        reasoning=str(data.get("reasoning", "")),
        confidence=float(data.get("confidence", 0.0)),
    )


__all__ = ["NliOpenAIGpt54Adapter"]
