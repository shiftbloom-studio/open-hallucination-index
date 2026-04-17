"""Unit tests for the Gemini 3 Pro NLI adapter and its supporting types.

All LLM interactions in this module MUST go through the ``_CannedLLM`` stub
(see below). Zero live Gemini calls; zero network; zero $.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put ``src/api`` on the import path so ``from interfaces.*`` / ``from
# adapters.*`` resolve the same way they do when the Lambda image runs.
_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

# Fabian has two worktrees sharing one venv (ohi-stream-b for this run,
# open-hallucination-index as the main checkout). The venv's editable
# install points at the MAIN checkout's src/api, so any earlier test in
# the full-suite run that imports ``config`` transitively loads
# ``adapters`` from there into sys.modules. Our path insert above would
# then be too late because subsequent ``from adapters.foo import ...``
# reuses the cached package and walks its frozen ``__path__`` — which
# points at the main checkout's adapters/, where this worktree's new
# ``nli_gemini`` module does NOT exist. Purging the flat-namespace
# sibling packages from the import cache forces the next import to
# re-resolve against the sys.path we just rewrote.
_FLAT_PACKAGES = {
    "adapters",
    "interfaces",
    "models",
    "pipeline",
    "config",
    "server",
    "services",
}
for _cached_name in list(sys.modules):
    _root = _cached_name.split(".", 1)[0]
    if _root in _FLAT_PACKAGES:
        del sys.modules[_cached_name]

from collections.abc import AsyncIterator  # noqa: E402
from typing import Any  # noqa: E402

from adapters.nli_gemini import NliGeminiAdapter  # noqa: E402
from interfaces.llm import LLMMessage, LLMProvider, LLMResponse  # noqa: E402
from interfaces.nli import NliAdapter, NliResult  # noqa: E402

# ruff: noqa: I001  — the block above is deliberately ordered after the
# sys.path insert so `from interfaces.*` resolves to THIS worktree's
# src/api tree, not any installed editable-install from a sibling checkout.


# ---------------------------------------------------------------------------
# Canned LLM stub — the ONLY thing the adapter talks to in these tests.
# Zero live Gemini calls, zero network, zero $.
# ---------------------------------------------------------------------------


class _CannedLLM(LLMProvider):
    """Records every call and pops the next reply off a scripted queue.

    Each queue entry is either:
    * ``str`` — returned verbatim as ``LLMResponse.content``
    * ``Exception`` — raised from ``complete()`` (simulates transport err)

    Exhausting the queue is itself an assertion signal: the test should
    not have called more times than it scripted. We raise ``AssertionError``
    so bugs that loop forever are caught instead of deadlocking.
    """

    def __init__(
        self,
        replies: list[str | Exception],
        *,
        model: str = "gemini-3-pro-preview",
    ) -> None:
        self._replies = list(replies)
        self._model = model
        self.calls: list[dict[str, Any]] = []
        self.health_calls = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "json_mode": json_mode,
                "max_tokens": max_tokens,
                "stop": stop,
            }
        )
        if not self._replies:
            raise AssertionError(
                "CannedLLM ran out of scripted replies — test undercounted calls"
            )
        nxt = self._replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return LLMResponse(content=nxt, model=self._model)

    async def complete_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if False:  # pragma: no cover - generator shape only
            yield ""

    async def health_check(self) -> bool:
        self.health_calls += 1
        return True

    @property
    def model_name(self) -> str:
        return self._model


def _json_reply(
    label: str,
    supporting: float,
    refuting: float,
    neutral: float,
    *,
    reasoning: str = "key fact",
    confidence: float = 0.9,
) -> str:
    import orjson

    return orjson.dumps(
        {
            "label": label,
            "supporting_score": supporting,
            "refuting_score": refuting,
            "neutral_score": neutral,
            "reasoning": reasoning,
            "confidence": confidence,
        }
    ).decode()


def test_nli_result_is_frozen_dataclass_with_expected_fields() -> None:
    """NliResult captures the NLI classification + 3-way scores + reasoning.

    The shape is the contract D1 will consume: D1 reads
    ``supporting_score`` and ``refuting_score`` to fold into the Beta
    posterior update, and surfaces ``label`` / ``reasoning`` in the API
    response for UI explanation.
    """
    import pytest

    result = NliResult(
        label="support",
        supporting_score=0.8,
        refuting_score=0.1,
        neutral_score=0.1,
        reasoning="Evidence directly confirms claim.",
        confidence=0.9,
    )
    assert result.label == "support"
    assert result.supporting_score == 0.8
    assert result.refuting_score == 0.1
    assert result.neutral_score == 0.1
    assert result.reasoning == "Evidence directly confirms claim."
    assert result.confidence == 0.9

    # Frozen: posterior code must never mutate an NliResult in place.
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        result.label = "refute"  # type: ignore[misc]


class _StructurallyValidAdapter:
    """Bare-minimum implementation exercising the NliAdapter Protocol
    shape. No behavior — the assertion is only that isinstance() at
    runtime accepts this against the Protocol."""

    async def classify(
        self, claim_text: str, evidence_text: str
    ) -> NliResult:  # pragma: no cover - protocol check only
        return NliResult(
            label="neutral",
            supporting_score=0.0,
            refuting_score=0.0,
            neutral_score=1.0,
            reasoning="stub",
            confidence=0.0,
        )

    async def health_check(self) -> bool:  # pragma: no cover
        return True


def test_nli_adapter_is_runtime_checkable_protocol() -> None:
    """NliAdapter is structurally typed so any class with matching
    ``classify`` + ``health_check`` coroutines counts as an adapter.
    D1's dependency injection relies on this duck-type contract.
    """
    assert isinstance(_StructurallyValidAdapter(), NliAdapter)


# ---------------------------------------------------------------------------
# NliGeminiAdapter — single-pass label dispatch
# ---------------------------------------------------------------------------


async def test_classify_support_label_single_pass() -> None:
    """Happy path: K=1, LLM returns well-formed support JSON, adapter
    returns an NliResult with the same fields."""
    llm = _CannedLLM(
        [
            _json_reply(
                "support",
                0.88,
                0.05,
                0.07,
                reasoning="Passage states the claim verbatim.",
                confidence=0.93,
            )
        ]
    )
    adapter = NliGeminiAdapter(llm=llm)

    result = await adapter.classify(
        "Marie Curie won two Nobel prizes.",
        "Marie Curie is the only woman to have won Nobel Prizes in two different sciences.",
    )

    assert isinstance(result, NliResult)
    assert result.label == "support"
    # Scores are renormalized, but should be very close to what the LLM
    # returned since 0.88 + 0.05 + 0.07 = 1.00.
    assert abs(result.supporting_score - 0.88) < 1e-6
    assert abs(result.refuting_score - 0.05) < 1e-6
    assert abs(result.neutral_score - 0.07) < 1e-6
    assert result.reasoning == "Passage states the claim verbatim."
    assert result.confidence == 0.93
    # Single-pass: exactly one LLM call.
    assert len(llm.calls) == 1
    # Adapter requests json_mode so Gemini emits strict JSON.
    assert llm.calls[0]["json_mode"] is True
    # The prompt must carry both the claim and the evidence so Gemini
    # has the full context.
    combined = "".join(m.content for m in llm.calls[0]["messages"])
    assert "Marie Curie won two Nobel prizes." in combined
    assert "only woman" in combined


async def test_classify_refute_label_single_pass() -> None:
    """K=1, LLM says refute → adapter returns a refute NliResult."""
    llm = _CannedLLM(
        [
            _json_reply(
                "refute",
                0.04,
                0.90,
                0.06,
                reasoning="Passage contradicts the claim's date.",
                confidence=0.88,
            )
        ]
    )
    adapter = NliGeminiAdapter(llm=llm)

    result = await adapter.classify(
        "Stephen Hawking died in 2001.",
        "Stephen Hawking died on 14 March 2018.",
    )

    assert result.label == "refute"
    assert result.refuting_score > result.supporting_score
    assert result.refuting_score > result.neutral_score
    assert result.reasoning == "Passage contradicts the claim's date."


async def test_classify_neutral_label_single_pass() -> None:
    """K=1, LLM says neutral → adapter returns a neutral NliResult; the
    evidence neither confirms nor contradicts the claim."""
    llm = _CannedLLM(
        [
            _json_reply(
                "neutral",
                0.10,
                0.10,
                0.80,
                reasoning="Passage is off-topic relative to the claim.",
                confidence=0.75,
            )
        ]
    )
    adapter = NliGeminiAdapter(llm=llm)

    result = await adapter.classify(
        "Marie Curie won two Nobel prizes.",
        "The Eiffel Tower was completed in 1889.",
    )

    assert result.label == "neutral"
    assert result.neutral_score > result.supporting_score
    assert result.neutral_score > result.refuting_score
    # Confidence is still the LLM's self-assessment, untouched by the
    # adapter even when the outcome is a "low-signal" neutral.
    assert result.confidence == 0.75


# ---------------------------------------------------------------------------
# Self-consistency (K > 1) — majority-vote label, averaged scores
# ---------------------------------------------------------------------------


async def test_self_consistency_k3_majority_vote() -> None:
    """K=3 with two support samples and one refute sample: adapter must
    pick ``support`` by majority vote, average the two support samples'
    scores, and discard the out-of-majority refute sample's scores."""
    llm = _CannedLLM(
        [
            _json_reply(
                "support",
                0.80,
                0.10,
                0.10,
                reasoning="first support key fact",
                confidence=0.90,
            ),
            _json_reply(
                "refute",
                0.10,
                0.80,
                0.10,
                reasoning="outlier sample",
                confidence=0.50,
            ),
            _json_reply(
                "support",
                0.90,
                0.05,
                0.05,
                reasoning="second support key fact",
                confidence=0.95,
            ),
        ]
    )
    adapter = NliGeminiAdapter(llm=llm, self_consistency_k=3)

    result = await adapter.classify("claim text", "evidence text")

    # 2 supports vs 1 refute → majority is support.
    assert result.label == "support"
    # Scores are averaged over the two winning samples only (NOT all
    # three), so the refute sample's 0.80 refuting_score is discarded:
    # support = (0.80 + 0.90) / 2 = 0.85
    # refuting = (0.10 + 0.05) / 2 = 0.075
    # neutral  = (0.10 + 0.05) / 2 = 0.075
    assert abs(result.supporting_score - 0.85) < 1e-6
    assert abs(result.refuting_score - 0.075) < 1e-6
    assert abs(result.neutral_score - 0.075) < 1e-6
    # Confidence is averaged over the winning samples: (0.90 + 0.95) / 2
    assert abs(result.confidence - 0.925) < 1e-6
    # Adapter made exactly K LLM calls.
    assert len(llm.calls) == 3
    # Every K-sampling call uses temperature=0.2 per plan §4.2.
    for call in llm.calls:
        assert call["temperature"] == 0.2


# ---------------------------------------------------------------------------
# Retry + terminal neutral fallback
# ---------------------------------------------------------------------------


async def test_transport_error_retries_then_succeeds() -> None:
    """Two transport failures, then success: adapter must retry and
    ultimately return the successful classification, not a fallback."""
    llm = _CannedLLM(
        [
            ConnectionError("network glitch 1"),
            TimeoutError("network glitch 2"),
            _json_reply(
                "support",
                0.9,
                0.05,
                0.05,
                reasoning="third attempt succeeds",
                confidence=0.9,
            ),
        ]
    )
    adapter = NliGeminiAdapter(llm=llm, max_retries=3)

    result = await adapter.classify("claim", "evidence")

    assert result.label == "support"
    assert result.reasoning == "third attempt succeeds"
    assert len(llm.calls) == 3


async def test_transport_error_terminal_neutral_fallback() -> None:
    """All attempts raise transport errors: adapter returns the
    ``nli_unavailable`` neutral fallback with ``confidence=0.0``.

    The pipeline's posterior update relies on this: a down LLM must not
    crash verification, it must produce a neutral NliResult that the
    Beta update can fold in as a no-op-ish signal.
    """
    llm = _CannedLLM(
        [
            ConnectionError("first"),
            ConnectionError("second"),
            ConnectionError("third"),
        ]
    )
    adapter = NliGeminiAdapter(llm=llm, max_retries=3)

    result = await adapter.classify("claim", "evidence")

    assert result.label == "neutral"
    assert result.supporting_score == 0.0
    assert result.refuting_score == 0.0
    assert result.neutral_score == 1.0
    assert result.reasoning == "nli_unavailable"
    assert result.confidence == 0.0
    # Exactly max_retries attempts; no more, no less.
    assert len(llm.calls) == 3


async def test_max_retries_param_is_respected() -> None:
    """max_retries=1 disables retry: one failure → immediate fallback."""
    llm = _CannedLLM([ConnectionError("boom")])
    adapter = NliGeminiAdapter(llm=llm, max_retries=1)

    result = await adapter.classify("c", "e")

    assert result.reasoning == "nli_unavailable"
    assert len(llm.calls) == 1


async def test_json_parse_error_retries_then_recovers() -> None:
    """LLM emits malformed JSON on first call, valid JSON on second:
    adapter retries and returns the valid classification. This is a
    realistic Gemini failure mode — the model occasionally wraps the
    JSON in markdown fences or prose, and Gemini 3's strict JSON mode
    is opt-in, not guaranteed."""
    llm = _CannedLLM(
        [
            "not valid json — missing braces",
            _json_reply(
                "support",
                0.9,
                0.05,
                0.05,
                reasoning="valid on retry",
                confidence=0.9,
            ),
        ]
    )
    adapter = NliGeminiAdapter(llm=llm, max_retries=3)

    result = await adapter.classify("c", "e")

    assert result.label == "support"
    assert result.reasoning == "valid on retry"
    assert len(llm.calls) == 2


async def test_json_parse_error_terminal_neutral_fallback() -> None:
    """Every attempt returns malformed JSON: adapter returns the
    ``nli_unavailable`` neutral fallback. Distinct from transport
    error path but funnelled into the same fallback sentinel.
    """
    llm = _CannedLLM(
        [
            "not json #1",
            "not json #2",
            "not json #3",
        ]
    )
    adapter = NliGeminiAdapter(llm=llm, max_retries=3)

    result = await adapter.classify("c", "e")

    assert result.label == "neutral"
    assert result.reasoning == "nli_unavailable"
    assert result.confidence == 0.0
    assert len(llm.calls) == 3


async def test_scores_are_renormalised_when_they_do_not_sum_to_one() -> None:
    """Gemini occasionally emits 3 scores whose sum drifts from 1.0
    (rounding, mis-following the prompt). The adapter must renormalise
    so downstream posterior-update math can trust the sum==1 invariant.
    """
    import orjson

    drifted = orjson.dumps(
        {
            "label": "support",
            "supporting_score": 0.8,
            "refuting_score": 0.1,
            "neutral_score": 0.05,  # sums to 0.95, not 1.0
            "reasoning": "drifted scores",
            "confidence": 0.9,
        }
    ).decode()
    llm = _CannedLLM([drifted])
    adapter = NliGeminiAdapter(llm=llm)

    result = await adapter.classify("c", "e")

    # Renormalised: each /= 0.95.
    assert abs(result.supporting_score - (0.8 / 0.95)) < 1e-6
    assert abs(result.refuting_score - (0.1 / 0.95)) < 1e-6
    assert abs(result.neutral_score - (0.05 / 0.95)) < 1e-6
    # And now the three sum to exactly 1.0.
    assert (
        abs(
            (
                result.supporting_score
                + result.refuting_score
                + result.neutral_score
            )
            - 1.0
        )
        < 1e-9
    )


async def test_zero_sum_scores_treated_as_parse_error() -> None:
    """All three scores being 0 is pathological — the adapter can't
    renormalise by 0. It must fall back rather than divide-by-zero."""
    import orjson

    zeros = orjson.dumps(
        {
            "label": "neutral",
            "supporting_score": 0.0,
            "refuting_score": 0.0,
            "neutral_score": 0.0,
            "reasoning": "no confidence in any label",
            "confidence": 0.9,
        }
    ).decode()
    llm = _CannedLLM([zeros, zeros, zeros])
    adapter = NliGeminiAdapter(llm=llm, max_retries=3)

    result = await adapter.classify("c", "e")

    assert result.reasoning == "nli_unavailable"


async def test_unknown_label_is_treated_as_parse_error() -> None:
    """JSON is valid but ``label`` is not one of support/refute/neutral:
    adapter counts it as a parse error and retries/falls back."""
    import orjson

    bad_label = orjson.dumps(
        {
            "label": "entails_strongly",  # not in the allowed set
            "supporting_score": 0.9,
            "refuting_score": 0.05,
            "neutral_score": 0.05,
            "reasoning": "bad label",
            "confidence": 0.9,
        }
    ).decode()
    llm = _CannedLLM([bad_label, bad_label, bad_label])
    adapter = NliGeminiAdapter(llm=llm, max_retries=3)

    result = await adapter.classify("c", "e")

    assert result.label == "neutral"
    assert result.reasoning == "nli_unavailable"


# ---------------------------------------------------------------------------
# Health check — delegates to underlying LLMProvider
# ---------------------------------------------------------------------------


async def test_health_check_delegates_to_underlying_llm() -> None:
    """Adapter has no state of its own worth probing — its health is
    the underlying LLMProvider's health. The Lambda health endpoint
    will call this via the DI container's wiring (D1 / Task 1.9)."""
    llm = _CannedLLM([])  # no replies needed; no classify calls in this test
    adapter = NliGeminiAdapter(llm=llm)

    ok = await adapter.health_check()

    assert ok is True
    assert llm.health_calls == 1


def test_nli_gemini_adapter_satisfies_nli_adapter_protocol() -> None:
    """Guard against signature drift in either NliAdapter or NliGeminiAdapter."""
    llm = _CannedLLM([])
    adapter = NliGeminiAdapter(llm=llm)
    assert isinstance(adapter, NliAdapter)


def test_nli_types_are_re_exported_from_interfaces_package() -> None:
    """The ``interfaces`` package already re-exports every other v2 port
    (NLIService, PCGInferenceService, ConformalCalibrator, ...). The
    LLM-based NLI contract should follow the same convention so D1 can
    ``from interfaces import NliAdapter, NliResult`` without a deeper
    import path. ``NliLabel`` is exported alongside because it's part of
    the public contract (the plan §4.2 prompt enumerates it)."""
    import interfaces as interfaces_pkg

    assert interfaces_pkg.NliAdapter is NliAdapter
    assert interfaces_pkg.NliResult is NliResult
    assert "NliAdapter" in interfaces_pkg.__all__
    assert "NliResult" in interfaces_pkg.__all__
    assert "NliLabel" in interfaces_pkg.__all__


# ---------------------------------------------------------------------------
# Low-confidence snap at self-consistency aggregation
# ---------------------------------------------------------------------------


async def test_self_consistency_low_confidence_snaps_to_zero() -> None:
    """All winning samples agree on the label but are each very low
    confidence: averaged confidence falls under the snap threshold and
    is forced to exactly 0.0. This gives D1 a clean ``confidence == 0.0``
    signal regardless of whether the pathway was a terminal fallback or a
    run of agreeing-but-unsure samples.

    The label vote and scores still stand — the adapter isn't rewriting
    the classification, only flagging it as low-signal via confidence.
    Reasoning is also preserved (not swapped for ``nli_unavailable``)
    because the UI should still be able to show the model's (weak)
    explanation for why it picked the label.
    """
    llm = _CannedLLM(
        [
            _json_reply(
                "support", 0.6, 0.2, 0.2,
                reasoning="weak signal 1", confidence=0.02,
            ),
            _json_reply(
                "support", 0.7, 0.2, 0.1,
                reasoning="weak signal 2", confidence=0.03,
            ),
            _json_reply(
                "support", 0.5, 0.3, 0.2,
                reasoning="weak signal 3", confidence=0.01,
            ),
        ]
    )
    adapter = NliGeminiAdapter(llm=llm, self_consistency_k=3)

    result = await adapter.classify("c", "e")

    # Average raw confidence is (0.02+0.03+0.01)/3 = 0.02 — below the
    # snap threshold, so forced to exactly 0.0.
    assert result.confidence == 0.0
    # The label vote and the renormalised scores are unaffected.
    assert result.label == "support"
    assert result.supporting_score > result.refuting_score
    # Not a terminal fallback; the reasoning is still a real one.
    assert result.reasoning != "nli_unavailable"
    assert result.reasoning in {
        "weak signal 1",
        "weak signal 2",
        "weak signal 3",
    }


async def test_self_consistency_above_floor_confidence_is_preserved() -> None:
    """Averaged confidence comfortably above the snap threshold passes
    through unchanged. Guards against a too-aggressive floor that would
    swallow merely-mediocre results."""
    llm = _CannedLLM(
        [
            _json_reply(
                "support", 0.8, 0.1, 0.1,
                reasoning="ok 1", confidence=0.65,
            ),
            _json_reply(
                "support", 0.7, 0.2, 0.1,
                reasoning="ok 2", confidence=0.70,
            ),
            _json_reply(
                "support", 0.75, 0.2, 0.05,
                reasoning="ok 3", confidence=0.75,
            ),
        ]
    )
    adapter = NliGeminiAdapter(llm=llm, self_consistency_k=3)

    result = await adapter.classify("c", "e")

    # Average 0.70 — well above any reasonable floor.
    assert abs(result.confidence - 0.70) < 1e-6
