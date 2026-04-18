"""Unit tests for ``Pipeline.verify`` phase-callback kwarg (Stream D2).

D2's async handler runs ``pipeline.verify()`` and needs to advance the
DynamoDB job record's ``phase`` field at each of the five natural
boundaries D1 identified in §5 of its handoff:

  1. decomposing         — entering L1 decomposition
  2. retrieving_evidence — entering L1 evidence retrieval
  3. classifying         — entering L3/L4 NLI posteriors
  4. calibrating         — entering the L5 per-claim conformal loop
  5. assembling          — entering L7 document assembly

The brief is explicit: phase callbacks fire AROUND the phases, not
inside them. So the contract is:

* ``pipeline.verify(text, ..., phase_callback=cb)`` calls
  ``await cb(phase_name)`` once per boundary, in the order above,
  BEFORE the phase's work starts.
* Passing ``phase_callback=None`` (or omitting it) preserves pre-D2
  behaviour exactly — no callback invocations.
* An exception raised by the callback must NOT crash the pipeline.
  The async handler may be racing DynamoDB writes; a transient failure
  there should produce a warning log at worst, never a ``verify()``
  that aborts mid-document.

All five boundaries fire even when the Phase 1 placeholder path runs
(nli_adapter=None) — the phases exist independently of whether NLI is
wired. That way the polling UI gets a consistent progress signal
regardless of what implementation is active underneath.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import pytest

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

_FLAT_PACKAGES = {
    "adapters",
    "interfaces",
    "models",
    "pipeline",
    "config",
    "server",
    "services",
}
_cached_iface = sys.modules.get("interfaces")
_cached_iface_file = getattr(_cached_iface, "__file__", "") or ""
if _cached_iface is None or str(_SRC_API) not in _cached_iface_file:
    for _cached_name in list(sys.modules):
        _root = _cached_name.split(".", 1)[0]
        if _root in _FLAT_PACKAGES:
            del sys.modules[_cached_name]


# ---------------------------------------------------------------------------
# Stub implementations for every Pipeline dependency. All return canned data
# with no I/O and no LLM calls.
# ---------------------------------------------------------------------------


class _StubDecomposer:
    model_id = "stub-decomposer"

    async def decompose(self, text: str) -> list:
        from models.entities import Claim, ClaimType

        # Two trivial claims — enough to exercise the L5 loop.
        return [
            Claim(
                id=UUID(int=1),
                text=f"first claim about {text[:10]}",
                subject="x",
                predicate="is",
                object="y",
                claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
            ),
            Claim(
                id=UUID(int=2),
                text=f"second claim about {text[:10]}",
                subject="a",
                predicate="is",
                object="b",
                claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
            ),
        ]


class _StubRetrieval:
    """``AdaptiveEvidenceCollector`` look-alike. Returns no evidence per
    claim — the placeholder posteriors path trips the uniform prior
    branch, which is irrelevant to what we're testing (phase ordering)."""

    async def collect(self, claim, mcp_sources=None):  # noqa: D401, ANN001
        class _Result:
            evidence = []

        return _Result()


class _StubConformal:
    async def calibrate(self, *, claim, belief, domain, stratum):  # noqa: ANN001
        from interfaces.conformal import CalibratedVerdict

        return CalibratedVerdict(
            p_true=belief.p_true,
            interval_lower=0.0,
            interval_upper=1.0,
            coverage_target=0.9,
            calibration_set_id=None,
            calibration_n=0,
            domain=domain,
            stratum=stratum,
            fallback_used="general",
        )


def _build_pipeline():
    from pipeline.pipeline import Pipeline

    return Pipeline(
        decomposer=_StubDecomposer(),
        retrieval=_StubRetrieval(),
        conformal=_StubConformal(),
        domain_router=None,
        nli=None,
        nli_adapter=None,
        pcg=None,
        domain_adapters=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_callback_fires_five_boundaries_in_order() -> None:
    recorded: list[str] = []

    async def cb(phase: str) -> None:
        recorded.append(phase)

    pipe = _build_pipeline()
    verdict = await pipe.verify("hello world", phase_callback=cb)

    assert recorded == [
        "decomposing",
        "retrieving_evidence",
        "classifying",
        "calibrating",
        "assembling",
    ]
    # Sanity — verdict still shaped correctly end-to-end.
    assert verdict is not None
    assert verdict.document_score is not None


@pytest.mark.asyncio
async def test_phase_callback_none_preserves_pre_d2_behavior() -> None:
    # Passing nothing should not raise, and verify still produces a verdict.
    pipe = _build_pipeline()
    verdict = await pipe.verify("hello world")
    assert verdict is not None


@pytest.mark.asyncio
async def test_phase_callback_exception_does_not_crash_pipeline() -> None:
    calls: list[str] = []

    async def flaky_cb(phase: str) -> None:
        calls.append(phase)
        if phase == "classifying":
            raise RuntimeError("dynamodb transient")

    pipe = _build_pipeline()
    # Must NOT raise — a flaky progress-reporter cannot abort a verdict.
    verdict = await pipe.verify("hello", phase_callback=flaky_cb)
    assert verdict is not None
    # And the later callbacks still fired — failure of one phase write
    # does not silently skip the next boundary's write.
    assert "calibrating" in calls
    assert "assembling" in calls
