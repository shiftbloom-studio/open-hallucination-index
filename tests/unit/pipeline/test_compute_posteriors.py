"""Unit tests for ``Pipeline._compute_posteriors`` (Phase 2 D1 NLI wiring).

D1 replaces the Phase 1 ``mean(similarity_score)`` placeholder with a
Beta-posterior update driven by an injected :class:`NliAdapter`. Per-
evidence ``classify()`` calls run concurrently under a
``Semaphore(10)`` so a 5-claim × 3-evidence fan-out does not serialise
15 LLM round-trips. This file pins all four of those behaviours:

1. Support-leaning NLI results push ``p_true`` notably above 0.5.
2. Refute-leaning NLI results push ``p_true`` notably below 0.5.
3. In-flight concurrency never exceeds 10 (the Semaphore cap), and at
   least one over-subscribed claim does observe >1 concurrent call —
   proving the gather is actually parallel, not accidentally serial.
4. ``reasoning == "nli_unavailable"`` results are *skipped*, not folded
   in as neutral evidence — a down LLM must leave the posterior at the
   uniform prior, not push it around with zeroed scores.

Tests are stubbed end-to-end: no live Gemini, no network, no $.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put ``src/api`` on the import path so ``from interfaces.*`` / ``from
# pipeline.*`` resolve the same way they do when the Lambda image runs.
_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

# Cross-worktree import gotcha (Stream B handover, plan §0): the shared
# venv's editable install points at the MAIN checkout's src/api, so any
# earlier test in the full-suite run that imports ``config`` transitively
# loads ``interfaces`` / ``adapters`` / ``pipeline`` from there into
# sys.modules. The sys.path insert above would then be too late because
# subsequent ``from pipeline.pipeline import ...`` reuses the cached
# package and walks its frozen ``__path__`` — which points at the main
# checkout's pipeline/, where this worktree's new Pipeline signature
# (``nli_adapter=`` kwarg) does NOT yet exist.
#
# The canonical snippet (per ``tests/unit/adapters/test_nli_gemini.py``
# header) unconditionally purges the flat-namespace packages. We adopt
# the same list verbatim, but skip the purge when ``interfaces`` is
# *already* resolved to this worktree — doing it twice in one session
# re-creates the module objects after Stream B's ``test_nli_gemini.py``
# has already bound ``NliAdapter``, which would break Stream B's
# ``interfaces_pkg.NliAdapter is NliAdapter`` identity assertion.
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

import asyncio  # noqa: E402
from uuid import uuid4  # noqa: E402

# ruff: noqa: I001  — ordering below is deliberate; must follow the
# sys.path insert above.
from interfaces.nli import NliResult  # noqa: E402
from models.domain import DomainAssignment  # noqa: E402
from models.entities import Claim, ClaimType, Evidence, EvidenceSource  # noqa: E402
from models.pcg import PosteriorBelief  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubDecomposer:
    """Minimal decomposer stub — ``_compute_posteriors`` never calls it,
    but Pipeline's ctor requires one."""

    async def decompose(self, text: str) -> list[Claim]:  # pragma: no cover
        return []

    async def decompose_with_context(
        self, text: str, context: str | None = None, max_claims: int | None = None
    ) -> list[Claim]:  # pragma: no cover
        return []

    async def health_check(self) -> bool:  # pragma: no cover
        return True


class _StubConformal:
    """Minimal conformal stub — ``_compute_posteriors`` never calls it,
    but Pipeline's ctor requires one."""

    async def calibrate(self, **kwargs: object) -> object:  # pragma: no cover
        raise AssertionError("conformal stub should not be called from _compute_posteriors")


class _StubNli:
    """Stub :class:`NliAdapter` returning a fixed :class:`NliResult` per
    call. Instrumented for concurrency observation so the Semaphore(10)
    bound inside the pipeline is asserted, not assumed.

    ``sleep_s`` is small but nonzero so that several fan-out tasks have
    time to pile up at the semaphore; without the yield, asyncio could
    schedule them effectively serially and the concurrency assertion
    would be a false negative.
    """

    def __init__(self, reply: NliResult, *, sleep_s: float = 0.02) -> None:
        self._reply = reply
        self._sleep_s = sleep_s
        self.calls: list[tuple[str, str]] = []
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def classify(self, claim_text: str, evidence_text: str) -> NliResult:
        async with self._lock:
            self.calls.append((claim_text, evidence_text))
            self._in_flight += 1
            if self._in_flight > self.max_in_flight:
                self.max_in_flight = self._in_flight
        try:
            await asyncio.sleep(self._sleep_s)
            return self._reply
        finally:
            async with self._lock:
                self._in_flight -= 1

    async def health_check(self) -> bool:  # pragma: no cover
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_ASSIGNMENT = DomainAssignment(
    weights={"general": 1.0, "biomedical": 0.0, "legal": 0.0, "code": 0.0, "social": 0.0},
    primary="general",
    soft=False,
)


def _claim(text: str = "c") -> Claim:
    return Claim(id=uuid4(), text=text, claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT)


def _evidence(content: str = "e") -> Evidence:
    return Evidence(source=EvidenceSource.MEDIAWIKI, content=content)


def _support_result(*, supporting: float = 0.90, refuting: float = 0.05) -> NliResult:
    neutral = 1.0 - supporting - refuting
    return NliResult(
        label="support",
        supporting_score=supporting,
        refuting_score=refuting,
        neutral_score=neutral,
        reasoning="supports",
        confidence=0.9,
    )


def _refute_result(*, supporting: float = 0.05, refuting: float = 0.90) -> NliResult:
    neutral = 1.0 - supporting - refuting
    return NliResult(
        label="refute",
        supporting_score=supporting,
        refuting_score=refuting,
        neutral_score=neutral,
        reasoning="refutes",
        confidence=0.9,
    )


def _unavailable_result() -> NliResult:
    """Matches the ``_NEUTRAL_FALLBACK`` sentinel emitted by
    :class:`~adapters.nli_gemini.NliGeminiAdapter` when every retry
    is exhausted (Stream B handoff)."""
    return NliResult(
        label="neutral",
        supporting_score=0.0,
        refuting_score=0.0,
        neutral_score=1.0,
        reasoning="nli_unavailable",
        confidence=0.0,
    )


def _asymmetric_neutral_result() -> NliResult:
    """A *real* neutral classification that still carries a mild
    support tilt — distinguishable from an ``nli_unavailable``
    sentinel because the Beta update *does* fold it in.

    Scores are asymmetric (0.4 / 0.1 / 0.5) so a folded-in result
    produces a p_true measurably different from the uniform 0.5,
    letting the test separate "folded" from "skipped".
    """
    return NliResult(
        label="neutral",
        supporting_score=0.4,
        refuting_score=0.1,
        neutral_score=0.5,
        reasoning="partially relevant",
        confidence=0.5,
    )


def _build_pipeline(nli_adapter: object | None) -> Pipeline:
    return Pipeline(
        decomposer=_StubDecomposer(),  # type: ignore[arg-type]
        retrieval=None,
        conformal=_StubConformal(),  # type: ignore[arg-type]
        nli_adapter=nli_adapter,  # type: ignore[arg-type]  # new D1 kwarg
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_supporting_evidence_pushes_p_true_above_uniform_prior() -> None:
    """Three support-leaning NLI results on one claim must drive the
    Beta posterior's ``p_true`` notably above 0.5 (the uniform prior
    at α=β=1). This is the headline Phase 2 behaviour change:
    /verify stops returning 0.5 for every claim once evidence is
    classified."""
    claim = _claim("Marie Curie won two Nobel prizes.")
    evidence = [_evidence("ev-a"), _evidence("ev-b"), _evidence("ev-c")]
    stub = _StubNli(_support_result())
    pipeline = _build_pipeline(nli_adapter=stub)

    posteriors = await pipeline._compute_posteriors(
        [claim],
        {claim.id: evidence},
        {claim.id: _DEFAULT_ASSIGNMENT},
    )

    belief = posteriors[claim.id]
    assert isinstance(belief, PosteriorBelief)
    # α=1+3*0.9=3.7, β=1+3*0.05=1.15 → p_true=3.7/4.85 ≈ 0.763.
    # Well above the uniform prior; pin with a generous floor so the
    # exact weighting can evolve without churning this test.
    assert belief.p_true > 0.6
    assert belief.p_false == pytest.approx(1.0 - belief.p_true)
    # All three evidence classifications observed.
    assert len(stub.calls) == 3


async def test_refuting_evidence_pushes_p_true_below_uniform_prior() -> None:
    """Mirror of the support case: three refute-leaning results drive
    ``p_true`` notably below 0.5."""
    claim = _claim("Stephen Hawking died in 2001.")
    evidence = [_evidence("ev-a"), _evidence("ev-b"), _evidence("ev-c")]
    stub = _StubNli(_refute_result())
    pipeline = _build_pipeline(nli_adapter=stub)

    posteriors = await pipeline._compute_posteriors(
        [claim],
        {claim.id: evidence},
        {claim.id: _DEFAULT_ASSIGNMENT},
    )

    belief = posteriors[claim.id]
    # Symmetric to the support case: p_true ≈ 1.15/4.85 ≈ 0.237.
    assert belief.p_true < 0.4
    assert belief.p_false == pytest.approx(1.0 - belief.p_true)
    assert len(stub.calls) == 3


async def test_semaphore_caps_in_flight_nli_calls_at_ten() -> None:
    """With more than 10 (claim, evidence) pairs fanning out under
    ``asyncio.gather``, the pipeline's internal ``Semaphore(10)`` must
    hold the high-water in-flight count to ≤ 10. We also want that
    high-water to be strictly greater than 1 — otherwise the test
    would pass even on a fully-serialised implementation."""
    claim = _claim("c")
    evidence = [_evidence(f"ev-{i}") for i in range(25)]
    stub = _StubNli(_support_result())
    pipeline = _build_pipeline(nli_adapter=stub)

    await pipeline._compute_posteriors(
        [claim],
        {claim.id: evidence},
        {claim.id: _DEFAULT_ASSIGNMENT},
    )

    # All 25 classifications eventually ran.
    assert len(stub.calls) == 25
    # The Semaphore(10) cap held.
    assert stub.max_in_flight <= 10, (
        f"Semaphore should cap concurrency at 10, observed {stub.max_in_flight}"
    )
    # And the pipeline actually ran in parallel — not a hidden serial loop.
    assert stub.max_in_flight > 1, (
        f"Expected concurrent classify calls, observed only {stub.max_in_flight}"
    )


async def test_nli_unavailable_results_are_skipped_not_folded_as_neutral() -> None:
    """The terminal-failure sentinel (``reasoning='nli_unavailable'``)
    must NOT push the Beta posterior around. An asymmetric real
    neutral (with a mild support tilt) must be folded in and move
    ``p_true`` off 0.5. The two together prove "skip" ≠ "fold neutral".

    Without this distinction a flaky LLM would silently bias every
    posterior toward 0.5 with zeroed scores even when the rest of
    the evidence disagreed — which is exactly what we don't want.
    """
    claim_unavailable = _claim("claim whose NLI is down")
    claim_neutral = _claim("claim classified as mild neutral")
    evidence = [_evidence("a"), _evidence("b"), _evidence("c")]

    stub_unavail = _StubNli(_unavailable_result())
    pipe_unavail = _build_pipeline(nli_adapter=stub_unavail)
    post_unavail = await pipe_unavail._compute_posteriors(
        [claim_unavailable],
        {claim_unavailable.id: list(evidence)},
        {claim_unavailable.id: _DEFAULT_ASSIGNMENT},
    )

    stub_neutral = _StubNli(_asymmetric_neutral_result())
    pipe_neutral = _build_pipeline(nli_adapter=stub_neutral)
    post_neutral = await pipe_neutral._compute_posteriors(
        [claim_neutral],
        {claim_neutral.id: list(evidence)},
        {claim_neutral.id: _DEFAULT_ASSIGNMENT},
    )

    # The stub WAS called for every evidence piece (proves we didn't
    # short-circuit before classify) — but the results were skipped,
    # leaving the Beta posterior at the uniform prior.
    assert len(stub_unavail.calls) == 3
    assert post_unavail[claim_unavailable.id].p_true == pytest.approx(0.5)

    # The asymmetric-neutral case DID fold in (s=0.4, r=0.1 per call),
    # so p_true shifts off 0.5 by a measurable margin (α=1+3*0.4=2.2,
    # β=1+3*0.1=1.3 → p_true ≈ 0.629).
    assert len(stub_neutral.calls) == 3
    assert abs(post_neutral[claim_neutral.id].p_true - 0.5) > 0.05


# ---------------------------------------------------------------------------
# pytest.approx shim (avoid a top-level import that would run before the
# sys.modules purge above in some runner configurations).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
