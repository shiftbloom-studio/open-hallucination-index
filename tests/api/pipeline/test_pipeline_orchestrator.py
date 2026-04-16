"""End-to-end orchestrator tests with mocked layers.

Verifies that the Phase 1 placeholder orchestrator produces valid
``DocumentVerdict`` objects end-to-end with all-mocked dependencies.
Real integration with LLM / graph / vector stores lands in Task 1.8
(when the /verify route is wired to the DI-provided pipeline).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from interfaces.decomposition import ClaimDecomposer  # noqa: E402
from models.entities import Claim, ClaimType, Evidence, EvidenceSource  # noqa: E402
from models.results import DocumentVerdict  # noqa: E402
from pipeline.conformal.calibration_store import InMemoryCalibrationStore  # noqa: E402
from pipeline.conformal.split_conformal import SplitConformalCalibrator  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Mock ports
# ---------------------------------------------------------------------------


class _MockDecomposer(ClaimDecomposer):
    def __init__(self, claims: list[Claim]) -> None:
        self._claims = claims

    async def decompose(self, text: str) -> list[Claim]:
        return self._claims

    async def decompose_with_context(
        self, text: str, context: str | None = None, max_claims: int | None = None
    ) -> list[Claim]:
        return self._claims

    async def health_check(self) -> bool:
        return True


class _MockCollector:
    """Satisfies the collector's ``collect`` surface without extra deps."""

    def __init__(self, evidence_by_claim_text: dict[str, list[Evidence]]) -> None:
        self._by_text = evidence_by_claim_text

    async def collect(self, claim: Claim, **kwargs: Any) -> Any:
        class _Result:
            def __init__(self, ev: list[Evidence]) -> None:
                self.evidence = ev

        return _Result(self._by_text.get(claim.text, []))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_with_zero_claims() -> None:
    pipeline = Pipeline(
        decomposer=_MockDecomposer([]),
        retrieval=None,
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv = await pipeline.verify("")
    assert isinstance(dv, DocumentVerdict)
    assert dv.claims == []
    assert dv.document_score == 1.0
    assert dv.pipeline_version == "ohi-v2.0"


@pytest.mark.asyncio
async def test_orchestrator_single_claim_with_no_evidence() -> None:
    claim = Claim(
        id=uuid4(), text="Einstein was born in 1879.", claim_type=ClaimType.TEMPORAL
    )
    pipeline = Pipeline(
        decomposer=_MockDecomposer([claim]),
        retrieval=None,
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv = await pipeline.verify("Einstein was born in 1879.")
    assert len(dv.claims) == 1
    # Phase 1: no evidence → p_true = 0.5 (maximum uncertainty)
    assert dv.claims[0].p_true == pytest.approx(0.5)
    # Phase 1 stub calibration → general fallback, null coverage target
    assert dv.claims[0].fallback_used == "general"
    assert dv.claims[0].coverage_target is None
    # Routes everything to general in Phase 1
    assert dv.claims[0].domain == "general"


@pytest.mark.asyncio
async def test_orchestrator_single_claim_with_evidence() -> None:
    claim = Claim(id=uuid4(), text="Claim", claim_type=ClaimType.UNCLASSIFIED)
    evidence = [
        Evidence(
            source=EvidenceSource.WIKIPEDIA,
            content="Supporting evidence.",
            similarity_score=0.85,
        ),
        Evidence(
            source=EvidenceSource.WIKIPEDIA,
            content="More supporting evidence.",
            similarity_score=0.75,
        ),
    ]
    collector = _MockCollector({"Claim": evidence})
    pipeline = Pipeline(
        decomposer=_MockDecomposer([claim]),
        retrieval=collector,  # type: ignore[arg-type]
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv = await pipeline.verify("Claim")
    # Phase 1 placeholder: posterior = mean of evidence similarity scores
    assert dv.claims[0].p_true == pytest.approx(0.80)
    # Evidence list is preserved on the verdict
    assert len(dv.claims[0].supporting_evidence) == 2


@pytest.mark.asyncio
async def test_orchestrator_processing_time_recorded() -> None:
    claim = Claim(id=uuid4(), text="Claim", claim_type=ClaimType.UNCLASSIFIED)
    pipeline = Pipeline(
        decomposer=_MockDecomposer([claim]),
        retrieval=None,
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv = await pipeline.verify("Some text.")
    assert dv.processing_time_ms >= 0.0


@pytest.mark.asyncio
async def test_orchestrator_model_versions_populated() -> None:
    claim = Claim(id=uuid4(), text="Claim", claim_type=ClaimType.UNCLASSIFIED)
    pipeline = Pipeline(
        decomposer=_MockDecomposer([claim]),
        retrieval=None,
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv = await pipeline.verify("Some text.")
    # Phase 1 placeholders show up explicitly — callers can see the
    # pipeline hasn't graduated to real layers yet.
    assert "phase1" in dv.model_versions["pcg"]
    assert "phase1" in dv.model_versions["domain_router"]


@pytest.mark.asyncio
async def test_orchestrator_preserves_rigor_tier() -> None:
    claim = Claim(id=uuid4(), text="Claim", claim_type=ClaimType.UNCLASSIFIED)
    pipeline = Pipeline(
        decomposer=_MockDecomposer([claim]),
        retrieval=None,
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv_fast = await pipeline.verify("Text", rigor="fast")
    dv_max = await pipeline.verify("Text", rigor="maximum")
    assert dv_fast.rigor == "fast"
    assert dv_max.rigor == "maximum"


@pytest.mark.asyncio
async def test_orchestrator_request_id_round_trip() -> None:
    claim = Claim(id=uuid4(), text="Claim", claim_type=ClaimType.UNCLASSIFIED)
    pipeline = Pipeline(
        decomposer=_MockDecomposer([claim]),
        retrieval=None,
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    rid = uuid4()
    dv = await pipeline.verify("Text", request_id=rid)
    assert dv.request_id == rid


@pytest.mark.asyncio
async def test_orchestrator_handles_retrieval_errors_gracefully() -> None:
    """A failing retrieval must not kill the request; evidence defaults
    to [] for that claim and the pipeline still produces a verdict."""

    class _BadCollector:
        async def collect(self, claim: Claim, **kwargs: Any) -> Any:
            raise RuntimeError("simulated backend failure")

    claim = Claim(id=uuid4(), text="Claim", claim_type=ClaimType.UNCLASSIFIED)
    pipeline = Pipeline(
        decomposer=_MockDecomposer([claim]),
        retrieval=_BadCollector(),  # type: ignore[arg-type]
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv = await pipeline.verify("Some text.")
    # Still returns a valid verdict; claim just has no evidence → p_true=0.5
    assert len(dv.claims) == 1
    assert dv.claims[0].supporting_evidence == []
    assert dv.claims[0].p_true == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_orchestrator_decomposition_coverage_reasonable() -> None:
    """Coverage is bounded [0, 1]."""
    claims = [
        Claim(id=uuid4(), text=f"Claim {i}", claim_type=ClaimType.UNCLASSIFIED)
        for i in range(3)
    ]
    pipeline = Pipeline(
        decomposer=_MockDecomposer(claims),
        retrieval=None,
        conformal=SplitConformalCalibrator(InMemoryCalibrationStore()),
    )
    dv = await pipeline.verify("Sentence one. Sentence two.")
    assert 0.0 <= dv.decomposition_coverage <= 1.0
