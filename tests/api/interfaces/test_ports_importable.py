"""Regression test: every v2 port is importable and carries the expected
members. This catches accidental signature drift or rename churn.

A minimal dummy implementation is also validated against each Protocol's
structural-typing check so we know a concrete adapter written against the
documented shape will pass ``isinstance(obj, Port)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/api is importable when running from repo root
_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from interfaces.conformal import CalibratedVerdict, ConformalCalibrator  # noqa: E402
from interfaces.domain import DomainAdapter, DomainRouter  # noqa: E402
from interfaces.feedback import FeedbackStore  # noqa: E402
from interfaces.nli import NLIService  # noqa: E402
from interfaces.pcg import PCGInferenceService  # noqa: E402
from models.domain import ALL_DOMAINS, Domain, DomainAssignment  # noqa: E402
from models.feedback import (  # noqa: E402
    CalibrationEntry,
    DisputedClaim,
    EvidenceCorrection,
    FeedbackSubmission,
)


# ---------------------------------------------------------------------------
# Presence checks (guard against accidental rename)
# ---------------------------------------------------------------------------


def test_domain_literal_covers_five_verticals() -> None:
    assert set(ALL_DOMAINS) == {
        "general",
        "biomedical",
        "legal",
        "code",
        "social",
    }


def test_domain_assignment_rejects_non_summing_weights() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DomainAssignment(
            weights={"general": 0.3, "biomedical": 0.3},  # sums to 0.6
            primary="general",
            soft=False,
        )


# ---------------------------------------------------------------------------
# Structural typing: dummy implementations pass isinstance()
# ---------------------------------------------------------------------------


class _DummyAdapter:
    """Minimal viable DomainAdapter. Should pass isinstance(DomainAdapter)."""

    @property
    def domain(self) -> Domain:
        return "general"

    def nli_model_id(self) -> str:
        return "dummy-nli"

    def source_credibility(self) -> dict[str, float]:
        return {}

    def calibration_set_id(self) -> str:
        return "general:any"

    def decomposition_hints(self) -> str | None:
        return None

    def claim_pair_relatedness_threshold(self) -> float:
        return 0.45


def test_dummy_adapter_satisfies_protocol() -> None:
    assert isinstance(_DummyAdapter(), DomainAdapter)


class _DummyRouter:
    async def route(self, claim):  # type: ignore[no-untyped-def]
        return DomainAssignment(weights={"general": 1.0}, primary="general", soft=False)


def test_dummy_router_satisfies_protocol() -> None:
    assert isinstance(_DummyRouter(), DomainRouter)


class _DummyNLI:
    async def claim_evidence(self, claim, evidence, adapter, *, rigor="balanced"):  # type: ignore[no-untyped-def]
        return []

    async def claim_claim(self, claims, adapter, *, rigor="balanced"):  # type: ignore[no-untyped-def]
        return {}

    async def health_check(self) -> bool:
        return True


def test_dummy_nli_satisfies_protocol() -> None:
    assert isinstance(_DummyNLI(), NLIService)


class _DummyPCG:
    async def infer(  # type: ignore[no-untyped-def]
        self,
        claims,
        evidence_per_claim,
        nli_claim_evidence,
        nli_claim_claim,
        adapter_per_claim,
        *,
        rigor="balanced",
    ):
        return {}


def test_dummy_pcg_satisfies_protocol() -> None:
    assert isinstance(_DummyPCG(), PCGInferenceService)


class _DummyConformal:
    async def calibrate(self, claim, belief, domain, stratum):  # type: ignore[no-untyped-def]
        return CalibratedVerdict(
            p_true=0.5,
            interval_lower=0.0,
            interval_upper=1.0,
            coverage_target=None,
            calibration_set_id=None,
            calibration_n=0,
            domain="general",
            stratum="general:any",
            fallback_used="general",
        )


def test_dummy_conformal_satisfies_protocol() -> None:
    assert isinstance(_DummyConformal(), ConformalCalibrator)


class _DummyFeedback:
    async def submit(self, submission):  # type: ignore[no-untyped-def]
        return "fake-id"

    async def promote_consensus(self) -> int:
        return 0

    async def get_calibration_set(self, partition):  # type: ignore[no-untyped-def]
        return []

    async def list_disputed_claims(self):  # type: ignore[no-untyped-def]
        return []


def test_dummy_feedback_satisfies_protocol() -> None:
    assert isinstance(_DummyFeedback(), FeedbackStore)


# ---------------------------------------------------------------------------
# CalibratedVerdict semantics
# ---------------------------------------------------------------------------


def test_calibrated_verdict_is_frozen_dataclass() -> None:
    import pytest

    cv = CalibratedVerdict(
        p_true=0.8,
        interval_lower=0.7,
        interval_upper=0.9,
        coverage_target=0.9,
        calibration_set_id="test",
        calibration_n=200,
        domain="general",
        stratum="general:temporal",
        fallback_used=None,
    )
    with pytest.raises(Exception):  # FrozenInstanceError / dataclass frozen
        cv.p_true = 0.9  # type: ignore[misc]


def test_feedback_submission_round_trip() -> None:
    from uuid import uuid4

    sub = FeedbackSubmission(
        request_id=uuid4(),
        claim_id=uuid4(),
        label="true",
        labeler_kind="user",
        labeler_id_hash="a" * 64,
        rationale="looks correct",
        evidence_corrections=[
            EvidenceCorrection(evidence_id="ev-1", correct_classification="supports")
        ],
        ip_hash=None,
    )
    payload = sub.model_dump_json()
    parsed = FeedbackSubmission.model_validate_json(payload)
    assert parsed == sub


def test_calibration_entry_partition_format() -> None:
    from datetime import datetime
    from uuid import uuid4

    entry = CalibrationEntry(
        id=uuid4(),
        claim_id=uuid4(),
        true_label="true",
        source_tier="adjudicator",
        n_concordant=1,
        calibration_set_partition="biomedical:quantitative",
        posterior_at_label_time=0.82,
        model_versions_at_label_time={"nli_biomedical": "v3"},
        created_at=datetime(2026, 4, 16),
    )
    assert ":" in entry.calibration_set_partition


def test_disputed_claim_minimal_fields() -> None:
    from datetime import datetime
    from uuid import uuid4

    d = DisputedClaim(claim_id=uuid4(), first_disputed_at=datetime.now())
    assert d.resolved_at is None
