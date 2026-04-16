"""Tests for the v2 result models (ClaimVerdict, DocumentVerdict, NLI, PCG).

All pure-unit; no infra, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

# Ensure src/api is importable when running from repo root
_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from models.entities import Claim, ClaimType  # noqa: E402
from models.nli import NLIDistribution  # noqa: E402
from models.pcg import PosteriorBelief  # noqa: E402
from models.results import ClaimEdge, ClaimVerdict, DocumentVerdict, EdgeType  # noqa: E402


def _claim() -> Claim:
    return Claim(
        id=uuid4(),
        text="Einstein was born in 1879.",
        claim_type=ClaimType.TEMPORAL,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# ClaimVerdict
# ---------------------------------------------------------------------------


def test_claim_verdict_accepts_null_coverage_target_when_fallback() -> None:
    cv = ClaimVerdict(
        claim=_claim(),
        p_true=0.96,
        interval=(0.91, 0.99),
        coverage_target=None,
        domain="general",
        domain_assignment_weights={"general": 1.0},
        supporting_evidence=[],
        refuting_evidence=[],
        pcg_neighbors=[],
        nli_self_consistency_variance=0.012,
        bp_validated=None,
        information_gain=0.04,
        queued_for_review=False,
        calibration_set_id=None,
        calibration_n=0,
        fallback_used="general",
    )
    assert cv.coverage_target is None
    assert cv.fallback_used == "general"


def test_claim_verdict_rejects_invalid_p_true() -> None:
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim=_claim(),
            p_true=1.5,  # invalid
            interval=(0.0, 1.0),
            coverage_target=None,
            domain="general",
            domain_assignment_weights={"general": 1.0},
            supporting_evidence=[],
            refuting_evidence=[],
            pcg_neighbors=[],
            nli_self_consistency_variance=0.0,
            bp_validated=None,
            information_gain=0.0,
            queued_for_review=False,
            calibration_set_id=None,
            calibration_n=0,
            fallback_used="general",
        )


def test_claim_verdict_rejects_inverted_interval() -> None:
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim=_claim(),
            p_true=0.5,
            interval=(0.7, 0.3),  # upper < lower
            coverage_target=0.9,
            domain="general",
            domain_assignment_weights={"general": 1.0},
            supporting_evidence=[],
            refuting_evidence=[],
            pcg_neighbors=[],
            nli_self_consistency_variance=0.0,
            bp_validated=True,
            information_gain=0.0,
            queued_for_review=False,
            calibration_set_id="c1",
            calibration_n=200,
            fallback_used=None,
        )


def test_claim_verdict_accepts_bp_validated_true_false_or_none() -> None:
    for bp in (True, False, None):
        cv = ClaimVerdict(
            claim=_claim(),
            p_true=0.5,
            interval=(0.4, 0.6),
            coverage_target=0.9,
            domain="general",
            domain_assignment_weights={"general": 1.0},
            supporting_evidence=[],
            refuting_evidence=[],
            pcg_neighbors=[],
            nli_self_consistency_variance=0.0,
            bp_validated=bp,
            information_gain=0.0,
            queued_for_review=False,
            calibration_set_id="c1",
            calibration_n=200,
            fallback_used=None,
        )
        assert cv.bp_validated is bp


# ---------------------------------------------------------------------------
# ClaimEdge
# ---------------------------------------------------------------------------


def test_claim_edge_serialization_round_trip() -> None:
    edge = ClaimEdge(
        neighbor_claim_id=uuid4(),
        edge_type=EdgeType.ENTAIL,
        edge_strength=0.81,
    )
    payload = edge.model_dump_json()
    parsed = ClaimEdge.model_validate_json(payload)
    assert parsed == edge


def test_claim_edge_rejects_strength_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ClaimEdge(
            neighbor_claim_id=uuid4(),
            edge_type=EdgeType.CONTRADICT,
            edge_strength=1.5,
        )


# ---------------------------------------------------------------------------
# DocumentVerdict
# ---------------------------------------------------------------------------


def test_document_verdict_min_required_fields() -> None:
    dv = DocumentVerdict(
        document_score=0.74,
        document_interval=(0.61, 0.84),
        internal_consistency=0.83,
        claims=[],
        decomposition_coverage=0.0,
        processing_time_ms=87341.0,
        rigor="balanced",
        refinement_passes_executed=0,
        model_versions={"decomposer": "v1", "router": "n/a"},
        request_id=uuid4(),
    )
    assert dv.pipeline_version == "ohi-v2.0"
    assert dv.rigor == "balanced"
    assert dv.refinement_passes_executed == 0


def test_document_verdict_freeze_immutable() -> None:
    dv = DocumentVerdict(
        document_score=0.5,
        document_interval=(0.0, 1.0),
        internal_consistency=0.0,
        claims=[],
        decomposition_coverage=0.0,
        processing_time_ms=0.0,
        rigor="fast",
        refinement_passes_executed=0,
        model_versions={},
        request_id=uuid4(),
    )
    with pytest.raises(ValidationError):
        dv.document_score = 0.6  # type: ignore[misc]  # frozen


def test_document_verdict_rejects_inverted_interval() -> None:
    with pytest.raises(ValidationError):
        DocumentVerdict(
            document_score=0.5,
            document_interval=(0.8, 0.2),
            internal_consistency=0.5,
            claims=[],
            decomposition_coverage=0.0,
            processing_time_ms=0.0,
            rigor="fast",
            refinement_passes_executed=0,
            model_versions={},
            request_id=uuid4(),
        )


# ---------------------------------------------------------------------------
# NLIDistribution
# ---------------------------------------------------------------------------


def test_nli_distribution_accepts_valid_probabilities() -> None:
    d = NLIDistribution(
        entail=0.7, contradict=0.2, neutral=0.1, variance=0.0, nli_model_id="x"
    )
    assert abs((d.entail + d.contradict + d.neutral) - 1.0) < 1e-6


def test_nli_distribution_rejects_sum_not_one() -> None:
    with pytest.raises(ValidationError):
        NLIDistribution(
            entail=0.5, contradict=0.5, neutral=0.5, variance=0.0, nli_model_id="x"
        )


# ---------------------------------------------------------------------------
# PosteriorBelief
# ---------------------------------------------------------------------------


def test_posterior_belief_log_partition_bound_optional() -> None:
    b = PosteriorBelief(
        p_true=0.7,
        p_false=0.3,
        converged=True,
        algorithm="TRW-BP",
        iterations=5,
        edge_count=3,
        log_partition_bound=-1.234,
    )
    assert b.log_partition_bound == pytest.approx(-1.234)

    b_lbp = PosteriorBelief(
        p_true=0.7,
        p_false=0.3,
        converged=True,
        algorithm="LBP-fallback",
        iterations=5,
        edge_count=3,
    )
    assert b_lbp.log_partition_bound is None
