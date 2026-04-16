"""Integration tests for pipeline.assembly.document_verdict."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

_SRC_API = Path(__file__).resolve().parents[4] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from models.entities import Claim, ClaimType  # noqa: E402
from models.results import ClaimVerdict  # noqa: E402
from pipeline.assembly.claim_verdict import assemble_claim_verdict  # noqa: E402
from pipeline.assembly.document_verdict import assemble_document_verdict  # noqa: E402


def _make_cv(p_true: float, interval: tuple[float, float] = (0.0, 1.0)) -> ClaimVerdict:
    """Minimal ClaimVerdict helper."""
    return assemble_claim_verdict(
        claim=Claim(id=uuid4(), text="test", claim_type=ClaimType.UNCLASSIFIED),
        p_true=p_true,
        interval=interval,
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


# ---------------------------------------------------------------------------
# Empty / trivial cases
# ---------------------------------------------------------------------------


def test_empty_document_returns_perfect_trust() -> None:
    dv = assemble_document_verdict(
        [],
        correlation_matrix_R=None,
        internal_consistency=1.0,
        decomposition_coverage=0.0,
        processing_time_ms=1.0,
        rigor="balanced",
        refinement_passes_executed=0,
        model_versions={},
        request_id=uuid4(),
    )
    assert dv.document_score == 1.0
    assert dv.document_interval == (1.0, 1.0)
    assert dv.claims == []


def test_single_claim_document_score_equals_marginal() -> None:
    cv = _make_cv(p_true=0.7, interval=(0.5, 0.9))
    dv = assemble_document_verdict(
        [cv],
        correlation_matrix_R=None,
        internal_consistency=1.0,
        decomposition_coverage=1.0,
        processing_time_ms=10.0,
        rigor="fast",
        refinement_passes_executed=0,
        model_versions={"decomp": "v1"},
        request_id=uuid4(),
    )
    assert dv.document_score == pytest.approx(0.7)
    assert dv.document_interval == pytest.approx((0.5, 0.9))


def test_two_independent_claims_multiplies_probs() -> None:
    cvs = [_make_cv(0.8, (0.7, 0.9)), _make_cv(0.6, (0.5, 0.7))]
    dv = assemble_document_verdict(
        cvs,
        correlation_matrix_R=None,
        internal_consistency=1.0,
        decomposition_coverage=1.0,
        processing_time_ms=50.0,
        rigor="balanced",
        refinement_passes_executed=0,
        model_versions={},
        request_id=uuid4(),
    )
    # Identity correlation → product of marginals
    assert dv.document_score == pytest.approx(0.8 * 0.6, abs=1e-9)
    # Interval bounds also multiply under the copula
    assert dv.document_interval[0] == pytest.approx(0.7 * 0.5, abs=1e-9)
    assert dv.document_interval[1] == pytest.approx(0.9 * 0.7, abs=1e-9)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_document_interval_wraps_score() -> None:
    """For any document, lower <= score <= upper after assembly clamp."""
    cvs = [_make_cv(0.5, (0.3, 0.7)) for _ in range(4)]
    dv = assemble_document_verdict(
        cvs,
        correlation_matrix_R=None,
        internal_consistency=0.8,
        decomposition_coverage=1.0,
        processing_time_ms=120.0,
        rigor="balanced",
        refinement_passes_executed=0,
        model_versions={},
        request_id=uuid4(),
    )
    lo, hi = dv.document_interval
    assert lo <= dv.document_score <= hi, (
        f"Invariant broken: lo={lo}, score={dv.document_score}, hi={hi}"
    )


def test_explicit_identity_matrix_matches_none() -> None:
    """Passing None for correlation matrix == passing identity."""
    cvs = [_make_cv(0.5), _make_cv(0.5), _make_cv(0.5)]
    dv_none = assemble_document_verdict(
        cvs,
        correlation_matrix_R=None,
        internal_consistency=1.0,
        decomposition_coverage=1.0,
        processing_time_ms=10.0,
        rigor="fast",
        refinement_passes_executed=0,
        model_versions={},
        request_id=uuid4(),
    )
    dv_eye = assemble_document_verdict(
        cvs,
        correlation_matrix_R=np.eye(3),
        internal_consistency=1.0,
        decomposition_coverage=1.0,
        processing_time_ms=10.0,
        rigor="fast",
        refinement_passes_executed=0,
        model_versions={},
        request_id=uuid4(),
    )
    assert dv_none.document_score == pytest.approx(dv_eye.document_score, abs=1e-9)


def test_model_versions_and_rigor_preserved() -> None:
    cvs = [_make_cv(0.9)]
    dv = assemble_document_verdict(
        cvs,
        correlation_matrix_R=None,
        internal_consistency=1.0,
        decomposition_coverage=1.0,
        processing_time_ms=0.0,
        rigor="maximum",
        refinement_passes_executed=3,
        model_versions={"decomposer": "deb-v3", "router": "n/a"},
        request_id=uuid4(),
    )
    assert dv.rigor == "maximum"
    assert dv.refinement_passes_executed == 3
    assert dv.model_versions == {"decomposer": "deb-v3", "router": "n/a"}
    assert dv.pipeline_version == "ohi-v2.0"
