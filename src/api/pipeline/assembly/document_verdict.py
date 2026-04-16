"""Document-level output assembly. Spec §9.

Takes a list of per-claim ``ClaimVerdict`` objects and produces the
public ``DocumentVerdict`` with copula-aggregated joint probability,
conformal-derived interval bounds, and full model-version provenance.
"""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

import numpy as np

from models.results import ClaimVerdict, DocumentVerdict
from pipeline.assembly.copula import (
    build_correlation_matrix_identity,
    gaussian_copula_joint,
)

logger = logging.getLogger(__name__)


def assemble_document_verdict(
    claim_verdicts: list[ClaimVerdict],
    *,
    correlation_matrix_R: np.ndarray | None,
    internal_consistency: float,
    decomposition_coverage: float,
    processing_time_ms: float,
    rigor: Literal["fast", "balanced", "maximum"],
    refinement_passes_executed: int,
    model_versions: dict[str, str],
    request_id: UUID,
) -> DocumentVerdict:
    """Aggregate claim verdicts into the public document verdict.

    When ``correlation_matrix_R`` is None, the identity matrix is used
    — appropriate for Phase 1 (no claim↔claim edges available). The
    copula aggregator returns ``∏_i p_i`` exactly in that case.

    Intervals are aggregated by applying the same copula construction to
    ``interval_lower`` and ``interval_upper`` sequences. Monotonicity of
    the multivariate CDF makes this a valid lower/upper bound on the
    joint probability under the copula model.
    """
    n = len(claim_verdicts)

    if n == 0:
        # Empty document — nothing to verify, max trust, coherent, fast.
        return DocumentVerdict(
            document_score=1.0,
            document_interval=(1.0, 1.0),
            internal_consistency=internal_consistency if n else 1.0,
            claims=[],
            decomposition_coverage=decomposition_coverage,
            processing_time_ms=processing_time_ms,
            rigor=rigor,
            refinement_passes_executed=refinement_passes_executed,
            model_versions=model_versions,
            request_id=request_id,
        )

    r = (
        correlation_matrix_R
        if correlation_matrix_R is not None
        else build_correlation_matrix_identity(n)
    )

    p_per_claim = np.asarray([cv.p_true for cv in claim_verdicts])
    lower_per_claim = np.asarray([cv.interval[0] for cv in claim_verdicts])
    upper_per_claim = np.asarray([cv.interval[1] for cv in claim_verdicts])

    document_score = gaussian_copula_joint(p_per_claim, r)
    document_lower = gaussian_copula_joint(lower_per_claim, r)
    document_upper = gaussian_copula_joint(upper_per_claim, r)

    # Enforce ordering under numerical noise: the lower bound should
    # never exceed the upper bound, nor the score exceed either bound
    # in the degenerate independent case. Clamp gently.
    document_lower = min(document_lower, document_score, document_upper)
    document_upper = max(document_upper, document_score, document_lower)

    return DocumentVerdict(
        document_score=float(document_score),
        document_interval=(float(document_lower), float(document_upper)),
        internal_consistency=internal_consistency,
        claims=claim_verdicts,
        decomposition_coverage=decomposition_coverage,
        processing_time_ms=processing_time_ms,
        rigor=rigor,
        refinement_passes_executed=refinement_passes_executed,
        model_versions=model_versions,
        request_id=request_id,
    )
