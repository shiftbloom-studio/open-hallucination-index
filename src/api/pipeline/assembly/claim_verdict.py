"""Claim-level output assembly. Spec §9."""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.results import ClaimEdge, ClaimVerdict

if TYPE_CHECKING:
    from models.entities import Claim, Evidence
    from models.results import FallbackUsed
    from models.verdict_extensions import PCGObservability


def assemble_claim_verdict(
    claim: Claim,
    *,
    p_true: float,
    interval: tuple[float, float],
    coverage_target: float | None,
    domain: str,
    domain_assignment_weights: dict[str, float],
    supporting_evidence: list[Evidence],
    refuting_evidence: list[Evidence],
    pcg_neighbors: list[ClaimEdge],
    pcg: "PCGObservability | None" = None,
    nli_self_consistency_variance: float,
    bp_validated: bool | None,
    information_gain: float,
    queued_for_review: bool,
    calibration_set_id: str | None,
    calibration_n: int,
    fallback_used: FallbackUsed | None,
) -> ClaimVerdict:
    """Construct a frozen public ``ClaimVerdict``.

    Thin wrapper — kept as a function (rather than inline Pydantic
    construction at call sites) so future field additions only need to
    touch this one place, and so the pipeline orchestrator doesn't need
    to know the ClaimVerdict constructor signature directly.

    All validation (interval ordering, probability ranges, fallback
    enum) happens inside the Pydantic model.
    """
    return ClaimVerdict(
        claim=claim,
        p_true=p_true,
        interval=interval,
        coverage_target=coverage_target,
        domain=domain,
        domain_assignment_weights=domain_assignment_weights,
        supporting_evidence=supporting_evidence,
        refuting_evidence=refuting_evidence,
        pcg_neighbors=pcg_neighbors,
        pcg=pcg,
        nli_self_consistency_variance=nli_self_consistency_variance,
        bp_validated=bp_validated,
        information_gain=information_gain,
        queued_for_review=queued_for_review,
        calibration_set_id=calibration_set_id,
        calibration_n=calibration_n,
        fallback_used=fallback_used,
    )
