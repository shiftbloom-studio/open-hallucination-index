"""
v2 Result Models
================

Frozen, immutable Pydantic models for the v2 verification API. Replaces
the v1 VerificationResult / TrustScore / ClaimVerification trio.

See docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md §9.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from models.entities import Claim, Evidence

# ---------------------------------------------------------------------------
# Edge types in the Probabilistic Claim Graph (Phase 2 fills these in;
# Phase 1 emits empty pcg_neighbors lists).
# ---------------------------------------------------------------------------


class EdgeType(StrEnum):
    ENTAIL = auto()
    CONTRADICT = auto()


class ClaimEdge(BaseModel):
    neighbor_claim_id: UUID
    edge_type: EdgeType
    edge_strength: float = Field(..., ge=0.0, le=1.0)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# ClaimVerdict — the per-claim public output. Spec §9.
# ---------------------------------------------------------------------------


FallbackUsed = Literal["domain", "general", "non_converged"]


class ClaimVerdict(BaseModel):
    claim: Claim

    # Calibrated probability + conformal interval
    p_true: float = Field(..., ge=0.0, le=1.0)
    interval: tuple[float, float]
    coverage_target: float | None = Field(default=None, ge=0.0, le=1.0)

    # Provenance & explainability
    domain: str
    domain_assignment_weights: dict[str, float]
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    refuting_evidence: list[Evidence] = Field(default_factory=list)
    pcg_neighbors: list[ClaimEdge] = Field(default_factory=list)
    nli_self_consistency_variance: float = Field(..., ge=0.0)
    bp_validated: bool | None = None  # None when Gibbs skipped (benign graph)

    # Active learning
    information_gain: float = Field(..., ge=0.0)
    queued_for_review: bool = False

    # Calibration metadata
    calibration_set_id: str | None = None
    calibration_n: int = Field(..., ge=0)
    fallback_used: FallbackUsed | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_interval(self) -> ClaimVerdict:
        lo, hi = self.interval
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"interval must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
        return self


# ---------------------------------------------------------------------------
# DocumentVerdict — top-level public output. Spec §9.
# ---------------------------------------------------------------------------


Rigor = Literal["fast", "balanced", "maximum"]


class DocumentVerdict(BaseModel):
    document_score: float = Field(..., ge=0.0, le=1.0)
    document_interval: tuple[float, float]
    internal_consistency: float = Field(..., ge=0.0, le=1.0)
    claims: list[ClaimVerdict] = Field(default_factory=list)
    decomposition_coverage: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float = Field(..., ge=0.0)
    rigor: Rigor
    refinement_passes_executed: int = Field(..., ge=0)
    pipeline_version: str = Field(default="ohi-v2.0")
    model_versions: dict[str, str] = Field(default_factory=dict)
    request_id: UUID

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_doc_interval(self) -> DocumentVerdict:
        lo, hi = self.document_interval
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"document_interval must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
        return self
