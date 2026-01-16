"""
Domain Results
==============

Value objects representing verification outcomes, scores, and traces.
These flow from the verification pipeline to the API response.
"""

from __future__ import annotations

from enum import StrEnum, auto
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from models.entities import Claim, Evidence


class EvidenceClassification(StrEnum):
    """Granular evidence classification with confidence levels."""

    STRONG_SUPPORT = auto()  # 0.9 - Evidence directly confirms claim
    WEAK_SUPPORT = auto()  # 0.7 - Evidence provides contextual support
    NEUTRAL = auto()  # 0.5 - Evidence is unrelated or ambiguous
    WEAK_REFUTE = auto()  # 0.3 - Evidence suggests claim might be false
    STRONG_REFUTE = auto()  # 0.1 - Evidence directly contradicts claim

    def to_confidence(self) -> float:
        """Convert classification to confidence score."""
        mapping = {
            EvidenceClassification.STRONG_SUPPORT: 0.9,
            EvidenceClassification.WEAK_SUPPORT: 0.7,
            EvidenceClassification.NEUTRAL: 0.5,
            EvidenceClassification.WEAK_REFUTE: 0.3,
            EvidenceClassification.STRONG_REFUTE: 0.1,
        }
        return mapping[self]


class VerificationStatus(StrEnum):
    """Outcome of verifying a single claim."""

    SUPPORTED = auto()  # Evidence strongly supports the claim
    REFUTED = auto()  # Evidence contradicts the claim
    PARTIALLY_SUPPORTED = auto()  # Mixed or partial evidence
    UNVERIFIABLE = auto()  # No relevant evidence found
    UNCERTAIN = auto()  # Conflicting evidence, cannot determine


class CitationTrace(BaseModel):
    """
    Provenance trail explaining a verification decision.

    Provides transparency into how a claim was verified,
    enabling users to inspect the reasoning and sources.
    """

    claim_id: UUID
    status: VerificationStatus
    reasoning: str = Field(..., description="Human-readable explanation")

    # Evidence chain
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    refuting_evidence: list[Evidence] = Field(default_factory=list)

    # Confidence in this specific verification
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in verification outcome")

    # Strategy used
    verification_strategy: str = Field(
        ..., description="Strategy that produced this result (graph, vector, hybrid)"
    )

    model_config = {"frozen": True}


class ClaimVerification(BaseModel):
    """
    Complete verification result for a single claim.

    Bundles the original claim with its verification outcome
    and supporting citation trace.
    """

    claim: Claim
    status: VerificationStatus
    trace: CitationTrace

    # Per-claim score contribution
    score_contribution: float = Field(
        ..., ge=0.0, le=1.0, description="This claim's contribution to overall score"
    )

    model_config = {"frozen": True}


class TrustScore(BaseModel):
    """
    Aggregated trust assessment for a piece of text.

    Combines individual claim scores into a holistic measure
    with breakdown for interpretability.
    """

    # Global score
    overall: float = Field(
        ..., ge=0.0, le=1.0, description="Global trust score (0=untrusted, 1=fully trusted)"
    )

    # Breakdown
    claims_total: int = Field(..., ge=0)
    claims_supported: int = Field(default=0, ge=0)
    claims_refuted: int = Field(default=0, ge=0)
    claims_unverifiable: int = Field(default=0, ge=0)

    # Confidence in the score itself
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Meta-confidence in score reliability"
    )

    # Weighting info
    scoring_method: str = Field(
        default="weighted_average", description="Algorithm used for aggregation"
    )

    model_config = {"frozen": True}


class VerificationResult(BaseModel):
    """
    Complete verification response for an input text.

    This is the primary output of the verification pipeline,
    containing the trust score, all claim verifications, and metadata.
    """

    id: UUID = Field(default_factory=uuid4)

    # Input reference
    input_hash: str = Field(..., description="Hash of input text for deduplication")
    input_length: int = Field(..., ge=0, description="Character count of input")

    # Results
    trust_score: TrustScore
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)

    # Summary
    summary: str | None = Field(default=None, description="Human-readable summary of findings")

    # Metadata
    processing_time_ms: float = Field(..., ge=0.0)
    cached: bool = Field(default=False, description="Whether result was from cache")

    model_config = {"frozen": True}
