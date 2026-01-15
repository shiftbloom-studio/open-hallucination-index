"""API response schemas - re-exported from routes for shared use."""

from uuid import UUID

from pydantic import BaseModel, Field

from open_hallucination_index.domain.results import TrustScore, VerificationStatus, CitationTrace


class ClaimSummary(BaseModel):
    """Summarized claim for API response."""

    id: UUID
    text: str
    status: VerificationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    trace: CitationTrace | None = None


class VerifyTextResponse(BaseModel):
    """Response from text verification endpoint."""

    id: UUID
    trust_score: TrustScore
    summary: str | None
    claims: list[ClaimSummary]
    processing_time_ms: float
    cached: bool


class BatchVerifyResponse(BaseModel):
    """Response from batch verification."""

    results: list[VerifyTextResponse]
    total_processing_time_ms: float
