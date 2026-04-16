"""Feedback domain models for the v2 active-learning loop. Spec §10 + §12."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Public submission types (shape the /feedback endpoint accepts)
# ---------------------------------------------------------------------------


FeedbackLabel = Literal["true", "false", "unverifiable", "abstain"]
LabelerKind = Literal["user", "expert", "adjudicator"]
EvidenceCorrectionLabel = Literal["supports", "refutes", "irrelevant"]


class EvidenceCorrection(BaseModel):
    evidence_id: str
    correct_classification: EvidenceCorrectionLabel

    model_config = {"frozen": True}


class FeedbackSubmission(BaseModel):
    """Inbound feedback payload. `ip_hash` is populated server-side from
    a salted sha256 of the caller's address, never the raw IP."""

    request_id: UUID
    claim_id: UUID
    label: FeedbackLabel
    labeler_kind: LabelerKind
    labeler_id_hash: str = Field(..., min_length=8)
    rationale: str | None = None
    evidence_corrections: list[EvidenceCorrection] = Field(default_factory=list)
    ip_hash: str | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Stored types (what the Postgres adapter returns)
# ---------------------------------------------------------------------------


CalibrationSourceTier = Literal["consensus", "trusted", "adjudicator"]


class CalibrationEntry(BaseModel):
    """One row in the calibration_set table. See spec §12.

    The partition string is `"<domain>:<claim_type>"` (e.g.
    `"biomedical:quantitative"`) so Mondrian stratification in L5 can
    fetch entries for a specific stratum with a single indexed query.
    """

    id: UUID
    claim_id: UUID
    true_label: Literal["true", "false", "unverifiable"]
    source_tier: CalibrationSourceTier
    n_concordant: int = Field(..., ge=1)
    calibration_set_partition: str
    posterior_at_label_time: float = Field(..., ge=0.0, le=1.0)
    model_versions_at_label_time: dict[str, str]
    created_at: datetime
    retired_at: datetime | None = None

    model_config = {"frozen": True}


class DisputedClaim(BaseModel):
    """Disputed claims queue entry (spec §12 — disputed_claims_queue)."""

    claim_id: UUID
    first_disputed_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Opaque identifiers
# ---------------------------------------------------------------------------


# Explicit alias for readability — the Postgres adapter returns a UUID
# string, and callers treat it as opaque.
FeedbackId = str
