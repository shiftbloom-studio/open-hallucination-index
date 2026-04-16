"""Request schemas for the /verify endpoint. Spec §10."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from models.domain import Domain


class VerifyOptions(BaseModel):
    """Per-request knobs that flow into the pipeline orchestrator."""

    rigor: Literal["fast", "balanced", "maximum"] = "balanced"
    tier: Literal["local", "default", "max"] = "default"
    max_claims: int = Field(default=50, ge=1, le=100)
    include_pcg_neighbors: bool = True
    include_full_provenance: bool = True
    self_consistency_k: int | None = Field(default=None, ge=1, le=50)
    coverage_target: float = Field(default=0.90, ge=0.5, le=0.99)

    model_config = {"extra": "forbid"}


class VerifyRequest(BaseModel):
    """POST /api/v2/verify request body. Spec §10."""

    text: str = Field(..., min_length=0, max_length=50_000)
    context: str | None = Field(default=None, max_length=2_000)
    domain_hint: Domain | None = None
    options: VerifyOptions = Field(default_factory=VerifyOptions)
    request_id: UUID | None = None

    model_config = {"extra": "forbid"}
