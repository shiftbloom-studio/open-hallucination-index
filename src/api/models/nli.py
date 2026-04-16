"""Frozen value object for L3 NLI distributions. See spec §5."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class NLIDistribution(BaseModel):
    """Per-pair NLI categorical: entail / contradict / neutral + variance.

    The three probability fields must sum to ~1 (within 1% numerical
    tolerance). Variance is the sample variance across K self-consistency
    passes and is used as an edge-weight modulator in L4 PCG construction.
    """

    entail: float = Field(..., ge=0.0, le=1.0)
    contradict: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    variance: float = Field(..., ge=0.0)
    nli_model_id: str

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _sums_to_one(self) -> NLIDistribution:
        s = self.entail + self.contradict + self.neutral
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"entail+contradict+neutral must be ~1, got {s}")
        return self
