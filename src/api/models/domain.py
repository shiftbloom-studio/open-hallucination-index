"""Domain enum + assignment value object. See spec §4."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

Domain = Literal["general", "biomedical", "legal", "code", "social"]

ALL_DOMAINS: tuple[Domain, ...] = ("general", "biomedical", "legal", "code", "social")


class DomainAssignment(BaseModel):
    """Per-claim routing output from L2.

    `weights` is a full distribution over the 5 domains summing to ~1.
    `primary` is the argmax. `soft` is True when the top-1 and top-2
    weights are within 0.15 of each other — triggers mixture conformal
    in L5 and NLI-head mixing in L3 (Phase 3).
    """

    weights: dict[Domain, float]
    primary: Domain
    soft: bool

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _sums_to_one(self) -> DomainAssignment:
        s = sum(self.weights.values())
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"weights must sum to ~1, got {s}")
        return self
