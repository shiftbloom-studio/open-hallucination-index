"""Frozen value objects for L4 Probabilistic Claim Graph. See spec §6."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BPAlgorithm = Literal["TRW-BP", "LBP-fallback", "LBP-nonconvergent"]


class PosteriorBelief(BaseModel):
    """Per-node posterior after joint inference on the claim graph.

    `log_partition_bound` is only meaningful when `algorithm == 'TRW-BP'`
    (tree-reweighted BP gives an explicit upper bound on log Z). LBP
    fallbacks set it to None.
    """

    p_true: float = Field(..., ge=0.0, le=1.0)
    p_false: float = Field(..., ge=0.0, le=1.0)
    converged: bool
    algorithm: BPAlgorithm
    iterations: int = Field(..., ge=0)
    edge_count: int = Field(..., ge=0)
    log_partition_bound: float | None = None

    model_config = {"frozen": True}
