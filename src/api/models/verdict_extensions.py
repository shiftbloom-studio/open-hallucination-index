"""Wave 3 Stream P — verdict JSON extensions for PCG observability.

Adds a ``pcg`` sub-object on each ``ClaimVerdict`` surfacing the
probabilistic-claim-graph provenance so consumers (frontend BP badge,
integration tests, FEVER slice grader) can tell whether TRW-BP
converged, whether the damped LBP fallback fired, whether Gibbs sanity
flagged a mismatch, and how often the claim-claim NLI OpenAI primary
fell back to Gemini.

See ``docs/superpowers/specs/2026-04-18-wave3-pcg.md`` §4.3 for the
contract. Frozen-model discipline matches the rest of ``models/``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.pcg import BPAlgorithm


class PCGObservability(BaseModel):
    """Per-claim provenance of the L4 PCG inference pass.

    All fields are authoritative facts about how the posterior was
    produced — no derived metrics, no humanised labels. Consumers map
    to UI copy (e.g. ``LBP-nonconvergent`` → "approximate verdict").

    ``log_partition_bound`` is only meaningful when
    ``algorithm == "TRW-BP"``; the LBP and nonconvergent paths leave it
    ``None``. ``gibbs_mismatch`` is ``None`` unless the Gibbs sanity
    pass disagreed with the BP marginals beyond
    ``PCG_GIBBS_TOLERANCE``; in that case it carries the max absolute
    marginal delta observed, so downstream analysis can rank
    severity. ``cc_nli_fallback_fired_count`` is the number of
    claim-claim pairs whose OpenAI primary failed and whose Gemini
    fallback was invoked during this verify — surfaces cost + signal
    drift concerns.
    """

    algorithm: BPAlgorithm
    converged: bool
    iterations: int = Field(..., ge=0)
    edge_count: int = Field(..., ge=0)
    log_partition_bound: float | None = None
    gibbs_mismatch: float | None = Field(default=None, ge=0.0)
    cc_nli_fallback_fired_count: int = Field(default=0, ge=0)

    model_config = {"frozen": True}
