"""L3 NLI cross-encoder port. Spec §5."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from interfaces.domain import DomainAdapter
    from models.entities import Claim, Evidence
    from models.nli import NLIDistribution


RigorTier = Literal["fast", "balanced", "maximum"]


@runtime_checkable
class NLIService(Protocol):
    """Produces calibrated NLI distributions for (claim, evidence) and
    (claim, claim) pairs.

    Concrete implementations handle batching, temperature scaling, and
    (in Phase 3) per-domain head selection via ``adapter.nli_model_id()``.
    The base cross-encoder (Phase 2) ignores the adapter; the wrapping
    self-consistency layer (Task 2.2) does K stochastic passes.
    """

    async def claim_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence],
        adapter: DomainAdapter,
        *,
        rigor: RigorTier = "balanced",
    ) -> list[NLIDistribution]:
        """NLI distribution per (claim, evidence_i) pair, in input order."""

    async def claim_claim(
        self,
        claims: list[Claim],
        adapter: DomainAdapter,
        *,
        rigor: RigorTier = "balanced",
    ) -> dict[tuple[UUID, UUID], NLIDistribution]:
        """NLI distribution per unordered (claim_i, claim_j) pair.

        Keys are ordered tuples (id_i, id_j) with id_i < id_j lexically;
        DeBERTa-MNLI is asymmetric so callers of the PCG layer use both
        directions by re-indexing when needed.
        """

    async def health_check(self) -> bool: ...
