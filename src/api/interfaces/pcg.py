"""L4 Probabilistic Claim Graph inference port. Spec §6."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from interfaces.domain import DomainAdapter
    from models.entities import Claim, Evidence
    from models.nli import NLIDistribution
    from models.pcg import PosteriorBelief


RigorTier = Literal["fast", "balanced", "maximum"]


@runtime_checkable
class PCGInferenceService(Protocol):
    """Joint inference over the Ising-style claim graph.

    Builds the graph from L3 NLI outputs and evidence weights, then runs
    TRW-BP as primary with damped LBP as fallback; in ``balanced``+ rigor
    tiers a Gibbs MCMC sanity check validates the marginals (see §6).
    """

    async def infer(
        self,
        claims: list[Claim],
        evidence_per_claim: dict[UUID, list[Evidence]],
        nli_claim_evidence: dict[UUID, list[NLIDistribution]],
        nli_claim_claim: dict[tuple[UUID, UUID], NLIDistribution],
        adapter_per_claim: dict[UUID, DomainAdapter],
        *,
        rigor: RigorTier = "balanced",
    ) -> dict[UUID, PosteriorBelief]:
        """Returns posterior belief per claim, keyed by ``claim.id``."""
