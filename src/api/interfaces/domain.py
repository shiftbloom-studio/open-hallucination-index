"""L2 Domain Router + DomainAdapter ports. Spec §4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from models.domain import Domain, DomainAssignment
    from models.entities import Claim


@runtime_checkable
class DomainAdapter(Protocol):
    """Per-domain configuration bundle.

    Each of the 5 domain adapters (general / biomedical / legal / code /
    social) implements this port. The facade returned by a soft-assignment
    router also implements it by blending the underlying adapters.
    """

    @property
    def domain(self) -> Domain: ...

    def nli_model_id(self) -> str:
        """Checkpoint ID for this domain's fine-tuned NLI head."""

    def source_credibility(self) -> dict[str, float]:
        """Per-source credibility overrides on the global prior table.

        Keys are source names (e.g. ``"pubmed"``, ``"wikipedia_general"``);
        values are credibility weights in [0, 1]. Unspecified sources fall
        back to :mod:`pipeline.retrieval.source_credibility.DEFAULT_PRIORS`.
        """

    def calibration_set_id(self) -> str:
        """Opaque identifier for this domain's conformal calibration set
        (e.g. ``"biomedical:any"``). Used by L5 to fetch quantiles."""

    def decomposition_hints(self) -> str | None:
        """Optional additional system-prompt content for the L1 decomposer.

        For example, a medical adapter might add: "Treat drug names,
        dosages, and study results as atomic claims."
        """

    def claim_pair_relatedness_threshold(self) -> float:
        """Bi-encoder cosine threshold for pruning claim-pair NLI.

        Default 0.45; `social` and `legal` use 0.30 because contradictions
        in those domains are often lexically dissimilar.
        """


@runtime_checkable
class DomainRouter(Protocol):
    """Classifies each claim into one of the 5 domains (soft-assigned)."""

    async def route(self, claim: Claim) -> DomainAssignment: ...
