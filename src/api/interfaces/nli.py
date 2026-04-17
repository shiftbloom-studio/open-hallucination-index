"""L3 NLI ports. Spec §5.

Two concrete contracts live here:

* ``NLIService`` — the cross-encoder-style port (DeBERTa-MNLI + self-
  consistency wrapper). Predates Phase 2; referenced from the pipeline
  orchestrator. Left intact for the eventual cross-encoder deployment.
* ``NliAdapter`` — the LLM-based 3-way classifier port (Phase 2 Task 2.1
  / plan §4.2). Implemented by ``adapters.nli_gemini.NliGeminiAdapter``
  against Gemini 3 Pro with ``thinkingLevel=HIGH`` and
  ``safetySettings=BLOCK_NONE``.

D1 will wire ``NliAdapter`` (not ``NLIService``) into the verify pipeline;
the two coexist so the older cross-encoder plan isn't prematurely deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from interfaces.domain import DomainAdapter
    from models.entities import Claim, Evidence
    from models.nli import NLIDistribution


RigorTier = Literal["fast", "balanced", "maximum"]

NliLabel = Literal["support", "refute", "neutral"]


@dataclass(frozen=True, slots=True)
class NliResult:
    """Output of a single NLI classification pass.

    ``supporting_score``, ``refuting_score``, ``neutral_score`` form a
    proper probability distribution (sum == 1 modulo float noise); the
    adapter renormalises before returning so downstream code can trust
    that invariant. ``confidence`` is the model's own introspective
    score, independent of the 3-way distribution. ``reasoning`` is a
    single-sentence key-fact string suitable for UI display.

    The terminal-failure fallback produced by the adapter is
    ``NliResult(label="neutral", supporting_score=0.0, refuting_score=0.0,
    neutral_score=1.0, reasoning="nli_unavailable", confidence=0.0)`` —
    callers can detect it by either ``confidence == 0.0`` or
    ``reasoning == "nli_unavailable"``.
    """

    label: NliLabel
    supporting_score: float
    refuting_score: float
    neutral_score: float
    reasoning: str
    confidence: float


@runtime_checkable
class NliAdapter(Protocol):
    """LLM-backed NLI classifier port.

    One ``classify()`` call handles one (claim, evidence) pair. Concrete
    implementations own prompting, JSON parsing, retry, and self-
    consistency sampling. Batching is the caller's responsibility —
    the pipeline gathers calls under a ``Semaphore(N)`` in Phase 2 D1.

    The adapter must never raise from ``classify``; any transport, JSON,
    or validation failure is swallowed and returned as the
    ``nli_unavailable`` neutral fallback described on :class:`NliResult`.
    """

    async def classify(
        self, claim_text: str, evidence_text: str
    ) -> NliResult: ...

    async def health_check(self) -> bool: ...


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
