"""L1 → L2 → L3 → L4 → L5 → L6 → L7 orchestrator. Spec §2.

The single place where the v2 pipeline ordering is encoded. Each layer
is invoked through its port; Phase 1 supplies ``None`` for layers not
yet implemented, and the orchestrator falls back to a clearly-marked
placeholder that preserves the end-to-end return shape.

Phase 1 placeholders:
  * L2 Domain router → always routes to ``"general"`` with soft=False.
  * L3 NLI → skipped (empty NLI matrices).
  * L4 PCG → posterior per claim = naive evidence-support mean.
  * L6 information-gain hook → fixed IG=0, never queues for review.

When Phase 2 (NLI + PCG) and Phase 3 (domain router) land, their
concrete adapters swap in through the constructor; the Phase 1
placeholders get removed and the public ``verify`` signature stays the
same.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from interfaces.conformal import ConformalCalibrator
from interfaces.decomposition import ClaimDecomposer
from models.domain import Domain, DomainAssignment
from models.entities import Claim, Evidence
from models.pcg import PosteriorBelief
from pipeline.assembly import (
    assemble_claim_verdict,
    assemble_document_verdict,
)

if TYPE_CHECKING:
    from interfaces.domain import DomainAdapter, DomainRouter
    from interfaces.nli import NLIService
    from interfaces.pcg import PCGInferenceService
    from models.results import DocumentVerdict
    from pipeline.retrieval import AdaptiveEvidenceCollector


logger = logging.getLogger(__name__)


RigorTier = Literal["fast", "balanced", "maximum"]


class Pipeline:
    """Stateless orchestrator — one instance per app, no per-request state.

    Concrete adapters are passed at construction time; missing layers
    (Phase 1/2/3/4 rollout) are represented by ``None`` and handled via
    placeholder logic marked with ``PLACEHOLDER`` log lines.
    """

    def __init__(
        self,
        *,
        decomposer: ClaimDecomposer,
        retrieval: AdaptiveEvidenceCollector | None,
        conformal: ConformalCalibrator,
        domain_router: DomainRouter | None = None,
        nli: NLIService | None = None,
        pcg: PCGInferenceService | None = None,
        domain_adapters: dict[Domain, DomainAdapter] | None = None,
    ) -> None:
        self._decomposer = decomposer
        self._retrieval = retrieval
        self._conformal = conformal
        self._router = domain_router
        self._nli = nli
        self._pcg = pcg
        self._domain_adapters = domain_adapters or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        text: str,
        *,
        context: str | None = None,
        domain_hint: Domain | None = None,
        rigor: RigorTier = "balanced",
        retrieval_tier: Literal["local", "default", "max"] = "default",
        max_claims: int = 50,
        request_id: UUID | None = None,
    ) -> DocumentVerdict:
        """Run the full L1→L7 pipeline on a single text, return a DocumentVerdict."""
        del domain_hint  # Phase 1: domain hint honored in Phase 3 via the router
        request_id = request_id or uuid4()
        t0 = time.perf_counter()

        # L1 — decomposition
        claims = await self._decomposer.decompose(text)
        logger.debug("request %s: decomposed into %d claims", request_id, len(claims))

        # L1 — evidence retrieval per claim (placeholder when no collector)
        evidence_per_claim: dict[UUID, list[Evidence]] = {}
        if self._retrieval is not None and claims:
            for claim in claims:
                try:
                    # AdaptiveEvidenceCollector.collect returns a richer
                    # structure; Phase 1 only needs the raw Evidence list.
                    evidence_per_claim[claim.id] = await self._retrieve_evidence(claim)
                except Exception as exc:
                    logger.warning("L1 retrieval failed for claim %s: %s", claim.id, exc)
                    evidence_per_claim[claim.id] = []
        else:
            for claim in claims:
                evidence_per_claim[claim.id] = []

        # L2 — domain routing (placeholder: always general)
        assignments = await self._route_claims(claims)

        # L3 / L4 — NLI + PCG joint inference (placeholder: naive posterior)
        posteriors = await self._compute_posteriors(claims, evidence_per_claim, assignments)

        # L5 — conformal calibration per claim
        claim_verdicts = []
        for claim in claims:
            assignment = assignments[claim.id]
            primary = assignment.primary
            belief = posteriors[claim.id]
            stratum = f"{primary}:{claim.claim_type.value}"
            cal = await self._conformal.calibrate(
                claim=claim, belief=belief, domain=primary, stratum=stratum
            )

            # L6 — information-gain hook (placeholder: IG=0, never queue)
            information_gain = 0.0
            queued_for_review = False

            cv = assemble_claim_verdict(
                claim=claim,
                p_true=cal.p_true,
                interval=(cal.interval_lower, cal.interval_upper),
                coverage_target=cal.coverage_target,
                domain=primary,
                domain_assignment_weights={d: w for d, w in assignment.weights.items()},
                supporting_evidence=evidence_per_claim[claim.id],
                refuting_evidence=[],  # Phase 2: classified by NLI
                pcg_neighbors=[],  # Phase 2: populated from PCG
                nli_self_consistency_variance=0.0,
                bp_validated=None,
                information_gain=information_gain,
                queued_for_review=queued_for_review,
                calibration_set_id=cal.calibration_set_id,
                calibration_n=cal.calibration_n,
                fallback_used=cal.fallback_used,
            )
            claim_verdicts.append(cv)

        # L7 — document assembly
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        decomp_coverage = self._compute_decomp_coverage(text, claims)
        document_verdict = assemble_document_verdict(
            claim_verdicts,
            correlation_matrix_R=None,  # Phase 2: from PCG claim-claim NLI
            internal_consistency=1.0,  # Phase 2: from PCG KL divergence
            decomposition_coverage=decomp_coverage,
            processing_time_ms=elapsed_ms,
            rigor=rigor,
            refinement_passes_executed=0,  # Phase 2 iterative refinement
            model_versions=self._model_versions(),
            request_id=request_id,
        )
        logger.info(
            "request %s: %d claims, score=%.3f, %.1f ms",
            request_id,
            len(claims),
            document_verdict.document_score,
            elapsed_ms,
        )
        return document_verdict

    # ------------------------------------------------------------------
    # Internal helpers (Phase 1 placeholders)
    # ------------------------------------------------------------------

    async def _retrieve_evidence(self, claim: Claim) -> list[Evidence]:
        """Wrap AdaptiveEvidenceCollector.collect to surface a flat list.

        Phase 1: we trust the existing v1 collector to do the work; the
        claim's domain adapter is None at this stage so source-
        credibility overrides don't apply (defaults are fine).
        """
        assert self._retrieval is not None
        result = await self._retrieval.collect(claim, mcp_sources=None)
        # The collector's return shape exposes the raw evidence list.
        # If the adapter's public surface evolves, update here.
        if hasattr(result, "evidence"):
            return list(result.evidence)
        if isinstance(result, list):
            return list(result)
        logger.warning(
            "PLACEHOLDER: unknown collector return shape %s; returning []",
            type(result).__name__,
        )
        return []

    async def _route_claims(self, claims: list[Claim]) -> dict[UUID, DomainAssignment]:
        """Phase 1 placeholder: route every claim to 'general' with weight=1."""
        if self._router is not None:
            out: dict[UUID, DomainAssignment] = {}
            for claim in claims:
                out[claim.id] = await self._router.route(claim)
            return out

        logger.debug("PLACEHOLDER: L2 domain router unset, using 'general' fallback")
        default = DomainAssignment(weights={"general": 1.0}, primary="general", soft=False)
        return {claim.id: default for claim in claims}

    async def _compute_posteriors(
        self,
        claims: list[Claim],
        evidence_per_claim: dict[UUID, list[Evidence]],
        assignments: dict[UUID, DomainAssignment],
    ) -> dict[UUID, PosteriorBelief]:
        """Phase 1 placeholder: posterior = mean evidence similarity score.

        When NLI + PCG are wired (Task 2.8), this method swaps for a
        real call to ``self._pcg.infer(...)``. The placeholder keeps
        the pipeline end-to-end functional so /verify returns valid
        DocumentVerdicts during Phase 1 rollout.
        """
        del assignments  # Phase 1: domain doesn't influence the naive posterior
        if self._nli is not None and self._pcg is not None:
            # Full path will land in Task 2.8. Import here to avoid
            # mandatory NLI/PCG imports at module load.
            logger.debug("Running NLI + PCG joint inference")
            # Placeholder return: Task 2.8 wires the real flow.
            raise NotImplementedError(
                "NLI + PCG wiring lands in Task 2.8; currently only "
                "the Phase 1 placeholder posterior is active."
            )

        out: dict[UUID, PosteriorBelief] = {}
        for claim in claims:
            evidence = evidence_per_claim[claim.id]
            if not evidence:
                # No evidence → prior of 0.5 (maximum uncertainty)
                p_true = 0.5
            else:
                similarities = [
                    ev.similarity_score for ev in evidence if ev.similarity_score is not None
                ]
                p_true = sum(similarities) / len(similarities) if similarities else 0.5
            out[claim.id] = PosteriorBelief(
                p_true=p_true,
                p_false=1.0 - p_true,
                converged=True,
                algorithm="TRW-BP",  # placeholder label — no actual BP yet
                iterations=0,
                edge_count=0,
            )
        return out

    @staticmethod
    def _compute_decomp_coverage(text: str, claims: list[Claim]) -> float:
        """Rough heuristic: ratio of claims to sentence-like units."""
        if not text.strip():
            return 0.0
        # Count sentence-terminators as a cheap proxy for "sentence count"
        import re

        sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
        ratio = len(claims) / sentence_count
        return float(min(1.0, ratio))

    def _model_versions(self) -> dict[str, str]:
        """Collect model version strings from the layers. Phase 1 stubs
        return placeholder tags; real versions come through in later
        phases as each layer is wired."""
        versions: dict[str, str] = {
            "decomposer": getattr(self._decomposer, "model_id", "phase1-default"),
            "domain_router": "phase1-placeholder-general",
            "pcg": "phase1-placeholder-mean-similarity",
            "conformal": "phase1-split-conformal-stub",
        }
        if self._nli is not None:
            versions["nli"] = getattr(self._nli, "model_id", "unknown")
        return versions
