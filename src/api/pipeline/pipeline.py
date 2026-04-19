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

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from interfaces.conformal import ConformalCalibrator
from interfaces.decomposition import ClaimDecomposer
from models.domain import Domain, DomainAssignment
from models.entities import Claim, Evidence
from models.nli import NLIDistribution
from models.pcg import PosteriorBelief
from models.results import ClaimEdge, EdgeType
from models.verdict_extensions import PCGObservability
from pipeline.assembly import (
    assemble_claim_verdict,
    assemble_document_verdict,
)

if TYPE_CHECKING:
    from adapters.nli_claim_claim import ClaimClaimNliDispatcher
    from interfaces.domain import DomainAdapter, DomainRouter
    from interfaces.nli import NliAdapter, NliResult, NLIService
    from interfaces.pcg import PCGInferenceService
    from models.results import DocumentVerdict
    from pipeline.retrieval import AdaptiveEvidenceCollector


logger = logging.getLogger(__name__)


RigorTier = Literal["fast", "balanced", "maximum"]

# Stream D2: progress-reporter callback. The async polling handler passes
# one of these so it can advance the DynamoDB record's ``phase`` field at
# each of the five natural boundaries below. See the module docstring of
# ``src/api/server/jobs.py`` for the polling protocol.
PhaseCallback = Callable[[str], Awaitable[None]]


async def _safe_fire(callback: PhaseCallback | None, phase: str) -> None:
    """Invoke ``callback`` if present; swallow exceptions.

    Progress reporting is best-effort — a transient DynamoDB write failure
    from the polling handler MUST NOT abort an in-flight verdict. We log
    and move on; the next phase will try again, and the final
    ``complete_job`` / ``fail_job`` write still runs after verify()
    returns.
    """
    if callback is None:
        return
    try:
        await callback(phase)
    except Exception as exc:  # noqa: BLE001
        logger.warning("phase callback %r failed: %s", phase, exc)


# Maximum concurrent NLI classify() calls per /verify request. Sized so a
# realistic 5-claim × 3-evidence fan-out (15 calls) runs fully parallel
# while still capping the tail when a long document surfaces many more
# pairs. Tuned with Gemini 3 Pro's per-request latency (HIGH thinking ≈
# 10–20 s) in mind: too low and latency blows past the Lambda timeout;
# too high and we'd hit Gemini's per-project QPS ceiling under load.
_NLI_MAX_CONCURRENCY = 10


@dataclass
class _PosteriorsResult:
    """Internal return type for ``_compute_posteriors``.

    Four dicts, keyed by claim id:
    * ``posteriors`` — the per-claim ``PosteriorBelief`` (p_true etc.)
    * ``buckets`` — ``(supporting_evidence, refuting_evidence)``; only
      populated when NLI ran (Phase-2 Beta path or Wave-3 PCG path
      use the same label-driven split — Hebel A + Fix 1+3).
    * ``pcg_observability`` — per-claim PCG observability block when
      the PCG path ran; ``None`` otherwise (placeholder / Beta paths).
    * ``pcg_edges`` — per-claim list of ``ClaimEdge`` neighbours
      derived from the PCG graph's binary factors; empty for paths
      that didn't build a graph.
    """

    posteriors: dict[UUID, PosteriorBelief]
    buckets: dict[UUID, tuple[list[Evidence], list[Evidence]]]
    pcg_observability: dict[UUID, PCGObservability | None] = field(default_factory=dict)
    pcg_edges: dict[UUID, list[ClaimEdge]] = field(default_factory=dict)


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
        nli_adapter: NliAdapter | None = None,
        pcg: PCGInferenceService | None = None,
        cc_nli_dispatcher: ClaimClaimNliDispatcher | None = None,
        domain_adapters: dict[Domain, DomainAdapter] | None = None,
    ) -> None:
        # ``nli`` (NLIService) is the cross-encoder port that predates
        # Phase 2 and is still reserved for a future cross-encoder
        # deployment. ``nli_adapter`` (NliAdapter) is the Phase 2 LLM-
        # backed 3-way classifier that D1 wires; both coexist so the
        # older port isn't prematurely deleted.
        #
        # Wave 3 Stream P adds two layered precedences in
        # ``_compute_posteriors``:
        #   1. PCG path — when ``pcg`` AND ``nli_adapter`` are wired
        #      (``cc_nli_dispatcher`` is optional; if omitted the PCG
        #      path runs without a binary-factor layer, effectively
        #      LBP-on-unaries). Produces TRW-BP / LBP / LBP-nonconvergent
        #      posteriors and populates ``PCGObservability``.
        #   2. Phase-2 Beta-posterior path — when ``nli_adapter`` is
        #      wired but ``pcg`` isn't. Keeps D1 behaviour for local
        #      dev / degraded-mode deploys.
        #   3. Phase-1 placeholder — no NLI adapter at all. Used in
        #      unit tests only.
        self._decomposer = decomposer
        self._retrieval = retrieval
        self._conformal = conformal
        self._router = domain_router
        self._nli = nli
        self._nli_adapter = nli_adapter
        self._pcg = pcg
        self._cc_nli_dispatcher = cc_nli_dispatcher
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
        phase_callback: PhaseCallback | None = None,
    ) -> DocumentVerdict:
        """Run the full L1→L7 pipeline on a single text, return a DocumentVerdict.

        ``phase_callback`` (Stream D2) is invoked once BEFORE each of the
        five natural phases — decomposing, retrieving_evidence,
        classifying, calibrating, assembling — so the async polling
        handler can advance a DynamoDB ``phase`` field while the pipeline
        runs. Callback failures are logged and swallowed; see
        :func:`_safe_fire` for the exact semantic.
        """
        del domain_hint  # Phase 1: domain hint honored in Phase 3 via the router
        request_id = request_id or uuid4()
        t0 = time.perf_counter()

        # L1 — decomposition
        await _safe_fire(phase_callback, "decomposing")
        claims = await self._decomposer.decompose(text)
        logger.debug("request %s: decomposed into %d claims", request_id, len(claims))

        # L1 — evidence retrieval per claim (placeholder when no collector)
        await _safe_fire(phase_callback, "retrieving_evidence")
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

        # L2 — domain routing (placeholder: always general). Not a polling
        # boundary — L2 is a trivial dict lookup in the placeholder path
        # and too fast for the UI to render anything meaningful.
        assignments = await self._route_claims(claims)

        # L3 / L4 — NLI + PCG joint inference
        await _safe_fire(phase_callback, "classifying")
        posteriors_result = await self._compute_posteriors(
            claims, evidence_per_claim, assignments, rigor=rigor
        )
        posteriors = posteriors_result.posteriors
        nli_buckets = posteriors_result.buckets
        pcg_observability = posteriors_result.pcg_observability
        pcg_edges_by_claim = posteriors_result.pcg_edges

        # L5 — conformal calibration per claim
        await _safe_fire(phase_callback, "calibrating")
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

            supporting_bucket, refuting_bucket = nli_buckets[claim.id]
            observability = pcg_observability.get(claim.id)
            neighbours = pcg_edges_by_claim.get(claim.id, [])
            # BP validation flag: True when Gibbs ran and matched BP
            # within tolerance; False when Gibbs flagged a mismatch;
            # None when Gibbs skipped (e.g. fast rigor / placeholder path).
            if observability is None:
                bp_validated: bool | None = None
            elif observability.gibbs_mismatch is None:
                # Gibbs ran (we have a PCG block) and didn't flag a mismatch.
                bp_validated = True
            else:
                bp_validated = False

            cv = assemble_claim_verdict(
                claim=claim,
                p_true=cal.p_true,
                interval=(cal.interval_lower, cal.interval_upper),
                coverage_target=cal.coverage_target,
                domain=primary,
                domain_assignment_weights={d: w for d, w in assignment.weights.items()},
                supporting_evidence=supporting_bucket,
                refuting_evidence=refuting_bucket,
                pcg_neighbors=neighbours,
                pcg=observability,
                nli_self_consistency_variance=0.0,
                bp_validated=bp_validated,
                information_gain=information_gain,
                queued_for_review=queued_for_review,
                calibration_set_id=cal.calibration_set_id,
                calibration_n=cal.calibration_n,
                fallback_used=cal.fallback_used,
            )
            claim_verdicts.append(cv)

        # L7 — document assembly
        await _safe_fire(phase_callback, "assembling")
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
        *,
        rigor: RigorTier = "balanced",
    ) -> _PosteriorsResult:
        """Compute posteriors + buckets + (optionally) PCG observability.

        Three code paths, in precedence order:

        1. **Wave 3 PCG path** (``self._pcg is not None AND
           self._nli_adapter is not None``): Fan out claim-evidence NLI
           under ``Semaphore(_NLI_MAX_CONCURRENCY)``, run the cc-NLI
           dispatcher (primary → fallback) for claim-claim pairs that
           pass the entity-overlap short-circuit, build the factor
           graph, run TRW-BP / damped LBP / Gibbs sanity, and return
           PCG marginals as ``PosteriorBelief``. Populates
           :class:`PCGObservability` per claim with algorithm, iteration
           counts, edge counts, log_partition_bound, gibbs_mismatch,
           and cc_nli_fallback_fired_count. Evidence buckets are the
           same label-driven split as the Beta path (Hebel A + Fix 1+3).

        2. **Phase-2 Beta-posterior path** (``self._nli_adapter is not
           None`` but no PCG): Beta(α, β) fold over claim-evidence NLI
           only. Kept for local dev + degraded-mode fallback when PCG
           isn't wired. Buckets: label-driven. No PCG observability.

        3. **Phase-1 placeholder** (no NLI adapter at all): posterior
           = mean evidence similarity. Keeps /verify end-to-end for
           tests that don't want to stub an NLI adapter.
        """
        del assignments  # Domain routing feeds L5 conformal, not L3/L4 here.

        if self._pcg is not None and self._nli_adapter is not None:
            return await self._compute_posteriors_via_pcg(
                claims, evidence_per_claim, rigor=rigor
            )

        if self._nli_adapter is not None:
            return await self._compute_posteriors_via_nli_adapter(
                claims, evidence_per_claim
            )

        # Placeholder path: no NLI adapter, so no per-evidence label
        # signal. Buckets default to the pre-NLI convention (everything
        # in supporting) so older deploys without D1+ wired still render.
        out: dict[UUID, PosteriorBelief] = {}
        buckets: dict[UUID, tuple[list[Evidence], list[Evidence]]] = {}
        observability: dict[UUID, PCGObservability | None] = {}
        pcg_edges: dict[UUID, list[ClaimEdge]] = {}
        for claim in claims:
            evidence = evidence_per_claim[claim.id]
            if not evidence:
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
            buckets[claim.id] = (list(evidence), [])
            observability[claim.id] = None
            pcg_edges[claim.id] = []
        return _PosteriorsResult(
            posteriors=out,
            buckets=buckets,
            pcg_observability=observability,
            pcg_edges=pcg_edges,
        )

    async def _compute_posteriors_via_nli_adapter(
        self,
        claims: list[Claim],
        evidence_per_claim: dict[UUID, list[Evidence]],
    ) -> _PosteriorsResult:
        """Beta-posterior fold over LLM-classified NLI results, plus
        label-driven evidence buckets.

        One ``classify`` call per (claim, evidence) pair, bounded by
        :data:`_NLI_MAX_CONCURRENCY`. Each task returns the original
        ``Evidence`` alongside its ``NliResult`` so assembly can
        partition evidence by ``label`` without re-classifying:

        * ``label == "support"`` → ``supporting_evidence`` bucket
          AND folded into Beta(α, β) posterior.
        * ``label == "refute"``  → ``refuting_evidence`` bucket
          AND folded into Beta(α, β) posterior.
        * ``label == "neutral"`` → dropped from both buckets AND
          **skipped** in the Beta fold. Neutrals represent "no signal"
          (off-topic or genuinely uncertain); previously their tilted
          scores were still being folded, which accumulated noise in
          the posterior on claims with many off-topic passages (post-
          v2.0 Einstein-Russia bug). Unified semantic now: if it's not
          in a display bucket, it's not in the posterior.
        * ``reasoning == "nli_unavailable"`` (terminal-failure sentinel)
          → dropped from both buckets AND skipped in the Beta fold.

        For ``support`` / ``refute``-labeled results, the Beta update
        still folds **both** ``supporting_score`` and ``refuting_score``
        (not just the label's score), so a passage with
        ``label="support"`` and ``refuting_score=0.2`` contributes 0.2
        to β. The label gates whether we fold at all; once we fold,
        the 2-D score vector shapes the magnitude.
        """
        assert self._nli_adapter is not None

        semaphore = asyncio.Semaphore(_NLI_MAX_CONCURRENCY)

        async def _classify_pair(
            claim_id: UUID,
            evidence: Evidence,
            claim_text: str,
            evidence_text: str,
        ) -> tuple[UUID, Evidence, NliResult]:
            async with semaphore:
                result = await self._nli_adapter.classify(claim_text, evidence_text)
            return claim_id, evidence, result

        tasks = [
            _classify_pair(claim.id, ev, claim.text, ev.content)
            for claim in claims
            for ev in evidence_per_claim.get(claim.id, [])
        ]
        classified: list[tuple[UUID, Evidence, NliResult]] = (
            list(await asyncio.gather(*tasks)) if tasks else []
        )

        # Partition by claim_id once; feeds both the Beta update and
        # the bucket split in one pass (no second walk over tasks).
        results_by_claim: dict[UUID, list[NliResult]] = {c.id: [] for c in claims}
        buckets: dict[UUID, tuple[list[Evidence], list[Evidence]]] = {
            c.id: ([], []) for c in claims
        }
        for claim_id, evidence, nli_result in classified:
            results_by_claim[claim_id].append(nli_result)
            # Terminal-failure sentinels carry no signal → drop from
            # both buckets (matches their exclusion from the Beta fold).
            if nli_result.reasoning == "nli_unavailable":
                continue
            if nli_result.label == "support":
                buckets[claim_id][0].append(_annotate_evidence_with_nli(evidence, nli_result))
            elif nli_result.label == "refute":
                buckets[claim_id][1].append(_annotate_evidence_with_nli(evidence, nli_result))
            # label == "neutral": drop from both buckets (off-topic
            # passage — neither confirms nor contradicts the claim).

        out: dict[UUID, PosteriorBelief] = {}
        for claim in claims:
            alpha, beta, folded = _beta_update_from_nli(results_by_claim[claim.id])
            p_true = alpha / (alpha + beta)
            out[claim.id] = PosteriorBelief(
                p_true=p_true,
                p_false=1.0 - p_true,
                converged=True,
                # Still the TRW-BP literal (the only one allowed today by
                # the PosteriorBelief.algorithm field); distinguished in
                # ``_model_versions`` as "phase2-beta-posterior-from-nli".
                algorithm="TRW-BP",
                iterations=folded,
                edge_count=0,
            )
        # Beta path produces no PCG observability or neighbour edges.
        observability: dict[UUID, PCGObservability | None] = {c.id: None for c in claims}
        pcg_edges: dict[UUID, list[ClaimEdge]] = {c.id: [] for c in claims}
        return _PosteriorsResult(
            posteriors=out,
            buckets=buckets,
            pcg_observability=observability,
            pcg_edges=pcg_edges,
        )

    async def _compute_posteriors_via_pcg(
        self,
        claims: list[Claim],
        evidence_per_claim: dict[UUID, list[Evidence]],
        *,
        rigor: RigorTier = "balanced",
    ) -> _PosteriorsResult:
        """Full Wave 3 PCG path.

        Pipeline:
          1. Fan out claim-evidence NLI (same Semaphore + label-driven
             bucket logic as the Beta path).
          2. Convert claim-evidence ``NliResult`` → ``NLIDistribution``.
          3. Run the cc-NLI dispatcher over claim pairs that survive the
             entity-overlap short-circuit + hard cap.
          4. Build the factor graph.
          5. Run PCG ``infer`` (TRW-BP primary, LBP fallback, Gibbs
             sanity).
          6. Wrap marginals + graph edges + dispatcher metrics into
             ``PosteriorBelief`` + ``PCGObservability`` + per-claim
             ``ClaimEdge`` lists.
        """
        assert self._pcg is not None
        assert self._nli_adapter is not None

        # --- Step 1: claim-evidence NLI fan-out ------------------------
        semaphore = asyncio.Semaphore(_NLI_MAX_CONCURRENCY)

        async def _classify_pair(
            claim_id: UUID,
            evidence: Evidence,
            claim_text: str,
            evidence_text: str,
        ) -> tuple[UUID, Evidence, NliResult]:
            async with semaphore:
                result = await self._nli_adapter.classify(claim_text, evidence_text)
            return claim_id, evidence, result

        tasks = [
            _classify_pair(claim.id, ev, claim.text, ev.content)
            for claim in claims
            for ev in evidence_per_claim.get(claim.id, [])
        ]
        classified: list[tuple[UUID, Evidence, NliResult]] = (
            list(await asyncio.gather(*tasks)) if tasks else []
        )

        # Partition NLI results + build display buckets (same label-driven
        # rule as the Beta path).
        results_by_claim: dict[UUID, list[NliResult]] = {c.id: [] for c in claims}
        buckets: dict[UUID, tuple[list[Evidence], list[Evidence]]] = {
            c.id: ([], []) for c in claims
        }
        for claim_id, evidence, nli_result in classified:
            results_by_claim[claim_id].append(nli_result)
            if nli_result.reasoning == "nli_unavailable":
                continue
            if nli_result.label == "support":
                buckets[claim_id][0].append(_annotate_evidence_with_nli(evidence, nli_result))
            elif nli_result.label == "refute":
                buckets[claim_id][1].append(_annotate_evidence_with_nli(evidence, nli_result))

        # --- Step 2: convert to NLIDistribution for the PCG layer ------
        # Import locally to avoid circular import at module load time.
        from adapters.nli_claim_claim import _nli_result_to_distribution  # noqa: PLC0415

        nli_claim_evidence: dict[UUID, list[NLIDistribution]] = {}
        for claim in claims:
            nli_claim_evidence[claim.id] = [
                _nli_result_to_distribution(r, nli_model_id="nli-claim-evidence")
                for r in results_by_claim[claim.id]
                if r.reasoning != "nli_unavailable"
            ]

        # --- Step 3: claim-claim NLI dispatch --------------------------
        nli_claim_claim: dict[tuple[UUID, UUID], NLIDistribution] = {}
        fallback_fired_count = 0
        if self._cc_nli_dispatcher is not None and len(claims) >= 2:
            dispatch = await self._cc_nli_dispatcher.classify_pairs(claims)
            nli_claim_claim = dispatch.distributions
            fallback_fired_count = dispatch.fallback_fired_count

        # --- Step 4+5: PCG inference -----------------------------------
        # adapter_per_claim is reserved for per-domain TRW-BP weights —
        # empty dict for now (all general domain, Wave 3 scope).
        adapter_per_claim: dict[UUID, object] = {}  # DomainAdapter typing — unused until Wave 4
        try:
            posteriors = await self._pcg.infer(
                claims=claims,
                evidence_per_claim=evidence_per_claim,
                nli_claim_evidence=nli_claim_evidence,
                nli_claim_claim=nli_claim_claim,
                adapter_per_claim=adapter_per_claim,  # type: ignore[arg-type]
                rigor=rigor,
            )
        except Exception as exc:
            logger.error(
                "PCG infer failed: %s — falling back to Phase-2 Beta posterior",
                exc,
            )
            return await self._compute_posteriors_via_nli_adapter(claims, evidence_per_claim)

        # --- Step 6: assemble per-claim observability + edges ----------
        observability: dict[UUID, PCGObservability | None] = {}
        pcg_edges: dict[UUID, list[ClaimEdge]] = {c.id: [] for c in claims}

        # Per-claim PCG observability block.
        gibbs_mismatch = getattr(self._pcg, "last_gibbs_mismatch", None)
        for claim in claims:
            belief = posteriors[claim.id]
            observability[claim.id] = PCGObservability(
                algorithm=belief.algorithm,
                converged=belief.converged,
                iterations=belief.iterations,
                edge_count=belief.edge_count,
                log_partition_bound=belief.log_partition_bound,
                gibbs_mismatch=gibbs_mismatch,
                cc_nli_fallback_fired_count=fallback_fired_count,
            )

        # Build pcg_neighbors from the cc-NLI pairs. Each pair produces
        # one ClaimEdge on each endpoint.
        for (id_a, id_b), dist in nli_claim_claim.items():
            # Pick edge type from the dominant NLI mass.
            if dist.entail >= dist.contradict:
                edge_type = EdgeType.ENTAIL
                strength = max(0.0, min(1.0, dist.entail - dist.neutral))
            else:
                edge_type = EdgeType.CONTRADICT
                strength = max(0.0, min(1.0, dist.contradict - dist.neutral))
            if id_a in pcg_edges:
                pcg_edges[id_a].append(
                    ClaimEdge(neighbor_claim_id=id_b, edge_type=edge_type, edge_strength=strength)
                )
            if id_b in pcg_edges:
                pcg_edges[id_b].append(
                    ClaimEdge(neighbor_claim_id=id_a, edge_type=edge_type, edge_strength=strength)
                )

        return _PosteriorsResult(
            posteriors=posteriors,
            buckets=buckets,
            pcg_observability=observability,
            pcg_edges=pcg_edges,
        )

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
        """Collect model version strings from the layers. Wave 3
        adds ``pcg`` tag for the TRW-BP / LBP / Gibbs stack and
        ``cc_nli`` tag for the claim-claim dispatcher (primary + fallback)."""
        if self._pcg is not None and self._nli_adapter is not None:
            pcg_tag = "wave3-trw-bp-lbp-gibbs-cc-openai-v1"
        elif self._nli_adapter is not None:
            pcg_tag = "phase2-beta-posterior-from-nli"
        else:
            pcg_tag = "phase1-placeholder-mean-similarity"
        versions: dict[str, str] = {
            "decomposer": getattr(self._decomposer, "model_id", "phase1-default"),
            "domain_router": "phase1-placeholder-general",
            "pcg": pcg_tag,
            "conformal": "phase1-split-conformal-stub",
        }
        if self._nli is not None:
            versions["nli"] = getattr(self._nli, "model_id", "unknown")
        if self._nli_adapter is not None:
            # NliAdapter doesn't expose a model_id (it composes over an
            # LLMProvider); best we can do is surface the adapter class.
            versions["nli_adapter"] = type(self._nli_adapter).__name__
        if self._cc_nli_dispatcher is not None:
            versions["cc_nli"] = type(self._cc_nli_dispatcher).__name__
            primary = getattr(self._cc_nli_dispatcher, "_primary", None)
            if primary is not None:
                versions["cc_nli_primary"] = type(primary).__name__
        return versions


def _beta_update_from_nli(
    nli_results: list[NliResult],
) -> tuple[float, float, int]:
    """Fold NLI classifications into a Beta(α, β) posterior.

    Starting from a uniform prior (α=β=1, p_true=0.5), each decisive NLI
    result (``label ∈ {"support", "refute"}``) contributes:

    * ``α += supporting_score``
    * ``β += refuting_score``

    Two classes of result are **skipped** (no α/β contribution):

    1. Terminal-failure sentinels (``reasoning == "nli_unavailable"``) —
       they carry no information, so they must not push the posterior
       around.
    2. ``label == "neutral"`` results — these correspond to passages
       that neither confirm nor contradict the claim (off-topic or
       genuinely uncertain). Post-v2.0-live-bug (Einstein-Russia case,
       p_true=0.33 for a 100%-refutable claim): off-topic-but-score-
       tilted neutrals were accumulating noise in the posterior. The
       fix is to treat the ``neutral`` label as a first-class "no
       signal" marker matching its bucket semantic (neutrals already
       dropped from both supporting_evidence and refuting_evidence
       display buckets). Only ``support`` / ``refute`` labels move α/β.

    The ``support`` / ``refute`` buckets still fold *both* scores (not
    just the label's score), so a passage with ``label="support"`` and
    ``refuting_score=0.2`` still contributes 0.2 to β. The label gates
    whether we fold at all; once we decide to fold, the full 2-D score
    shapes the magnitude.

    Returns ``(alpha, beta, folded_count)`` for the caller's
    :class:`PosteriorBelief` construction.
    """
    alpha = 1.0
    beta = 1.0
    folded = 0
    for result in nli_results:
        if result.reasoning == "nli_unavailable":
            continue
        if result.label == "neutral":
            continue
        alpha += result.supporting_score
        beta += result.refuting_score
        folded += 1
    return alpha, beta, folded


def _annotate_evidence_with_nli(
    evidence: Evidence,
    nli_result: NliResult,
) -> Evidence:
    """Attach per-evidence NLI metadata for downstream rendering.

    The bucket label decides whether the evidence appears under
    supporting or refuting evidence; this helper stamps the exact
    classifier output onto the ``Evidence`` object so the UI can
    explain *why* it landed there and show the score used for that
    bucket.
    """
    bucket_score = (
        nli_result.supporting_score
        if nli_result.label == "support"
        else nli_result.refuting_score
    )
    structured_data = dict(evidence.structured_data or {})
    structured_data.update(
        {
            "nli_label": nli_result.label,
            "nli_reasoning": (
                None
                if nli_result.reasoning == "nli_unavailable"
                else nli_result.reasoning
            ),
            "supporting_score": nli_result.supporting_score,
            "refuting_score": nli_result.refuting_score,
            "neutral_score": nli_result.neutral_score,
            "nli_confidence": nli_result.confidence,
            "bucket_score": bucket_score,
        }
    )
    return evidence.model_copy(
        update={
            "classification_confidence": bucket_score,
            "structured_data": structured_data,
        }
    )
