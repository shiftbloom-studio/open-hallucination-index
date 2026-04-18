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
    from interfaces.nli import NliAdapter, NLIService
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
        domain_adapters: dict[Domain, DomainAdapter] | None = None,
    ) -> None:
        # ``nli`` (NLIService) is the cross-encoder port that predates
        # Phase 2 and is still reserved for a future cross-encoder
        # deployment. ``nli_adapter`` (NliAdapter) is the Phase 2 LLM-
        # backed 3-way classifier that D1 wires; both coexist so the
        # older port isn't prematurely deleted. The new adapter takes
        # precedence in ``_compute_posteriors`` when both are set.
        self._decomposer = decomposer
        self._retrieval = retrieval
        self._conformal = conformal
        self._router = domain_router
        self._nli = nli
        self._nli_adapter = nli_adapter
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

        # L3 / L4 — NLI + PCG joint inference (placeholder: naive posterior)
        await _safe_fire(phase_callback, "classifying")
        posteriors, nli_buckets = await self._compute_posteriors(
            claims, evidence_per_claim, assignments
        )

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

            cv = assemble_claim_verdict(
                claim=claim,
                p_true=cal.p_true,
                interval=(cal.interval_lower, cal.interval_upper),
                coverage_target=cal.coverage_target,
                domain=primary,
                domain_assignment_weights={d: w for d, w in assignment.weights.items()},
                supporting_evidence=supporting_bucket,
                refuting_evidence=refuting_bucket,
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
    ) -> tuple[
        dict[UUID, PosteriorBelief],
        dict[UUID, tuple[list[Evidence], list[Evidence]]],
    ]:
        """Compute posteriors AND the NLI-label-driven evidence buckets.

        Returns ``(posteriors, buckets)`` where ``buckets[claim_id]`` is
        the pair ``(supporting_evidence, refuting_evidence)`` — evidence
        items classified as ``label="support"`` go in the first list,
        ``label="refute"`` in the second, and ``label="neutral"`` (plus
        any terminal-failure sentinels) are dropped from both. Assembly
        consumes these buckets directly instead of the pre-D1 hardcode
        that put everything in ``supporting_evidence``.

        Three code paths, in precedence order:

        1. **Phase 2 LLM-NLI path** (``self._nli_adapter is not None``):
           Fan out ``NliAdapter.classify`` over every (claim, evidence)
           pair under an ``asyncio.Semaphore`` bound by
           :data:`_NLI_MAX_CONCURRENCY`, fold each successful
           ``supporting_score`` / ``refuting_score`` into a per-claim
           Beta(α, β) update, and return ``α / (α + β)``. Sentinel
           ``nli_unavailable`` results are *skipped* (no signal) rather
           than folded as neutral — a flaky LLM must not silently bias
           the posterior toward zero. Buckets reflect the per-pair NLI
           ``label``.

        2. **Future cross-encoder + PCG path** (``self._nli is not None
           and self._pcg is not None``): raises ``NotImplementedError``
           until Task 2.8 wires the real joint-inference flow.

        3. **Phase 1 placeholder path** (everything else): posterior =
           mean evidence similarity score. Keeps the pipeline end-to-
           end functional so /verify returns valid ``DocumentVerdict``
           objects even before any NLI is injected. Buckets fall back
           to the pre-NLI convention: all evidence in supporting, none
           in refuting (no NLI to discriminate).
        """
        del assignments  # Domain routing feeds L5 conformal, not L3/L4 here.

        if self._nli_adapter is not None:
            return await self._compute_posteriors_via_nli_adapter(
                claims, evidence_per_claim
            )

        if self._nli is not None and self._pcg is not None:
            # Full cross-encoder + TRW-BP path will land in Task 2.8.
            logger.debug("Running NLI + PCG joint inference")
            raise NotImplementedError(
                "NLI + PCG wiring lands in Task 2.8; currently only "
                "the Phase 1 placeholder posterior is active."
            )

        # Placeholder path: no NLI adapter, so no per-evidence label
        # signal. Buckets default to the pre-NLI convention (everything
        # in supporting) so older deploys without D1+ wired still render.
        out: dict[UUID, PosteriorBelief] = {}
        buckets: dict[UUID, tuple[list[Evidence], list[Evidence]]] = {}
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
            buckets[claim.id] = (list(evidence), [])
        return out, buckets

    async def _compute_posteriors_via_nli_adapter(
        self,
        claims: list[Claim],
        evidence_per_claim: dict[UUID, list[Evidence]],
    ) -> tuple[
        dict[UUID, PosteriorBelief],
        dict[UUID, tuple[list[Evidence], list[Evidence]]],
    ]:
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
        ) -> tuple[UUID, Evidence, "NliResult"]:
            async with semaphore:
                result = await self._nli_adapter.classify(claim_text, evidence_text)
            return claim_id, evidence, result

        tasks = [
            _classify_pair(claim.id, ev, claim.text, ev.content)
            for claim in claims
            for ev in evidence_per_claim.get(claim.id, [])
        ]
        classified: list[tuple[UUID, Evidence, "NliResult"]] = (
            list(await asyncio.gather(*tasks)) if tasks else []
        )

        # Partition by claim_id once; feeds both the Beta update and
        # the bucket split in one pass (no second walk over tasks).
        results_by_claim: dict[UUID, list["NliResult"]] = {c.id: [] for c in claims}
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
                buckets[claim_id][0].append(evidence)
            elif nli_result.label == "refute":
                buckets[claim_id][1].append(evidence)
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
        return out, buckets

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
            "pcg": (
                "phase2-beta-posterior-from-nli"
                if self._nli_adapter is not None
                else "phase1-placeholder-mean-similarity"
            ),
            "conformal": "phase1-split-conformal-stub",
        }
        if self._nli is not None:
            versions["nli"] = getattr(self._nli, "model_id", "unknown")
        if self._nli_adapter is not None:
            # NliAdapter doesn't expose a model_id (it composes over an
            # LLMProvider); best we can do is surface the adapter class.
            versions["nli_adapter"] = type(self._nli_adapter).__name__
        return versions


def _beta_update_from_nli(
    nli_results: list["NliResult"],
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
