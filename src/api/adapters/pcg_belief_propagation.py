"""Main PCG adapter — Wave 3 Stream P.

Satisfies :class:`~interfaces.pcg.PCGInferenceService`. Orchestrates
the three inference paths (TRW-BP primary, damped LBP fallback, Gibbs
sanity) on a factor graph built from the claim-evidence + claim-claim
NLI tables.

Rigor tier decides what runs:

* ``fast``: damped LBP only, no Gibbs, binary-factor layer omitted.
  Matches the port's "fast-means-fast" contract — skip everything
  that a user might wait on for a cached / exploratory query.
* ``balanced`` (default): TRW-BP primary. On non-convergence, fall
  back to damped LBP. If that also fails, return the last-iter
  marginal with ``converged=False`` and
  ``algorithm="LBP-nonconvergent"``. Gibbs runs in parallel with the
  BP pass; mismatch beyond ``gibbs_tolerance`` populates the
  ``gibbs_mismatch`` observability field (surfaced but non-blocking).
* ``maximum``: same as ``balanced`` with tighter tolerance + larger
  Gibbs sample count.

Output: ``dict[UUID, PosteriorBelief]`` keyed by claim id. Observability
goes into each :class:`PosteriorBelief`'s fields (algorithm, iterations,
edge_count, log_partition_bound) — the pipeline layer reads these and
packs a :class:`~models.verdict_extensions.PCGObservability` block onto
each :class:`ClaimVerdict` downstream. ``gibbs_mismatch`` is surfaced
through the adapter's :attr:`last_gibbs_mismatch` attribute (set during
``infer``) because ``PosteriorBelief`` is frozen without that field.

Deterministic: Gibbs seed is derived from the UUIDs of the claims so
repeat verifies produce identical sanity verdicts — integration tests
would flake otherwise.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from adapters._pcg_gibbs import max_marginal_delta, run_gibbs
from adapters._pcg_graph import build_factor_graph
from adapters._pcg_lbp import run_damped_lbp
from adapters._pcg_trw_bp import run_trw_bp
from models.pcg import PosteriorBelief

if TYPE_CHECKING:
    from interfaces.domain import DomainAdapter
    from models.entities import Claim, Evidence
    from models.nli import NLIDistribution

logger = logging.getLogger(__name__)


RigorTier = Literal["fast", "balanced", "maximum"]


class PCGBeliefPropagationAdapter:
    """In-repo TRW-BP + damped LBP + Gibbs implementation.

    Constructor args tune the inference paths; defaults match the Wave
    3 spec and the env vars wired through ``settings.py`` /
    ``dependencies.py``. The cc-NLI adapter is passed here but only
    used to record fallback-firing counts — the dispatcher runs
    upstream in the pipeline and hands pre-resolved NLI tables to
    ``infer``.
    """

    def __init__(
        self,
        *,
        max_iters: int = 200,
        convergence_tol: float = 1e-4,
        damping_factor: float = 0.8,
        rigor_default: RigorTier = "balanced",
        entity_overlap_threshold: int = 1,
        gibbs_burn_in: int = 500,
        gibbs_samples: int = 2000,
        gibbs_tolerance: float = 0.05,
        claim_claim_max_pairs: int = 200,
    ) -> None:
        self._max_iters = max_iters
        self._conv_tol = convergence_tol
        self._damping = damping_factor
        self._rigor_default = rigor_default
        self._overlap_threshold = entity_overlap_threshold
        self._gibbs_burn_in = gibbs_burn_in
        self._gibbs_samples = gibbs_samples
        self._gibbs_tol = gibbs_tolerance
        self._max_pairs = claim_claim_max_pairs
        # Transient state exposed for the pipeline observability layer.
        self.last_gibbs_mismatch: float | None = None

    async def infer(
        self,
        claims: list["Claim"],
        evidence_per_claim: dict[UUID, list["Evidence"]],
        nli_claim_evidence: dict[UUID, list["NLIDistribution"]],
        nli_claim_claim: dict[tuple[UUID, UUID], "NLIDistribution"],
        adapter_per_claim: dict[UUID, "DomainAdapter"],
        *,
        rigor: RigorTier = "balanced",
    ) -> dict[UUID, PosteriorBelief]:
        """Run PCG inference and return per-claim posteriors.

        Delegates graph construction to
        :func:`~adapters._pcg_graph.build_factor_graph`, then dispatches
        the chosen inference pipeline and wraps marginals into
        :class:`PosteriorBelief` frozen objects.
        """
        del evidence_per_claim  # reserved for future edge-weight adapters
        del adapter_per_claim  # reserved for per-domain TRW-BP weights

        # Empty-input edge case: no claims → no posteriors.
        if not claims:
            self.last_gibbs_mismatch = None
            return {}

        # Fast rigor skips the binary-factor layer entirely — produces
        # per-claim independent marginals from evidence unaries only.
        cc_table = {} if rigor == "fast" else nli_claim_claim
        graph = build_factor_graph(
            claims=claims,
            nli_claim_evidence=nli_claim_evidence,
            nli_claim_claim=cc_table,
        )

        # Reset the transient mismatch state each call.
        self.last_gibbs_mismatch = None

        if rigor == "fast":
            lbp_result = run_damped_lbp(
                graph,
                max_iters=max(1, self._max_iters // 4),
                convergence_tol=self._conv_tol,
                damping_factor=self._damping,
            )
            return _wrap_marginals(
                graph,
                lbp_result.marginals,
                algorithm="LBP-fallback" if lbp_result.converged else "LBP-nonconvergent",
                converged=lbp_result.converged,
                iterations=lbp_result.iterations,
                log_partition_bound=None,
            )

        # balanced / maximum: TRW-BP primary.
        max_iters_this = self._max_iters if rigor == "balanced" else int(self._max_iters * 1.5)
        trw_result = run_trw_bp(
            graph,
            max_iters=max_iters_this,
            convergence_tol=self._conv_tol,
        )

        if trw_result.converged:
            # Gibbs sanity (balanced + maximum).
            self._run_gibbs_sanity(graph, trw_result.marginals, rigor=rigor)
            return _wrap_marginals(
                graph,
                trw_result.marginals,
                algorithm="TRW-BP",
                converged=True,
                iterations=trw_result.iterations,
                log_partition_bound=trw_result.log_partition_bound,
            )

        # TRW-BP non-convergent → damped LBP fallback.
        logger.info(
            "TRW-BP did not converge in %d iters; falling back to damped LBP",
            trw_result.iterations,
        )
        lbp_result = run_damped_lbp(
            graph,
            max_iters=max_iters_this,
            convergence_tol=self._conv_tol,
            damping_factor=self._damping,
        )

        if lbp_result.converged:
            self._run_gibbs_sanity(graph, lbp_result.marginals, rigor=rigor)
            return _wrap_marginals(
                graph,
                lbp_result.marginals,
                algorithm="LBP-fallback",
                converged=True,
                iterations=trw_result.iterations + lbp_result.iterations,
                log_partition_bound=None,
            )

        logger.warning(
            "Both TRW-BP (%d iters) and damped LBP (%d iters) non-convergent; "
            "returning last-iter LBP marginals with converged=False",
            trw_result.iterations,
            lbp_result.iterations,
        )
        # Still run Gibbs sanity so the verdict surfaces the mismatch
        # number even when BP itself failed.
        self._run_gibbs_sanity(graph, lbp_result.marginals, rigor=rigor)
        return _wrap_marginals(
            graph,
            lbp_result.marginals,
            algorithm="LBP-nonconvergent",
            converged=False,
            iterations=trw_result.iterations + lbp_result.iterations,
            log_partition_bound=None,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_gibbs_sanity(
        self,
        graph,
        reference_marginals,
        *,
        rigor: RigorTier,
    ) -> None:
        """Gibbs sanity pass. Sets ``last_gibbs_mismatch`` when delta
        exceeds ``gibbs_tolerance``.

        Uses a deterministic seed derived from the claim-id sequence
        so repeated verifies produce identical sanity verdicts.
        """
        # Tighter sampling budget in maximum rigor.
        samples = self._gibbs_samples * (2 if rigor == "maximum" else 1)
        burn = self._gibbs_burn_in * (2 if rigor == "maximum" else 1)
        seed = _deterministic_seed_from_uuids(graph.claim_ids)

        try:
            gibbs = run_gibbs(
                graph,
                samples=samples,
                burn_in=burn,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 — sanity must not break verdict
            logger.warning("Gibbs sanity failed to run: %s", exc)
            self.last_gibbs_mismatch = None
            return

        delta = max_marginal_delta(reference_marginals, gibbs.marginals)
        if delta > self._gibbs_tol:
            logger.warning(
                "Gibbs-vs-BP mismatch detected: max marginal delta %.4f > tol %.4f "
                "(continuing with BP marginals; verdict will carry gibbs_mismatch warning)",
                delta,
                self._gibbs_tol,
            )
            self.last_gibbs_mismatch = delta
        else:
            self.last_gibbs_mismatch = None

    async def health_check(self) -> bool:
        """Pure-python adapter, no external deps — always healthy
        provided numpy imports (ctor succeeded)."""
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_seed_from_uuids(uuids: list[UUID]) -> int:
    """Derive a fixed seed from a list of UUIDs via SHA-256 prefix."""
    h = hashlib.sha256()
    for u in uuids:
        h.update(u.bytes)
    # Take first 8 bytes → int64-ish seed.
    return int.from_bytes(h.digest()[:8], byteorder="big", signed=False) % (2**31)


def _wrap_marginals(
    graph,
    marginals,
    *,
    algorithm: str,
    converged: bool,
    iterations: int,
    log_partition_bound: float | None,
) -> dict[UUID, PosteriorBelief]:
    """Wrap the per-node marginals into ``PosteriorBelief`` objects
    keyed by claim id."""
    out: dict[UUID, PosteriorBelief] = {}
    for i, claim_id in enumerate(graph.claim_ids):
        p_false = float(marginals[i, 0])
        p_true = float(marginals[i, 1])
        # Re-normalise for numerical noise (log_normalize_2d already
        # does this; re-check here as a belt-and-braces).
        total = p_true + p_false
        if total > 0 and abs(total - 1.0) > 1e-6:
            p_true /= total
            p_false /= total
        # Clamp to [0, 1] (guard against numerical underflow).
        p_true = max(0.0, min(1.0, p_true))
        p_false = max(0.0, min(1.0, 1.0 - p_true))
        out[claim_id] = PosteriorBelief(
            p_true=p_true,
            p_false=p_false,
            converged=converged,
            algorithm=algorithm,  # type: ignore[arg-type]  # narrowed by dispatch
            iterations=iterations,
            edge_count=graph.n_edges,
            log_partition_bound=log_partition_bound,
        )
    return out


__all__ = [
    "PCGBeliefPropagationAdapter",
    "RigorTier",
]
