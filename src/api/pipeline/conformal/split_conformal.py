"""Split-conformal calibrator. Spec §7.

Implements the full split-conformal algorithm but gracefully degrades to
``fallback_used="general"`` when the relevant calibration partition has
fewer than the spec-§7 minimum of 50 entries.

Phase 1 behaviour (stub): every partition is empty, so every claim
comes back with ``coverage_target=None``, ``fallback_used="general"``,
``interval=(0.0, 1.0)``. Honest by design — we don't claim coverage we
cannot deliver.

Phase 3+ fills the calibration sets via the adjudicator-labeling sprint
and the nightly re-scoring job; the same code path then emits real
intervals without any algorithm change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from interfaces.conformal import CalibratedVerdict, ConformalCalibrator

if TYPE_CHECKING:
    from models.domain import Domain
    from models.entities import Claim
    from models.pcg import PosteriorBelief
    from pipeline.conformal.calibration_store import CalibrationStore

logger = logging.getLogger(__name__)


class SplitConformalCalibrator(ConformalCalibrator):
    """Wraps an L4 posterior with a distribution-free prediction interval.

    The ``stratum`` argument is ``"<domain>:<claim_type>"`` or
    ``"<domain>:any"`` for the per-domain fallback. Mondrian-stratum
    fallback-cascade logic (Task 3.7) lives in the pipeline
    orchestrator, not in this class — this class calibrates one
    (claim, posterior, stratum) tuple at a time.
    """

    def __init__(self, store: CalibrationStore, *, default_coverage: float = 0.90) -> None:
        self._store = store
        self._default_coverage = default_coverage

    async def calibrate(
        self,
        claim: Claim,
        belief: PosteriorBelief,
        domain: Domain,
        stratum: str,
    ) -> CalibratedVerdict:
        del claim  # unused — reserved for claim-type-specific logic later

        # Honest Phase 1 / stratum-too-small path: no coverage guarantee.
        # The stub-store returns None for every partition in Phase 1.
        alpha = 1.0 - self._default_coverage
        q = self._store.get_quantile(stratum, alpha=alpha)
        n = self._store.get_n(stratum)

        if q is None:
            logger.debug("Stratum %r has n=%d (< min); emitting general fallback.", stratum, n)
            return CalibratedVerdict(
                p_true=belief.p_true,
                interval_lower=0.0,
                interval_upper=1.0,
                coverage_target=None,
                calibration_set_id=None,
                calibration_n=n,
                domain=domain,
                stratum=stratum,
                fallback_used="general",
            )

        # Non-converged L4 beliefs: per spec §7, exclude from the
        # coverage guarantee. The point estimate is still emitted best-
        # effort; the interval widens to [0, 1].
        if not belief.converged:
            return CalibratedVerdict(
                p_true=belief.p_true,
                interval_lower=0.0,
                interval_upper=1.0,
                coverage_target=None,
                calibration_set_id=stratum,
                calibration_n=n,
                domain=domain,
                stratum=stratum,
                fallback_used="non_converged",
            )

        lower = max(0.0, belief.p_true - q)
        upper = min(1.0, belief.p_true + q)

        return CalibratedVerdict(
            p_true=belief.p_true,
            interval_lower=lower,
            interval_upper=upper,
            coverage_target=self._default_coverage,
            calibration_set_id=stratum,
            calibration_n=n,
            domain=domain,
            stratum=stratum,
            fallback_used=None,
        )
