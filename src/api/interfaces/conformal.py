"""L5 Conformal calibration port + CalibratedVerdict value object. Spec §7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from models.domain import Domain
    from models.entities import Claim
    from models.pcg import PosteriorBelief


FallbackLevel = Literal["domain", "general", "non_converged"]


@dataclass(frozen=True)
class CalibratedVerdict:
    """L5 internal output. L7 maps this into the public ClaimVerdict.

    - ``coverage_target`` is ``None`` when fallback_used != None.
    - ``calibration_set_id`` is ``None`` when fallback_used == 'general'
      (no real calibration set was consulted).
    - ``interval`` is ``(0.0, 1.0)`` when fallback_used == 'non_converged'
      (uninformative by design — see §7).
    """

    p_true: float
    interval_lower: float
    interval_upper: float
    coverage_target: float | None
    calibration_set_id: str | None
    calibration_n: int
    domain: Domain
    stratum: str
    fallback_used: FallbackLevel | None


@runtime_checkable
class ConformalCalibrator(Protocol):
    """Wraps an L4 posterior with a distribution-free prediction interval."""

    async def calibrate(
        self,
        claim: Claim,
        belief: PosteriorBelief,
        domain: Domain,
        stratum: str,
    ) -> CalibratedVerdict: ...
