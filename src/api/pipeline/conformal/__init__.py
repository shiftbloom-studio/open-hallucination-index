"""L5 Conformal calibration. Spec §7.

Phase 1 ships an honest stub: no real calibration set exists yet, so
every claim comes back with ``coverage_target=None``, ``fallback_used=
"general"``, and an uninformative ``interval=(0.0, 1.0)``. Phase 3
populates per-domain calibration sets via the adjudicator-labeling
sprint; at that point this module's conditional branches light up and
emit real conformal intervals.

The full split-conformal algorithm is still implemented here — only
the calibration set is empty. That keeps Phase 1 and Phase 3 one
config change apart instead of a rewrite.
"""

from __future__ import annotations

from pipeline.conformal.calibration_store import (
    CalibrationStore,
    InMemoryCalibrationStore,
)
from pipeline.conformal.split_conformal import SplitConformalCalibrator

__all__ = [
    "CalibrationStore",
    "InMemoryCalibrationStore",
    "SplitConformalCalibrator",
]
