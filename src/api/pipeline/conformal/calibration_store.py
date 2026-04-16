"""Calibration-set storage port + Phase 1 in-memory implementation.

The store exposes three operations L5 uses at request time:

  * ``get_quantile(partition)`` — the ``(1-α)`` conformal quantile for a
    given partition (``"<domain>:<claim_type>"`` or ``"<domain>:any"``
    when per-stratum data is insufficient).
  * ``get_n(partition)`` — number of calibration entries. Used by the
    Mondrian fallback cascade in Task 3.7 to decide when to fall back.
  * ``add_entry(partition, posterior, true_label)`` — write a single
    calibration entry. Phase 4 (Task 4.1) replaces this with the
    Postgres-backed implementation that reads from ``calibration_set``.

The Phase 1 in-memory store is deliberately empty — every partition has
n=0, so L5 always takes the ``fallback_used="general"`` path. This lets
the full pipeline return valid ``DocumentVerdict`` objects without
fabricating coverage guarantees we can't honor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationStoreEntry:
    """One entry in the calibration set (Phase 1 in-memory form)."""

    partition: str
    posterior: float
    true_label: bool  # True if the claim was judged true, False otherwise


class CalibrationStore(ABC):
    """Port for calibration-set storage.

    Phase 4 swaps in ``PostgresCalibrationStore`` reading from the
    ``calibration_set`` table per spec §12.
    """

    @abstractmethod
    def get_n(self, partition: str) -> int:
        """Number of entries in the given partition."""

    @abstractmethod
    def get_quantile(self, partition: str, *, alpha: float = 0.10) -> float | None:
        """Conformal ``(1-α)`` quantile for the partition.

        Returns None when ``get_n(partition) < 50`` (spec §7 minimum
        stratum size). Callers handle the None by falling back through
        the Mondrian cascade (stratum → domain → general), or by
        emitting ``fallback_used`` and ``coverage_target=null``.
        """

    @abstractmethod
    def add_entry(self, partition: str, posterior: float, *, true_label: bool) -> None:
        """Record one calibration entry."""


class InMemoryCalibrationStore(CalibrationStore):
    """Process-local store — fine for tests + Phase 1.

    Entries are grouped by partition. Per-partition nonconformity scores
    are recomputed on every ``get_quantile`` call (small N; negligible
    cost). The implementation follows spec §7:

        s_k = |b_k - 𝟙[true_label_k]|
        q̂   = ⌈(n+1)(1-α)⌉ / n empirical quantile of {s_k}
    """

    _MIN_N = 50  # spec §7 minimum stratum size

    def __init__(self) -> None:
        self._by_partition: dict[str, list[CalibrationStoreEntry]] = {}

    def get_n(self, partition: str) -> int:
        return len(self._by_partition.get(partition, []))

    def get_quantile(self, partition: str, *, alpha: float = 0.10) -> float | None:
        entries = self._by_partition.get(partition, [])
        n = len(entries)
        if n < self._MIN_N:
            return None

        # Nonconformity scores: |b_k − 𝟙[true_label_k]|
        scores = sorted(abs(e.posterior - (1.0 if e.true_label else 0.0)) for e in entries)
        # Empirical (1-α) quantile with the standard (n+1) adjustment.
        # rank = ceil((n+1)(1-α)); cap at n.
        import math

        rank = math.ceil((n + 1) * (1.0 - alpha))
        rank = min(max(rank, 1), n)
        return float(scores[rank - 1])

    def add_entry(self, partition: str, posterior: float, *, true_label: bool) -> None:
        if not (0.0 <= posterior <= 1.0):
            raise ValueError(f"posterior must be in [0, 1], got {posterior}")
        self._by_partition.setdefault(partition, []).append(
            CalibrationStoreEntry(partition=partition, posterior=posterior, true_label=true_label)
        )
