"""Shared types for the benchmark harness.

Kept deliberately small — these are the only types every engine adapter,
dataset loader, runner, and metric function needs to agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BenchmarkExample:
    """One row from a benchmark dataset.

    `expected_label` is the benchmark's native label encoding. Callers are
    responsible for passing compatible key-extractors to the metrics module
    so labels from different benchmarks never get mixed up.
    """

    id: str
    text: str
    expected_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """Per-example result produced by a VerificationEngine.

    `p_true` and `interval` are optional because v1 (baseline) only emits a
    point estimate, whereas v2 emits a calibrated probability plus a
    conformal interval. Metric functions handle None gracefully.
    """

    example_id: str
    predicted_label: str
    p_true: float | None
    interval: tuple[float, float] | None
    raw_response: dict[str, Any]
    latency_ms: float
    error: str | None


@runtime_checkable
class VerificationEngine(Protocol):
    """Adapter protocol any engine (v1, v2-phaseN, mock) must satisfy."""

    name: str

    async def verify(
        self, example: BenchmarkExample
    ) -> BenchmarkResult:  # pragma: no cover - protocol
        ...
