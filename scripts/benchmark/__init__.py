"""OHI v2 benchmark harness.

Engine-agnostic measurement infrastructure for the plan's Phase 0–4
acceptance gates. See `docs/superpowers/plans/2026-04-16-ohi-v2-
implementation.md` — Task 0.2.

Public surface:

    from scripts.benchmark.types import (
        BenchmarkExample,
        BenchmarkResult,
        VerificationEngine,
    )
    from scripts.benchmark.runner import run_benchmark
    from scripts.benchmark.metrics import (
        compute_f1,
        compute_ece,
        compute_calibration_coverage,
        aggregate_per_domain,
    )

All metric functions accept generic "key extractors" so the same harness
can score any benchmark (binary, multi-class, calibrated, or not) without
per-benchmark coupling in the metrics module.
"""

from __future__ import annotations
