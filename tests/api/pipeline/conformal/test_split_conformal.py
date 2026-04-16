"""Tests for pipeline.conformal.split_conformal + calibration_store.

Covers three distinct paths:
  1. Phase 1 stub path (empty store → fallback_used="general").
  2. Algorithm correctness (populated store → real interval, coverage
     approximately honoured on synthetic held-out data).
  3. Non-converged belief handling (excluded from calibration guarantee).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from uuid import uuid4

import pytest

_SRC_API = Path(__file__).resolve().parents[4] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from models.entities import Claim, ClaimType  # noqa: E402
from models.pcg import PosteriorBelief  # noqa: E402
from pipeline.conformal.calibration_store import InMemoryCalibrationStore  # noqa: E402
from pipeline.conformal.split_conformal import SplitConformalCalibrator  # noqa: E402


def _claim() -> Claim:
    return Claim(id=uuid4(), text="test claim", claim_type=ClaimType.UNCLASSIFIED)


def _belief(p: float, *, converged: bool = True) -> PosteriorBelief:
    return PosteriorBelief(
        p_true=p,
        p_false=1.0 - p,
        converged=converged,
        algorithm="TRW-BP",
        iterations=3,
        edge_count=0,
    )


# ---------------------------------------------------------------------------
# Phase 1 stub path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase1_empty_store_emits_general_fallback() -> None:
    calibrator = SplitConformalCalibrator(InMemoryCalibrationStore())
    result = await calibrator.calibrate(
        _claim(), _belief(0.8), domain="general", stratum="general:any"
    )
    assert result.fallback_used == "general"
    assert result.coverage_target is None
    assert result.interval_lower == 0.0
    assert result.interval_upper == 1.0
    assert result.calibration_set_id is None
    assert result.calibration_n == 0
    # Point estimate preserved
    assert result.p_true == 0.8


@pytest.mark.asyncio
async def test_stratum_below_minimum_still_triggers_fallback() -> None:
    """Adding 20 entries (below the 50-entry minimum) still returns the
    general fallback — we don't cheat by emitting "tight" intervals
    from an underpowered calibration set."""
    store = InMemoryCalibrationStore()
    for _ in range(20):
        store.add_entry("general:any", posterior=0.7, true_label=True)
    calibrator = SplitConformalCalibrator(store)
    result = await calibrator.calibrate(
        _claim(), _belief(0.8), domain="general", stratum="general:any"
    )
    assert result.fallback_used == "general"
    assert result.coverage_target is None
    assert result.calibration_n == 20


# ---------------------------------------------------------------------------
# Algorithm correctness path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_populated_store_emits_real_interval() -> None:
    """With 100 well-calibrated entries, the interval should actually
    shrink around the posterior. Not a strict coverage test (we need
    more samples for that) — just that the `fallback_used` flag is off
    and the interval is narrower than [0, 1]."""
    store = InMemoryCalibrationStore()
    rng = random.Random(42)
    for _ in range(100):
        # A well-calibrated model: posteriors are close to truth with
        # small symmetric noise. Nonconformity scores will be small.
        truth = rng.random() < 0.5
        posterior = (0.95 if truth else 0.05) + rng.uniform(-0.03, 0.03)
        posterior = max(0.0, min(1.0, posterior))
        store.add_entry("general:temporal", posterior=posterior, true_label=truth)

    calibrator = SplitConformalCalibrator(store)
    result = await calibrator.calibrate(
        _claim(), _belief(0.5), domain="general", stratum="general:temporal"
    )
    assert result.fallback_used is None
    assert result.coverage_target == pytest.approx(0.90)
    assert result.calibration_n == 100
    assert result.calibration_set_id == "general:temporal"
    # Interval should be narrower than [0, 1] for a well-calibrated model
    width = result.interval_upper - result.interval_lower
    assert width < 1.0
    # And wrap the point estimate
    assert result.interval_lower <= result.p_true <= result.interval_upper


@pytest.mark.asyncio
async def test_empirical_coverage_roughly_on_target() -> None:
    """With a known calibration distribution, verify that the computed
    intervals cover the truth at approximately the target rate on a
    held-out split. This is the actual conformal guarantee."""
    rng = random.Random(1234)

    # Generate a calibration set of 200 entries where:
    # - true claims have posterior ≈ 0.8 (with noise)
    # - false claims have posterior ≈ 0.2 (with noise)
    store = InMemoryCalibrationStore()
    for _ in range(200):
        truth = rng.random() < 0.5
        posterior = (0.80 if truth else 0.20) + rng.uniform(-0.1, 0.1)
        posterior = max(0.0, min(1.0, posterior))
        store.add_entry("general:any", posterior=posterior, true_label=truth)

    calibrator = SplitConformalCalibrator(store)

    # Evaluate on an independent held-out set from the same distribution
    covered = 0
    total = 200
    for _ in range(total):
        truth = rng.random() < 0.5
        posterior = (0.80 if truth else 0.20) + rng.uniform(-0.1, 0.1)
        posterior = max(0.0, min(1.0, posterior))

        result = await calibrator.calibrate(
            _claim(), _belief(posterior), domain="general", stratum="general:any"
        )
        # Map boolean truth to probability-truth for interval coverage check
        truth_prob = 1.0 if truth else 0.0
        if result.interval_lower <= truth_prob <= result.interval_upper:
            covered += 1

    empirical_coverage = covered / total
    # Target 90% with ±7pt tolerance (generous for 200-sample held-out)
    assert 0.83 <= empirical_coverage <= 0.97, (
        f"Empirical coverage {empirical_coverage:.3f} outside tolerance of 0.90"
    )


# ---------------------------------------------------------------------------
# Non-converged beliefs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_converged_belief_gets_non_converged_fallback() -> None:
    """Spec §7: non-converged L4 beliefs are excluded from the coverage
    guarantee. Even with a populated calibration set, the result should
    be the non-converged fallback (interval = [0, 1], coverage = None)."""
    store = InMemoryCalibrationStore()
    for _ in range(100):
        store.add_entry("general:any", posterior=0.9, true_label=True)

    calibrator = SplitConformalCalibrator(store)
    result = await calibrator.calibrate(
        _claim(),
        _belief(0.6, converged=False),
        domain="general",
        stratum="general:any",
    )
    assert result.fallback_used == "non_converged"
    assert result.coverage_target is None
    assert result.interval_lower == 0.0
    assert result.interval_upper == 1.0
    # Point estimate still emitted best-effort
    assert result.p_true == 0.6


# ---------------------------------------------------------------------------
# CalibrationStore edge cases
# ---------------------------------------------------------------------------


def test_store_rejects_invalid_posterior() -> None:
    store = InMemoryCalibrationStore()
    with pytest.raises(ValueError):
        store.add_entry("general:any", posterior=1.5, true_label=True)


def test_store_empty_partition_has_zero_n() -> None:
    store = InMemoryCalibrationStore()
    assert store.get_n("general:any") == 0


def test_store_tracks_counts_per_partition() -> None:
    store = InMemoryCalibrationStore()
    store.add_entry("general:any", 0.5, true_label=True)
    store.add_entry("general:temporal", 0.5, true_label=True)
    store.add_entry("general:any", 0.5, true_label=False)
    assert store.get_n("general:any") == 2
    assert store.get_n("general:temporal") == 1
    assert store.get_n("biomedical:any") == 0
