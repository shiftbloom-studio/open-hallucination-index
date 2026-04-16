"""Unit tests for scripts/benchmark/metrics.py.

All tests are deterministic and synthetic — they exercise the metric
implementations against known inputs with closed-form expected outputs.
No network, no model, no filesystem.
"""

from __future__ import annotations

import math

import pytest

from scripts.benchmark.metrics import (
    aggregate_per_domain,
    compute_calibration_coverage,
    compute_ece,
    compute_f1,
)
from scripts.benchmark.types import BenchmarkResult


def _result(
    *,
    example_id: str = "x",
    predicted_label: str = "true",
    p_true: float | None = None,
    interval: tuple[float, float] | None = None,
    metadata: dict | None = None,
    error: str | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        example_id=example_id,
        predicted_label=predicted_label,
        p_true=p_true,
        interval=interval,
        raw_response={"metadata": metadata or {}},
        latency_ms=1.0,
        error=error,
    )


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------


def test_f1_perfect_prediction_is_one() -> None:
    results = [_result(example_id=str(i), predicted_label="true") for i in range(5)]
    true_labels = {str(i): "true" for i in range(5)}
    f1 = compute_f1(results, lambda r: r.predicted_label, true_labels.__getitem__)
    assert f1 == pytest.approx(1.0)


def test_f1_all_wrong_is_zero() -> None:
    results = [_result(example_id=str(i), predicted_label="false") for i in range(5)]
    true_labels = {str(i): "true" for i in range(5)}
    f1 = compute_f1(results, lambda r: r.predicted_label, true_labels.__getitem__)
    assert f1 == pytest.approx(0.0)


def test_f1_balanced_classes_macro() -> None:
    # 2 true / 2 false, predictions: 1 correct each → macro-F1 = 0.5
    preds = ["true", "false", "true", "false"]
    truth = ["true", "true", "false", "false"]
    results = [
        _result(example_id=str(i), predicted_label=p) for i, p in enumerate(preds)
    ]
    true_labels = {str(i): t for i, t in enumerate(truth)}
    f1 = compute_f1(results, lambda r: r.predicted_label, true_labels.__getitem__)
    assert f1 == pytest.approx(0.5, abs=0.01)


def test_f1_skips_errored_results() -> None:
    results = [
        _result(example_id="0", predicted_label="true"),
        _result(example_id="1", predicted_label="true", error="timeout"),
    ]
    true_labels = {"0": "true", "1": "true"}
    f1 = compute_f1(results, lambda r: r.predicted_label, true_labels.__getitem__)
    # Only one result scored; perfect
    assert f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_is_zero() -> None:
    # Predicted p_true=1.0 and true label is "true" for every example → ECE=0
    results = [_result(example_id=str(i), p_true=1.0) for i in range(10)]
    true_labels = {str(i): "true" for i in range(10)}
    ece = compute_ece(results, lambda r_id: true_labels[r_id] == "true", n_bins=10)
    assert ece == pytest.approx(0.0)


def test_ece_maximally_miscalibrated_is_one() -> None:
    # Predicted p_true=1.0 but true label is always "false" → ECE=1
    results = [_result(example_id=str(i), p_true=1.0) for i in range(10)]
    true_labels = {str(i): "false" for i in range(10)}
    ece = compute_ece(results, lambda r_id: true_labels[r_id] == "true", n_bins=10)
    assert ece == pytest.approx(1.0)


def test_ece_bin_averaging() -> None:
    # 5 predictions in bin [0.4, 0.5): p=0.45, 3 true / 5 = 0.6
    # |bin_conf 0.45 - bin_acc 0.6| = 0.15 (weighted by 5/5)
    results = [_result(example_id=str(i), p_true=0.45) for i in range(5)]
    truth = {"0": True, "1": True, "2": True, "3": False, "4": False}
    ece = compute_ece(results, lambda r_id: truth[r_id], n_bins=10)
    assert ece == pytest.approx(0.15, abs=0.01)


def test_ece_skips_none_p_true() -> None:
    results = [_result(example_id=str(i), p_true=None) for i in range(3)]
    truth = {"0": True, "1": True, "2": True}
    ece = compute_ece(results, lambda r_id: truth[r_id], n_bins=10)
    # No calibrated probabilities → undefined; return NaN
    assert math.isnan(ece)


# ---------------------------------------------------------------------------
# Calibration coverage
# ---------------------------------------------------------------------------


def test_coverage_all_intervals_contain_truth_is_one() -> None:
    # interval [0.4, 0.8], truth = 0.7 (inside) → coverage 1.0
    results = [_result(example_id=str(i), interval=(0.4, 0.8)) for i in range(5)]
    truth_probs = {str(i): 0.7 for i in range(5)}
    coverage = compute_calibration_coverage(
        results, truth_probs.__getitem__, target=0.90
    )
    assert coverage == pytest.approx(1.0)


def test_coverage_no_intervals_contain_truth_is_zero() -> None:
    results = [_result(example_id=str(i), interval=(0.4, 0.6)) for i in range(5)]
    truth_probs = {str(i): 0.9 for i in range(5)}
    coverage = compute_calibration_coverage(
        results, truth_probs.__getitem__, target=0.90
    )
    assert coverage == pytest.approx(0.0)


def test_coverage_half_contain_truth_is_half() -> None:
    results = [
        _result(example_id="0", interval=(0.4, 0.6)),  # truth 0.5 → inside
        _result(example_id="1", interval=(0.4, 0.6)),  # truth 0.9 → outside
    ]
    truth = {"0": 0.5, "1": 0.9}
    coverage = compute_calibration_coverage(results, truth.__getitem__, target=0.90)
    assert coverage == pytest.approx(0.5)


def test_coverage_returns_nan_when_no_intervals_present() -> None:
    results = [_result(example_id=str(i), interval=None) for i in range(3)]
    truth = {"0": 0.5, "1": 0.5, "2": 0.5}
    coverage = compute_calibration_coverage(results, truth.__getitem__, target=0.90)
    assert math.isnan(coverage)


# ---------------------------------------------------------------------------
# Per-domain aggregation
# ---------------------------------------------------------------------------


def test_aggregate_per_domain_partitions_results() -> None:
    results = [
        _result(example_id="0", metadata={"domain": "general"}),
        _result(example_id="1", metadata={"domain": "general"}),
        _result(example_id="2", metadata={"domain": "biomedical"}),
    ]
    by_domain = aggregate_per_domain(
        results,
        lambda r: r.raw_response["metadata"]["domain"],
    )
    assert set(by_domain.keys()) == {"general", "biomedical"}
    assert len(by_domain["general"]) == 2
    assert len(by_domain["biomedical"]) == 1
