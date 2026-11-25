"""Benchmark metrics: F1, ECE, calibration coverage, per-domain aggregation.

All functions use generic key-extractor callables so the same
implementation scores any benchmark (binary, multi-class, calibrated, or
not). Metric functions silently drop errored results (where
`BenchmarkResult.error is not None`) and return `nan` when a metric is
undefined for the given inputs (e.g. ECE on a result set with no
probabilities).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

from scripts.benchmark.types import BenchmarkResult


# ---------------------------------------------------------------------------
# F1 (macro-averaged)
# ---------------------------------------------------------------------------


def compute_f1(
    results: list[BenchmarkResult],
    predicted_label_fn: Callable[[BenchmarkResult], str],
    true_label_fn: Callable[[str], str],
) -> float:
    """Macro-averaged F1 over all distinct labels present in the data.

    Errored results (those with `error is not None`) are dropped before
    scoring so a partial benchmark run still produces a meaningful number.
    """
    scored = [r for r in results if r.error is None]
    if not scored:
        return float("nan")

    # Collect the union of predicted and true labels
    labels: set[str] = set()
    for r in scored:
        labels.add(predicted_label_fn(r))
        labels.add(true_label_fn(r.example_id))

    per_label_f1: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for r in scored:
            pred = predicted_label_fn(r)
            truth = true_label_fn(r.example_id)
            if pred == label and truth == label:
                tp += 1
            elif pred == label and truth != label:
                fp += 1
            elif pred != label and truth == label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            per_label_f1.append(0.0)
        else:
            per_label_f1.append(2 * precision * recall / (precision + recall))

    return sum(per_label_f1) / len(per_label_f1) if per_label_f1 else float("nan")


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------


def compute_ece(
    results: list[BenchmarkResult],
    is_true_fn: Callable[[str], bool],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error using equal-width binning.

    ECE = sum over bins of (|bin| / N) * |bin_confidence - bin_accuracy|.

    Returns `nan` if no result has a `p_true` value.
    """
    calibrated = [r for r in results if r.error is None and r.p_true is not None]
    if not calibrated:
        return float("nan")

    n = len(calibrated)
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    weighted_error = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Right edge is inclusive for the last bin so p_true=1.0 lands somewhere
        bin_members = [
            r
            for r in calibrated
            if (lo <= r.p_true < hi) or (i == n_bins - 1 and r.p_true == hi)  # type: ignore[operator]
        ]
        if not bin_members:
            continue
        bin_confidence = sum(r.p_true for r in bin_members) / len(bin_members)  # type: ignore[misc]
        bin_accuracy = sum(1 for r in bin_members if is_true_fn(r.example_id)) / len(
            bin_members
        )
        weighted_error += (len(bin_members) / n) * abs(bin_confidence - bin_accuracy)

    return weighted_error


# ---------------------------------------------------------------------------
# Calibration coverage
# ---------------------------------------------------------------------------


def compute_calibration_coverage(
    results: list[BenchmarkResult],
    truth_prob_fn: Callable[[str], float],
    *,
    target: float = 0.90,
) -> float:
    """Empirical interval coverage = fraction of results whose conformal
    interval contains the true probability.

    `target` is informational (the target coverage the intervals were
    constructed to meet). Returns `nan` if no result has an interval.
    """
    del (
        target
    )  # retained for API symmetry; actual coverage target lives in result metadata

    with_intervals = [r for r in results if r.error is None and r.interval is not None]
    if not with_intervals:
        return float("nan")

    covered = 0
    for r in with_intervals:
        lo, hi = r.interval  # type: ignore[misc]
        truth = truth_prob_fn(r.example_id)
        if lo <= truth <= hi:
            covered += 1

    return covered / len(with_intervals)


# ---------------------------------------------------------------------------
# Per-domain aggregation
# ---------------------------------------------------------------------------


def aggregate_per_domain(
    results: list[BenchmarkResult],
    domain_fn: Callable[[BenchmarkResult], str],
) -> dict[str, list[BenchmarkResult]]:
    """Partition results by a caller-supplied domain extractor.

    Useful for Phase 3+ where the full benchmark suite runs across five
    domains (general / biomedical / legal / code / social) and each domain
    needs its own F1 + ECE numbers.
    """
    by_domain: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_domain[domain_fn(r)].append(r)
    return dict(by_domain)


# ---------------------------------------------------------------------------
# Convenience: summary dict for JSON output
# ---------------------------------------------------------------------------


def summarize(
    results: list[BenchmarkResult],
    *,
    predicted_label_fn: Callable[[BenchmarkResult], str],
    true_label_fn: Callable[[str], str],
    is_true_fn: Callable[[str], bool] | None = None,
) -> dict[str, float | int]:
    """Produce a summary dict ready for JSON serialization."""
    summary: dict[str, float | int] = {
        "n_total": len(results),
        "n_errored": sum(1 for r in results if r.error is not None),
        "f1_macro": compute_f1(results, predicted_label_fn, true_label_fn),
        "mean_latency_ms": (
            sum(r.latency_ms for r in results if r.error is None)
            / max(1, sum(1 for r in results if r.error is None))
        ),
    }
    if is_true_fn is not None:
        ece = compute_ece(results, is_true_fn)
        summary["ece"] = ece if not math.isnan(ece) else 0.0
    return summary
