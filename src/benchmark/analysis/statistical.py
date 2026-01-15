"""
Statistical Significance Testing
=================================

Research-grade statistical tests for classifier comparison:
- McNemar's Test for paired binary classifiers
- DeLong Test for AUC comparison
- Bootstrap Confidence Intervals for all metrics
- Permutation Tests for significance

References:
-----------
* DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. (1988).
  "Comparing the areas under two or more correlated receiver
  operating characteristic curves: a nonparametric approach."
  Biometrics, 44(3), 837-845.

* McNemar, Q. (1947). "Note on the sampling error of the
  difference between correlated proportions or percentages."
  Psychometrika, 12(2), 153-157.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from benchmark.models import ConfidenceInterval, ResultMetric


# =============================================================================
# McNemar's Test
# =============================================================================


@dataclass
class McNemarResult:
    """
    Result of McNemar's test for paired binary classifiers.

    Attributes:
        chi2: Chi-squared statistic (with continuity correction).
        chi2_uncorrected: Chi-squared without continuity correction.
        p_value: Two-sided p-value.
        exact_p_value: Exact p-value (binomial test).
        n_concordant: Number of cases where both agree.
        n_discordant_ab: Cases where A correct, B wrong.
        n_discordant_ba: Cases where A wrong, B correct.
        odds_ratio: Odds ratio for disagreements.
        is_significant: Whether p < alpha.
    """

    chi2: float
    chi2_uncorrected: float
    p_value: float
    exact_p_value: float
    n_concordant: int
    n_discordant_ab: int
    n_discordant_ba: int
    odds_ratio: float
    is_significant: bool
    alpha: float = 0.05


def mcnemar_test(
    predictions_a: NDArray[np.bool_],
    predictions_b: NDArray[np.bool_],
    ground_truth: NDArray[np.bool_],
    alpha: float = 0.05,
) -> McNemarResult:
    """
    Perform McNemar's test comparing two classifiers on paired data.

    McNemar's test is the gold standard for comparing two classifiers
    on the same dataset. It only considers discordant pairs where
    the classifiers disagree.

    Args:
        predictions_a: Boolean predictions from classifier A.
        predictions_b: Boolean predictions from classifier B.
        ground_truth: Ground truth labels.
        alpha: Significance level.

    Returns:
        McNemarResult with test statistics.
    """
    # Build contingency table of correctness
    correct_a = predictions_a == ground_truth
    correct_b = predictions_b == ground_truth

    # Count cells
    both_correct = np.sum(correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)
    a_correct_b_wrong = np.sum(correct_a & ~correct_b)  # b
    a_wrong_b_correct = np.sum(~correct_a & correct_b)  # c

    b = int(a_correct_b_wrong)
    c = int(a_wrong_b_correct)
    n_concordant = int(both_correct + both_wrong)

    # McNemar's test with continuity correction
    if b + c > 0:
        chi2_corrected = (abs(b - c) - 1) ** 2 / (b + c)
        chi2_uncorrected = (b - c) ** 2 / (b + c)
        p_value = float(1 - scipy_stats.chi2.cdf(chi2_corrected, 1))
    else:
        chi2_corrected = 0.0
        chi2_uncorrected = 0.0
        p_value = 1.0

    # Exact test using binomial distribution
    # Under null hypothesis, P(success) = 0.5
    n = b + c
    if n > 0:
        exact_p = float(2 * min(
            scipy_stats.binom.cdf(min(b, c), n, 0.5),
            1 - scipy_stats.binom.cdf(max(b, c) - 1, n, 0.5),
        ))
        exact_p = min(exact_p, 1.0)
    else:
        exact_p = 1.0

    # Odds ratio
    odds_ratio = b / c if c > 0 else float("inf") if b > 0 else 1.0

    return McNemarResult(
        chi2=chi2_corrected,
        chi2_uncorrected=chi2_uncorrected,
        p_value=p_value,
        exact_p_value=exact_p,
        n_concordant=n_concordant,
        n_discordant_ab=b,
        n_discordant_ba=c,
        odds_ratio=odds_ratio,
        is_significant=p_value < alpha,
        alpha=alpha,
    )


def mcnemar_from_results(
    results_a: list[ResultMetric],
    results_b: list[ResultMetric],
    alpha: float = 0.05,
) -> McNemarResult:
    """
    Compute McNemar's test from ResultMetric lists.

    Args:
        results_a: Results from strategy A.
        results_b: Results from strategy B.
        alpha: Significance level.

    Returns:
        McNemarResult.
    """
    # Align results by case_id
    results_a_by_id = {r.case_id: r for r in results_a if not r.has_error}
    results_b_by_id = {r.case_id: r for r in results_b if not r.has_error}

    common_ids = set(results_a_by_id.keys()) & set(results_b_by_id.keys())

    pred_a = np.array([results_a_by_id[i].predicted for i in sorted(common_ids)])
    pred_b = np.array([results_b_by_id[i].predicted for i in sorted(common_ids)])
    truth = np.array([results_a_by_id[i].expected for i in sorted(common_ids)])

    return mcnemar_test(pred_a, pred_b, truth, alpha)


# =============================================================================
# DeLong Test for AUC Comparison
# =============================================================================


@dataclass
class DeLongResult:
    """
    Result of DeLong's test for comparing two AUCs.

    Attributes:
        auc_a: AUC of classifier A.
        auc_b: AUC of classifier B.
        auc_diff: AUC(A) - AUC(B).
        variance: Variance of the difference.
        z_statistic: Z-score for the difference.
        p_value: Two-sided p-value.
        ci_lower: Lower bound of 95% CI for AUC difference.
        ci_upper: Upper bound of 95% CI for AUC difference.
        is_significant: Whether the difference is significant.
    """

    auc_a: float
    auc_b: float
    auc_diff: float
    variance: float
    z_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    alpha: float = 0.05


def _compute_midrank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute midranks for tied values."""
    sorted_idx = np.argsort(x)
    n = len(x)
    ranks = np.zeros(n)

    i = 0
    while i < n:
        j = i
        while j < n and x[sorted_idx[j]] == x[sorted_idx[i]]:
            j += 1
        # Assign average rank to all tied values
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank
        i = j

    return ranks


def _delong_cov(
    y_true: NDArray[np.int_],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Compute DeLong covariance matrix for two classifiers.

    Based on DeLong, DeLong & Clarke-Pearson (1988).

    Returns:
        Tuple of (variance_a, variance_b, covariance).
    """
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    if n0 == 0 or n1 == 0:
        return (0.0, 0.0, 0.0)

    # Separate positive and negative scores
    pos_idx = y_true == 1
    neg_idx = y_true == 0

    # Compute structural components
    # For each positive, compute fraction of negatives with lower score
    v10_a = np.zeros(n1)
    v10_b = np.zeros(n1)
    pos_scores_a = scores_a[pos_idx]
    pos_scores_b = scores_b[pos_idx]
    neg_scores_a = scores_a[neg_idx]
    neg_scores_b = scores_b[neg_idx]

    for i in range(int(n1)):
        v10_a[i] = np.mean(neg_scores_a < pos_scores_a[i]) + 0.5 * np.mean(
            neg_scores_a == pos_scores_a[i]
        )
        v10_b[i] = np.mean(neg_scores_b < pos_scores_b[i]) + 0.5 * np.mean(
            neg_scores_b == pos_scores_b[i]
        )

    # For each negative, compute fraction of positives with higher score
    v01_a = np.zeros(n0)
    v01_b = np.zeros(n0)

    for i in range(int(n0)):
        v01_a[i] = np.mean(pos_scores_a > neg_scores_a[i]) + 0.5 * np.mean(
            pos_scores_a == neg_scores_a[i]
        )
        v01_b[i] = np.mean(pos_scores_b > neg_scores_b[i]) + 0.5 * np.mean(
            pos_scores_b == neg_scores_b[i]
        )

    # Compute variances and covariance
    s10_aa = np.var(v10_a, ddof=1) if n1 > 1 else 0.0
    s10_bb = np.var(v10_b, ddof=1) if n1 > 1 else 0.0
    s10_ab = np.cov(v10_a, v10_b, ddof=1)[0, 1] if n1 > 1 else 0.0

    s01_aa = np.var(v01_a, ddof=1) if n0 > 1 else 0.0
    s01_bb = np.var(v01_b, ddof=1) if n0 > 1 else 0.0
    s01_ab = np.cov(v01_a, v01_b, ddof=1)[0, 1] if n0 > 1 else 0.0

    var_a = s10_aa / n1 + s01_aa / n0
    var_b = s10_bb / n1 + s01_bb / n0
    cov_ab = s10_ab / n1 + s01_ab / n0

    return (float(var_a), float(var_b), float(cov_ab))


def delong_test(
    y_true: NDArray[np.int_],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    alpha: float = 0.05,
) -> DeLongResult:
    """
    DeLong test for comparing two correlated AUCs.

    This is the standard method for comparing two classifiers'
    ROC AUCs when evaluated on the same test set.

    Args:
        y_true: Ground truth labels (0 or 1).
        scores_a: Prediction scores from classifier A.
        scores_b: Prediction scores from classifier B.
        alpha: Significance level.

    Returns:
        DeLongResult with test statistics and confidence interval.
    """
    from benchmark.metrics import ROCAnalysis

    # Compute AUCs
    roc_a = ROCAnalysis.compute(y_true, scores_a)
    roc_b = ROCAnalysis.compute(y_true, scores_b)
    auc_a = roc_a.auc
    auc_b = roc_b.auc
    auc_diff = auc_a - auc_b

    # Compute covariance matrix
    var_a, var_b, cov_ab = _delong_cov(y_true, scores_a, scores_b)

    # Variance of the difference
    var_diff = var_a + var_b - 2 * cov_ab

    if var_diff > 0:
        z_stat = auc_diff / np.sqrt(var_diff)
        p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z_stat))))
    else:
        z_stat = 0.0
        p_value = 1.0 if abs(auc_diff) < 1e-10 else 0.0

    # Confidence interval for the difference
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var_diff) if var_diff > 0 else 0.0
    ci_lower = auc_diff - z_alpha * se
    ci_upper = auc_diff + z_alpha * se

    return DeLongResult(
        auc_a=auc_a,
        auc_b=auc_b,
        auc_diff=auc_diff,
        variance=var_diff,
        z_statistic=float(z_stat),
        p_value=p_value,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        is_significant=p_value < alpha,
        alpha=alpha,
    )


def delong_from_results(
    results_a: list[ResultMetric],
    results_b: list[ResultMetric],
    alpha: float = 0.05,
) -> DeLongResult:
    """
    Compute DeLong test from ResultMetric lists.

    Args:
        results_a: Results from strategy A.
        results_b: Results from strategy B.
        alpha: Significance level.

    Returns:
        DeLongResult.
    """
    results_a_by_id = {r.case_id: r for r in results_a if not r.has_error}
    results_b_by_id = {r.case_id: r for r in results_b if not r.has_error}

    common_ids = set(results_a_by_id.keys()) & set(results_b_by_id.keys())

    scores_a = np.array([results_a_by_id[i].trust_score for i in sorted(common_ids)])
    scores_b = np.array([results_b_by_id[i].trust_score for i in sorted(common_ids)])
    truth = np.array([1 if results_a_by_id[i].expected else 0 for i in sorted(common_ids)])

    return delong_test(truth, scores_a, scores_b, alpha)


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_ci(
    data: NDArray[np.float64],
    statistic_fn: Callable[[NDArray[np.float64]], float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for a statistic.

    Uses the percentile method which is robust and simple.

    Args:
        data: Input data array.
        statistic_fn: Function that computes the statistic from data.
        n_iterations: Number of bootstrap samples.
        confidence_level: Confidence level (0 to 1).
        random_state: Random seed for reproducibility.

    Returns:
        ConfidenceInterval with bootstrap CI.
    """
    rng = np.random.default_rng(random_state)
    n = len(data)

    if n == 0:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence_level,
            method="bootstrap",
        )

    point_estimate = statistic_fn(data)
    bootstrap_stats = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Sample with replacement
        indices = rng.integers(0, n, size=n)
        bootstrap_sample = data[indices]
        bootstrap_stats[i] = statistic_fn(bootstrap_sample)

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method="bootstrap",
    )


def bootstrap_metric_ci(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    metric_fn: Callable[[NDArray[np.int_], NDArray[np.int_]], float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> ConfidenceInterval:
    """
    Bootstrap CI for classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metric_fn: Function(y_true, y_pred) -> float.
        n_iterations: Bootstrap iterations.
        confidence_level: Confidence level.
        random_state: Random seed.

    Returns:
        ConfidenceInterval.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    if n == 0:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence_level,
            method="bootstrap",
        )

    point_estimate = metric_fn(y_true, y_pred)
    bootstrap_stats = np.zeros(n_iterations)

    for i in range(n_iterations):
        indices = rng.integers(0, n, size=n)
        bootstrap_stats[i] = metric_fn(y_true[indices], y_pred[indices])

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method="bootstrap",
    )


def bootstrap_auc_ci(
    y_true: NDArray[np.int_],
    y_scores: NDArray[np.float64],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> ConfidenceInterval:
    """
    Bootstrap CI for AUC-ROC.

    Args:
        y_true: Ground truth labels.
        y_scores: Prediction scores.
        n_iterations: Bootstrap iterations.
        confidence_level: Confidence level.
        random_state: Random seed.

    Returns:
        ConfidenceInterval for AUC.
    """
    from benchmark.metrics import ROCAnalysis

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    if n == 0:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence_level,
            method="bootstrap",
        )

    point_estimate = ROCAnalysis.compute(y_true, y_scores).auc
    bootstrap_stats = np.zeros(n_iterations)

    for i in range(n_iterations):
        indices = rng.integers(0, n, size=n)
        bootstrap_stats[i] = ROCAnalysis.compute(y_true[indices], y_scores[indices]).auc

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method="bootstrap",
    )


# =============================================================================
# Wilson Score Interval for Proportions
# =============================================================================


def wilson_ci(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Wilson score interval for a proportion.

    More accurate than normal approximation for small samples
    or extreme proportions.

    Args:
        successes: Number of successes.
        trials: Total number of trials.
        confidence_level: Confidence level.

    Returns:
        ConfidenceInterval using Wilson method.
    """
    if trials == 0:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence_level,
            method="wilson",
        )

    p = successes / trials
    z = scipy_stats.norm.ppf(1 - (1 - confidence_level) / 2)
    z2 = z * z

    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denominator

    return ConfidenceInterval(
        point_estimate=p,
        lower=float(max(0, center - spread)),
        upper=float(min(1, center + spread)),
        confidence_level=confidence_level,
        method="wilson",
    )
