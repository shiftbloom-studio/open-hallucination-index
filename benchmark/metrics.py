"""
Benchmark Metrics
=================

Statistical metrics for classification evaluation including:
- Confusion Matrix with all derived metrics
- Calibration metrics (Brier Score, ECE, MCE)
- Latency statistics with percentiles
- ROC and PR curve analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Confusion Matrix
# =============================================================================


@dataclass
class ConfusionMatrix:
    """
    Confusion matrix with comprehensive derived metrics.

    For hallucination detection:
    - Positive class: Factual (label=True)
    - Negative class: Hallucination (label=False)

    Critical metrics for this domain:
    - hallucination_pass_rate: Rate of hallucinations marked as safe (FP rate)
    - mcc: Matthews Correlation Coefficient, best for imbalanced data
    """

    tp: int = 0  # True Positive: fact correctly identified
    tn: int = 0  # True Negative: hallucination correctly identified
    fp: int = 0  # False Positive: hallucination marked as fact (DANGEROUS)
    fn: int = 0  # False Negative: fact marked as hallucination

    @property
    def total(self) -> int:
        """Total number of predictions."""
        return self.tp + self.tn + self.fp + self.fn

    @property
    def positives(self) -> int:
        """Total actual positives (facts)."""
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        """Total actual negatives (hallucinations)."""
        return self.tn + self.fp

    @property
    def predicted_positives(self) -> int:
        """Total predicted positives."""
        return self.tp + self.fp

    @property
    def predicted_negatives(self) -> int:
        """Total predicted negatives."""
        return self.tn + self.fn

    # -------------------------------------------------------------------------
    # Standard Classification Metrics
    # -------------------------------------------------------------------------

    @property
    def accuracy(self) -> float:
        """Overall accuracy: (TP + TN) / Total."""
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision (PPV): TP / (TP + FP)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall (Sensitivity, TPR): TP / (TP + FN)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def sensitivity(self) -> float:
        """Alias for recall."""
        return self.recall

    @property
    def specificity(self) -> float:
        """Specificity (TNR): TN / (TN + FP)."""
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def f2_score(self) -> float:
        """F2 Score: Weighted F-score emphasizing recall (β=2)."""
        return self._fbeta_score(beta=2.0)

    @property
    def f05_score(self) -> float:
        """F0.5 Score: Weighted F-score emphasizing precision (β=0.5)."""
        return self._fbeta_score(beta=0.5)

    def _fbeta_score(self, beta: float) -> float:
        """Generalized F-beta score."""
        p, r = self.precision, self.recall
        beta_sq = beta**2
        denom = beta_sq * p + r
        return (1 + beta_sq) * p * r / denom if denom > 0 else 0.0

    # -------------------------------------------------------------------------
    # Advanced Metrics
    # -------------------------------------------------------------------------

    @property
    def mcc(self) -> float:
        """
        Matthews Correlation Coefficient.

        Ranges from -1 to +1. Best metric for imbalanced datasets.
        +1: perfect prediction
         0: random prediction
        -1: inverse prediction
        """
        num = self.tp * self.tn - self.fp * self.fn
        denom = math.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        return num / denom if denom > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """Balanced accuracy: (Sensitivity + Specificity) / 2."""
        return (self.recall + self.specificity) / 2

    @property
    def cohen_kappa(self) -> float:
        """
        Cohen's Kappa coefficient.

        Measures inter-rater agreement accounting for chance.
        """
        if self.total == 0:
            return 0.0

        p_o = self.accuracy
        p_e = (
            (self.positives * self.predicted_positives)
            + (self.negatives * self.predicted_negatives)
        ) / (self.total**2)

        return (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0

    @property
    def informedness(self) -> float:
        """
        Informedness (Youden's J statistic): Sensitivity + Specificity - 1.

        Also known as Bookmaker Informedness.
        Ranges from -1 to +1, with 0 indicating random performance.
        """
        return self.sensitivity + self.specificity - 1

    @property
    def markedness(self) -> float:
        """
        Markedness (deltaP): PPV + NPV - 1.

        The probability that a prediction is correct.
        """
        npv = self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0.0
        return self.precision + npv - 1

    # -------------------------------------------------------------------------
    # Hallucination-Specific Metrics
    # -------------------------------------------------------------------------

    @property
    def hallucination_pass_rate(self) -> float:
        """
        Rate of hallucinations incorrectly marked as safe.

        FP / (TN + FP) = 1 - Specificity = False Positive Rate

        THIS IS THE MOST CRITICAL METRIC FOR SAFETY.
        Lower is better. Target: < 5%
        """
        return self.fp / self.negatives if self.negatives > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Alias for hallucination_pass_rate."""
        return self.hallucination_pass_rate

    @property
    def false_negative_rate(self) -> float:
        """Rate of facts incorrectly marked as hallucinations: FN / (TP + FN)."""
        return self.fn / self.positives if self.positives > 0 else 0.0

    @property
    def negative_predictive_value(self) -> float:
        """NPV: TN / (TN + FN)."""
        denom = self.tn + self.fn
        return self.tn / denom if denom > 0 else 0.0

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def as_dict(self) -> dict[str, float]:
        """Export all metrics as a dictionary."""
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "total": self.total,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1_score": self.f1_score,
            "f2_score": self.f2_score,
            "mcc": self.mcc,
            "balanced_accuracy": self.balanced_accuracy,
            "cohen_kappa": self.cohen_kappa,
            "hallucination_pass_rate": self.hallucination_pass_rate,
        }

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        """Combine two confusion matrices."""
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


# =============================================================================
# Calibration Metrics
# =============================================================================


@dataclass
class CalibrationMetrics:
    """
    Calibration and probabilistic scoring metrics.

    Measures how well the predicted probabilities match actual outcomes.
    Well-calibrated models have predictions that match their true frequencies.
    """

    brier_score: float = 0.0  # Mean squared error of probability estimates
    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Maximum Calibration Error
    ace: float = 0.0  # Adaptive Calibration Error (equal-size bins)
    reliability_diagram: list[tuple[float, float, int]] = field(default_factory=list)
    log_loss: float = 0.0  # Negative log-likelihood

    @staticmethod
    def compute(
        y_true: NDArray[np.int_],
        y_prob: NDArray[np.float64],
        n_bins: int = 10,
    ) -> "CalibrationMetrics":
        """
        Compute calibration metrics from predictions.

        Args:
            y_true: Ground truth labels (0 or 1).
            y_prob: Predicted probabilities.
            n_bins: Number of bins for ECE/reliability diagram.

        Returns:
            CalibrationMetrics instance with all computed values.
        """
        if len(y_true) == 0:
            return CalibrationMetrics()

        # Brier Score
        brier = float(np.mean((y_prob - y_true) ** 2))

        # Log Loss (with clipping to avoid log(0))
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        log_loss = float(
            -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
        )

        # ECE and reliability diagram
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])

        ece = 0.0
        mce = 0.0
        reliability = []

        for i in range(n_bins):
            mask = bin_indices == i
            bin_size = int(np.sum(mask))

            if bin_size > 0:
                bin_acc = float(np.mean(y_true[mask]))
                bin_conf = float(np.mean(y_prob[mask]))

                gap = abs(bin_acc - bin_conf)
                ece += (bin_size / len(y_true)) * gap
                mce = max(mce, gap)
                reliability.append((bin_conf, bin_acc, bin_size))

        # ACE: Adaptive Calibration Error (equal-mass bins)
        ace = CalibrationMetrics._compute_ace(y_true, y_prob, n_bins)

        return CalibrationMetrics(
            brier_score=brier,
            ece=float(ece),
            mce=float(mce),
            ace=ace,
            reliability_diagram=reliability,
            log_loss=log_loss,
        )

    @staticmethod
    def _compute_ace(
        y_true: NDArray[np.int_],
        y_prob: NDArray[np.float64],
        n_bins: int,
    ) -> float:
        """Compute Adaptive Calibration Error with equal-mass bins."""
        if len(y_true) < n_bins:
            return 0.0

        sorted_indices = np.argsort(y_prob)
        bin_size = len(y_true) // n_bins
        ace = 0.0

        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(y_true)
            indices = sorted_indices[start:end]

            if len(indices) > 0:
                bin_acc = float(np.mean(y_true[indices]))
                bin_conf = float(np.mean(y_prob[indices]))
                ace += abs(bin_acc - bin_conf) / n_bins

        return float(ace)


# =============================================================================
# Latency Statistics
# =============================================================================


@dataclass
class LatencyStats:
    """
    Latency statistics with percentiles.

    Provides comprehensive latency analysis for benchmarking.
    """

    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p75_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    iqr_ms: float = 0.0  # Interquartile range
    count: int = 0

    @staticmethod
    def compute(latencies: Sequence[float]) -> "LatencyStats":
        """
        Compute latency statistics from a list of latencies.

        Args:
            latencies: Sequence of latency values in milliseconds.

        Returns:
            LatencyStats instance with computed values.
        """
        if not latencies:
            return LatencyStats()

        arr = np.array(latencies)
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))

        return LatencyStats(
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p75_ms=p75,
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            iqr_ms=p75 - p25,
            count=len(arr),
        )

    @property
    def coefficient_of_variation(self) -> float:
        """CV: std / mean (measure of relative variability)."""
        return self.std_ms / self.mean_ms if self.mean_ms > 0 else 0.0


# =============================================================================
# ROC Analysis
# =============================================================================


@dataclass
class ROCAnalysis:
    """
    ROC curve analysis with optimal threshold detection.

    Computes ROC curve, AUC, and finds optimal threshold using Youden's J.
    """

    auc: float = 0.0
    optimal_threshold: float = 0.5
    youden_j: float = 0.0
    thresholds: list[float] = field(default_factory=list)
    tpr: list[float] = field(default_factory=list)  # True Positive Rate
    fpr: list[float] = field(default_factory=list)  # False Positive Rate

    @staticmethod
    def compute(
        y_true: NDArray[np.int_],
        y_scores: NDArray[np.float64],
    ) -> "ROCAnalysis":
        """
        Compute ROC curve and find optimal threshold using Youden's J.

        Args:
            y_true: Ground truth labels (0 or 1).
            y_scores: Predicted scores/probabilities.

        Returns:
            ROCAnalysis instance with curve data and optimal threshold.
        """
        if len(y_true) == 0:
            return ROCAnalysis()

        # Sort by scores descending
        sorted_idx = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_idx]
        y_scores_sorted = y_scores[sorted_idx]

        # Create thresholds
        unique_scores = np.unique(y_scores_sorted)
        thresholds = np.concatenate([[1.0 + 1e-10], unique_scores, [0.0 - 1e-10]])

        tpr_list: list[float] = []
        fpr_list: list[float] = []

        p = int(np.sum(y_true))
        n = len(y_true) - p

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))

            tpr = tp / p if p > 0 else 0.0
            fpr = fp / n if n > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i - 1] - fpr_list[i]) * (tpr_list[i - 1] + tpr_list[i]) / 2

        # Find optimal threshold using Youden's J
        youden_values = np.array(tpr_list) - np.array(fpr_list)
        best_idx = int(np.argmax(youden_values))

        return ROCAnalysis(
            auc=float(auc),
            optimal_threshold=float(thresholds[best_idx]),
            youden_j=float(youden_values[best_idx]),
            thresholds=thresholds.tolist(),
            tpr=tpr_list,
            fpr=fpr_list,
        )

    @property
    def sensitivity_at_90_specificity(self) -> float:
        """Find sensitivity at 90% specificity."""
        for i, fpr_val in enumerate(self.fpr):
            if fpr_val <= 0.10:  # Specificity = 1 - FPR >= 0.90
                return self.tpr[i]
        return 0.0

    @property
    def specificity_at_90_sensitivity(self) -> float:
        """Find specificity at 90% sensitivity."""
        for i, tpr_val in enumerate(self.tpr):
            if tpr_val >= 0.90:
                return 1.0 - self.fpr[i]
        return 0.0


# =============================================================================
# PR Curve Analysis
# =============================================================================


@dataclass
class PRCurveAnalysis:
    """
    Precision-Recall curve analysis.

    More informative than ROC for imbalanced datasets.
    """

    auc_pr: float = 0.0  # Area under PR curve
    average_precision: float = 0.0
    thresholds: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1_at_thresholds: list[float] = field(default_factory=list)
    optimal_threshold_f1: float = 0.5

    @staticmethod
    def compute(
        y_true: NDArray[np.int_],
        y_scores: NDArray[np.float64],
    ) -> "PRCurveAnalysis":
        """
        Compute Precision-Recall curve.

        Args:
            y_true: Ground truth labels (0 or 1).
            y_scores: Predicted scores/probabilities.

        Returns:
            PRCurveAnalysis instance.
        """
        if len(y_true) == 0:
            return PRCurveAnalysis()

        # Sort by scores descending
        sorted_idx = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_idx]
        y_scores_sorted = y_scores[sorted_idx]

        unique_scores = np.unique(y_scores_sorted)
        thresholds = np.concatenate([[1.0 + 1e-10], unique_scores, [0.0 - 1e-10]])

        precision_list: list[float] = []
        recall_list: list[float] = []
        f1_list: list[float] = []

        total_positives = int(np.sum(y_true))

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)

        # Compute AUC-PR using trapezoidal rule
        auc_pr = 0.0
        for i in range(1, len(recall_list)):
            auc_pr += (recall_list[i] - recall_list[i - 1]) * (
                precision_list[i] + precision_list[i - 1]
            ) / 2

        # Average Precision (step-function approximation)
        ap = 0.0
        for i in range(1, len(recall_list)):
            ap += (recall_list[i] - recall_list[i - 1]) * precision_list[i]

        # Find optimal threshold for F1
        best_f1_idx = int(np.argmax(f1_list))

        return PRCurveAnalysis(
            auc_pr=float(abs(auc_pr)),
            average_precision=float(abs(ap)),
            thresholds=thresholds.tolist(),
            precision=precision_list,
            recall=recall_list,
            f1_at_thresholds=f1_list,
            optimal_threshold_f1=float(thresholds[best_f1_idx]),
        )
