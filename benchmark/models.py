"""
Benchmark Data Models
=====================

Core data structures for benchmark cases, results, and reports.
Uses frozen dataclasses for immutability and hashability where appropriate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class VerificationStrategy(str, Enum):
    """Available OHI verification strategies."""

    VECTOR = "vector_semantic"
    GRAPH = "graph_exact"
    HYBRID = "hybrid"
    CASCADING = "cascading"
    MCP = "mcp_enhanced"
    ADAPTIVE = "adaptive"

    @classmethod
    def all_values(cls) -> list[str]:
        """Get all strategy values as strings."""
        return [s.value for s in cls]

    @classmethod
    def from_string(cls, value: str) -> "VerificationStrategy":
        """Parse strategy from string value."""
        for strategy in cls:
            if strategy.value == value:
                return strategy
        raise ValueError(f"Unknown strategy: {value}")


class DifficultyLevel(str, Enum):
    """Difficulty levels for benchmark cases."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        """Numerical weight for difficulty-weighted metrics."""
        weights = {
            DifficultyLevel.EASY: 1.0,
            DifficultyLevel.MEDIUM: 1.5,
            DifficultyLevel.HARD: 2.0,
            DifficultyLevel.CRITICAL: 3.0,
        }
        return weights.get(self, 1.0)


class HallucinationType(str, Enum):
    """Types of hallucination patterns in benchmark dataset."""

    ENTITY_SWAP = "entity_swap"
    ATTRIBUTE_ERROR = "attribute_error"
    TEMPORAL_ERROR = "temporal_error"
    NUMERICAL_ERROR = "numerical_error"
    NEGATION = "negation"
    FABRICATION = "fabrication"
    PARTIAL_TRUTH = "partial_truth"  # Multi-claim with mixed true/false
    CONTEXT_CONFUSION = "context_confusion"


# =============================================================================
# Core Data Classes
# =============================================================================


@dataclass(frozen=True)
class BenchmarkCase:
    """
    A single test case from the benchmark dataset.

    Frozen for hashability and to prevent accidental mutation.

    Attributes:
        id: Unique case identifier.
        domain: Subject domain (general, technical, medical, etc.).
        difficulty: Difficulty level of the claim.
        label: Ground truth - True if factual, False if hallucination.
        text: The claim text to verify.
        notes: Additional notes about the case.
        hallucination_type: Type of hallucination pattern (if applicable).
    """

    id: int
    domain: str
    difficulty: str
    label: bool  # True = factual, False = hallucination
    text: str
    notes: str = ""
    hallucination_type: str | None = None

    @cached_property
    def is_multi_claim(self) -> bool:
        """
        Detect if text contains multiple claims.

        Multi-claim texts are common hallucination patterns where
        one claim is true and another is false.
        """
        indicators = [
            " and ",
            " but ",
            " however ",
            " while ",
            " although ",
            " whereas ",
            "; ",
        ]
        text_lower = self.text.lower()
        return any(ind in text_lower for ind in indicators)

    @cached_property
    def text_hash(self) -> str:
        """Unique hash for deduplication and caching."""
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    @cached_property
    def word_count(self) -> int:
        """Number of words in the claim text."""
        return len(self.text.split())

    @cached_property
    def difficulty_level(self) -> DifficultyLevel:
        """Parse difficulty as enum."""
        try:
            return DifficultyLevel(self.difficulty.lower())
        except ValueError:
            return DifficultyLevel.MEDIUM


@dataclass
class ResultMetric:
    """
    Result from a single verification test.

    Mutable to allow post-processing and enrichment.

    Attributes:
        case_id: Reference to the BenchmarkCase.
        strategy: Verification strategy used.
        expected: Ground truth label.
        predicted: Predicted label (trust_score >= threshold).
        trust_score: Raw trust score from API (0.0 to 1.0).
        latency_ms: End-to-end latency in milliseconds.
        domain: Domain of the test case.
        difficulty: Difficulty level.
        is_multi_claim: Whether the case contains multiple claims.
        claims_count: Number of claims detected by the API.
        processing_time_api_ms: Internal API processing time.
        error: Error message if the request failed.
        response_id: API response ID for traceability.
        raw_response: Full API response for debugging.
    """

    case_id: int
    strategy: str
    expected: bool
    predicted: bool
    trust_score: float
    latency_ms: float
    domain: str = ""
    difficulty: str = ""
    is_multi_claim: bool = False
    claims_count: int = 0
    processing_time_api_ms: float = 0.0
    error: str | None = None
    response_id: str | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def is_correct(self) -> bool:
        """Whether the prediction matched the ground truth."""
        return self.expected == self.predicted

    @property
    def is_tp(self) -> bool:
        """True Positive: correctly identified fact as factual."""
        return self.expected and self.predicted

    @property
    def is_tn(self) -> bool:
        """True Negative: correctly identified hallucination as false."""
        return (not self.expected) and (not self.predicted)

    @property
    def is_fp(self) -> bool:
        """
        False Positive: hallucination misclassified as factual.

        This is the MOST DANGEROUS error type - the system believed a lie.
        """
        return (not self.expected) and self.predicted

    @property
    def is_fn(self) -> bool:
        """False Negative: fact misclassified as hallucination."""
        return self.expected and (not self.predicted)

    @property
    def has_error(self) -> bool:
        """Whether this result had an error."""
        return self.error is not None


@dataclass
class ConfidenceInterval:
    """
    Confidence interval for a metric.

    Attributes:
        point_estimate: The calculated metric value.
        lower: Lower bound of the CI.
        upper: Upper bound of the CI.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        method: Method used to compute the CI (bootstrap, wilson, etc.).
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    method: str = "bootstrap"

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower

    @property
    def margin_of_error(self) -> float:
        """Margin of error (half-width)."""
        return self.width / 2

    def contains(self, value: float) -> bool:
        """Check if a value falls within the interval."""
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        return f"{self.point_estimate:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


# =============================================================================
# Report Data Classes
# =============================================================================


@dataclass
class StrategyReport:
    """
    Comprehensive report for a single verification strategy.

    Contains all computed metrics, breakdowns, and analysis results.
    """

    strategy: str

    # Core metrics (imported from metrics module at runtime)
    confusion_matrix: Any  # ConfusionMatrix
    calibration: Any  # CalibrationMetrics
    latency: Any  # LatencyStats
    roc: Any  # ROCAnalysis
    pr_curve: Any | None = None  # PRCurveAnalysis

    # Stratified analysis
    by_domain: dict[str, Any] = field(default_factory=dict)
    by_difficulty: dict[str, Any] = field(default_factory=dict)
    by_hallucination_type: dict[str, Any] = field(default_factory=dict)

    # Multi-claim analysis
    multi_claim_cm: Any | None = None
    single_claim_cm: Any | None = None

    # Error analysis
    worst_fp_cases: list[dict] = field(default_factory=list)
    worst_fn_cases: list[dict] = field(default_factory=list)
    error_count: int = 0
    error_rate: float = 0.0

    # Confidence intervals for key metrics
    accuracy_ci: ConfidenceInterval | None = None
    f1_ci: ConfidenceInterval | None = None
    auc_ci: ConfidenceInterval | None = None
    mcc_ci: ConfidenceInterval | None = None


@dataclass
class StatisticalComparison:
    """
    Statistical comparison between two strategies.

    Attributes:
        strategy_a: First strategy name.
        strategy_b: Second strategy name.
        mcnemar_chi2: McNemar's test chi-squared statistic.
        mcnemar_p_value: McNemar's test p-value.
        delong_z: DeLong test Z statistic for AUC comparison.
        delong_p_value: DeLong test p-value.
        auc_difference: AUC(A) - AUC(B).
        accuracy_difference: Accuracy(A) - Accuracy(B).
        is_significant: Whether the difference is statistically significant.
    """

    strategy_a: str
    strategy_b: str
    mcnemar_chi2: float = 0.0
    mcnemar_p_value: float = 1.0
    delong_z: float = 0.0
    delong_p_value: float = 1.0
    auc_difference: float = 0.0
    accuracy_difference: float = 0.0
    is_significant: bool = False
    significance_level: float = 0.05


@dataclass
class BenchmarkReport:
    """
    Complete benchmark report across all strategies.

    The top-level report containing all results and analysis.
    """

    run_id: str
    timestamp: str
    dataset_path: str
    dataset_size: int
    threshold_used: float
    strategies_tested: list[str]

    # Per-strategy reports
    strategy_reports: dict[str, StrategyReport] = field(default_factory=dict)

    # Best performers
    best_strategy_accuracy: str = ""
    best_strategy_f1: str = ""
    best_strategy_mcc: str = ""
    best_strategy_auc: str = ""
    lowest_hallucination_rate: str = ""

    # Statistical comparisons
    comparisons: dict[str, StatisticalComparison] = field(default_factory=dict)

    # Legacy compatibility (deprecated, use comparisons instead)
    mcnemar_results: dict[str, dict[str, float]] = field(default_factory=dict)

    # Execution metadata
    total_runtime_seconds: float = 0.0
    api_health: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def created_at(self) -> datetime:
        """Parse timestamp to datetime."""
        return datetime.fromisoformat(self.timestamp)

    def get_ranking(self, metric: str = "f1_score") -> list[tuple[str, float]]:
        """
        Get strategies ranked by a specific metric.

        Args:
            metric: Metric name (accuracy, f1_score, mcc, auc).

        Returns:
            List of (strategy, value) tuples sorted descending.
        """
        rankings = []
        for strat, report in self.strategy_reports.items():
            if metric == "auc":
                value = report.roc.auc if report.roc else 0.0
            elif hasattr(report.confusion_matrix, metric):
                value = getattr(report.confusion_matrix, metric)
            else:
                value = 0.0
            rankings.append((strat, value))

        return sorted(rankings, key=lambda x: x[1], reverse=True)
