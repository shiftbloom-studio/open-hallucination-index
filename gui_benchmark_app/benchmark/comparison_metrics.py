"""
Comparison Benchmark Metrics
=============================

Extended metrics for multi-evaluator benchmark comparison:
- Hallucination Detection Metrics
- TruthfulQA Metrics (MC1, MC2, Generation)
- FActScore Metrics (Atomic Fact Precision)
- Latency Metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HallucinationMetrics:
    """
    Metrics for hallucination detection evaluation.
    
    Key safety metric: hallucination_pass_rate (lower is better)
    - Rate at which hallucinations are incorrectly marked as factual
    """
    
    total: int = 0
    correct: int = 0
    
    # Confusion matrix
    true_positives: int = 0   # Fact correctly identified
    true_negatives: int = 0   # Hallucination correctly identified
    false_positives: int = 0  # Hallucination marked as fact (DANGEROUS)
    false_negatives: int = 0  # Fact marked as hallucination
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 Score: Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """Specificity (TNR): TN / (TN + FP)."""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0
    
    @property
    def hallucination_pass_rate(self) -> float:
        """
        CRITICAL SAFETY METRIC: Rate of hallucinations marked as factual.
        
        Lower is better. FP / (FP + TN)
        """
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0
    
    @property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient."""
        import math
        num = self.true_positives * self.true_negatives - self.false_positives * self.false_negatives
        denom = math.sqrt(
            (self.true_positives + self.false_positives)
            * (self.true_positives + self.false_negatives)
            * (self.true_negatives + self.false_positives)
            * (self.true_negatives + self.false_negatives)
        )
        return num / denom if denom > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "specificity": round(self.specificity, 4),
            "hallucination_pass_rate": round(self.hallucination_pass_rate, 4),
            "mcc": round(self.mcc, 4),
            "confusion_matrix": {
                "tp": self.true_positives,
                "tn": self.true_negatives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
            },
        }


@dataclass
class TruthfulQAMetrics:
    """
    Metrics for TruthfulQA evaluation.
    
    Measures truthfulness in adversarial question answering.
    """
    
    total_questions: int = 0
    correct_predictions: int = 0
    
    # Detailed breakdown
    mc1_correct: int = 0  # Single correct answer
    mc1_total: int = 0
    mc2_correct: int = 0  # Multiple correct answers
    mc2_total: int = 0
    
    # Category-wise results
    category_results: dict[str, dict[str, int]] = field(default_factory=dict)
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy on TruthfulQA."""
        return self.correct_predictions / self.total_questions if self.total_questions > 0 else 0.0
    
    @property
    def mc1_accuracy(self) -> float:
        """MC1 accuracy (single correct answer)."""
        return self.mc1_correct / self.mc1_total if self.mc1_total > 0 else 0.0
    
    @property
    def mc2_accuracy(self) -> float:
        """MC2 accuracy (multiple correct answers)."""
        return self.mc2_correct / self.mc2_total if self.mc2_total > 0 else 0.0
    
    def get_category_accuracy(self, category: str) -> float:
        """Get accuracy for a specific category."""
        if category not in self.category_results:
            return 0.0
        result = self.category_results[category]
        total = result.get("total", 0)
        correct = result.get("correct", 0)
        return correct / total if total > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_questions": self.total_questions,
            "accuracy": round(self.accuracy, 4),
            "mc1_accuracy": round(self.mc1_accuracy, 4),
            "mc2_accuracy": round(self.mc2_accuracy, 4),
            "category_results": {
                cat: {
                    "accuracy": round(self.get_category_accuracy(cat), 4),
                    **result,
                }
                for cat, result in self.category_results.items()
            },
        }


@dataclass
class FActScoreMetrics:
    """
    Metrics for FActScore evaluation.
    
    FActScore = (# supported facts) / (# total facts)
    Measures atomic fact precision in longer-form text.
    """
    
    total_texts: int = 0
    total_facts: int = 0
    supported_facts: int = 0
    
    # Per-text scores
    scores: list[float] = field(default_factory=list)
    facts_per_text: list[int] = field(default_factory=list)
    
    @property
    def factscore(self) -> float:
        """Overall FActScore (atomic fact precision)."""
        return self.supported_facts / self.total_facts if self.total_facts > 0 else 0.0
    
    @property
    def avg_factscore(self) -> float:
        """Average FActScore across all texts."""
        return float(np.mean(self.scores)) if self.scores else 0.0
    
    @property
    def std_factscore(self) -> float:
        """Standard deviation of FActScores."""
        return float(np.std(self.scores)) if len(self.scores) > 1 else 0.0
    
    @property
    def avg_facts_per_text(self) -> float:
        """Average atomic facts extracted per text."""
        return float(np.mean(self.facts_per_text)) if self.facts_per_text else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_texts": self.total_texts,
            "total_facts": self.total_facts,
            "supported_facts": self.supported_facts,
            "factscore": round(self.factscore, 4),
            "avg_factscore": round(self.avg_factscore, 4),
            "std_factscore": round(self.std_factscore, 4),
            "avg_facts_per_text": round(self.avg_facts_per_text, 2),
        }


@dataclass
class LatencyMetrics:
    """
    Latency performance metrics.
    
    Comprehensive latency statistics for evaluation.
    """
    
    latencies_ms: list[float] = field(default_factory=list)
    
    @property
    def total_requests(self) -> int:
        return len(self.latencies_ms)
    
    @property
    def mean(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0
    
    @property
    def median(self) -> float:
        return float(np.median(self.latencies_ms)) if self.latencies_ms else 0.0
    
    @property
    def std(self) -> float:
        return float(np.std(self.latencies_ms)) if len(self.latencies_ms) > 1 else 0.0
    
    @property
    def p50(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0
    
    @property
    def p90(self) -> float:
        return float(np.percentile(self.latencies_ms, 90)) if self.latencies_ms else 0.0
    
    @property
    def p95(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0
    
    @property
    def p99(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0
    
    @property
    def min(self) -> float:
        return float(np.min(self.latencies_ms)) if self.latencies_ms else 0.0
    
    @property
    def max(self) -> float:
        return float(np.max(self.latencies_ms)) if self.latencies_ms else 0.0
    
    @property
    def throughput(self) -> float:
        """Estimated throughput based on P50 latency (req/s)."""
        return 1000.0 / self.p50 if self.p50 > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_requests": self.total_requests,
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "std_ms": round(self.std, 2),
            "p50_ms": round(self.p50, 2),
            "p90_ms": round(self.p90, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "throughput_rps": round(self.throughput, 2),
        }


@dataclass
class EvaluatorMetrics:
    """
    Combined metrics for a single evaluator.
    
    Aggregates all metric types for comparison.
    """
    
    evaluator_name: str
    hallucination: HallucinationMetrics = field(default_factory=HallucinationMetrics)
    truthfulqa: TruthfulQAMetrics = field(default_factory=TruthfulQAMetrics)
    factscore: FActScoreMetrics = field(default_factory=FActScoreMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "evaluator": self.evaluator_name,
            "hallucination": self.hallucination.to_dict(),
            "truthfulqa": self.truthfulqa.to_dict(),
            "factscore": self.factscore.to_dict(),
            "latency": self.latency.to_dict(),
        }
    
    def get_summary_scores(self) -> dict[str, float]:
        """
        Get summary scores for radar chart visualization.
        
        All scores normalized to 0-1 range where higher is better.
        """
        return {
            "Accuracy": self.hallucination.accuracy,
            "Precision": self.hallucination.precision,
            "Recall": self.hallucination.recall,
            "F1 Score": self.hallucination.f1_score,
            "Safety (1-HPR)": 1.0 - self.hallucination.hallucination_pass_rate,
            "TruthfulQA": self.truthfulqa.accuracy,
            "FActScore": self.factscore.avg_factscore,
            "Speed (1/P95)": min(1.0, 1000.0 / self.latency.p95) if self.latency.p95 > 0 else 0.0,
        }


@dataclass
class ComparisonReport:
    """
    Complete comparison report across all evaluators.
    """
    
    evaluators: dict[str, EvaluatorMetrics] = field(default_factory=dict)
    run_id: str = ""
    timestamp: str = ""
    config_summary: dict[str, Any] = field(default_factory=dict)
    
    def add_evaluator(self, metrics: EvaluatorMetrics) -> None:
        """Add evaluator metrics to report."""
        self.evaluators[metrics.evaluator_name] = metrics
    
    def get_ranking(self, metric: str = "f1_score") -> list[str]:
        """
        Get evaluators ranked by a specific metric.
        
        Args:
            metric: Metric name (f1_score, accuracy, factscore, etc.)
            
        Returns:
            List of evaluator names sorted by metric (best first)
        """
        scores: list[tuple[str, float]] = []
        
        for name, metrics in self.evaluators.items():
            if metric == "accuracy":
                score = metrics.hallucination.accuracy
            elif metric == "f1_score":
                score = metrics.hallucination.f1_score
            elif metric == "factscore":
                score = metrics.factscore.avg_factscore
            elif metric == "truthfulqa":
                score = metrics.truthfulqa.accuracy
            elif metric == "latency":
                # Lower latency is better, so negate
                score = -metrics.latency.p95
            elif metric == "safety":
                # Lower hallucination pass rate is better
                score = -metrics.hallucination.hallucination_pass_rate
            else:
                score = 0.0
            
            scores.append((name, score))
        
        # Sort descending (highest score first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scores]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert full report to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config_summary,
            "evaluators": {
                name: metrics.to_dict()
                for name, metrics in self.evaluators.items()
            },
            "rankings": {
                "by_f1": self.get_ranking("f1_score"),
                "by_accuracy": self.get_ranking("accuracy"),
                "by_safety": self.get_ranking("safety"),
                "by_factscore": self.get_ranking("factscore"),
                "by_latency": self.get_ranking("latency"),
            },
        }
    
    def get_comparison_table(self) -> list[dict[str, Any]]:
        """
        Generate comparison table data for visualization.
        
        Returns:
            List of rows with evaluator name and all metrics.
        """
        rows: list[dict[str, Any]] = []
        
        for name, metrics in self.evaluators.items():
            row = {
                "Evaluator": name,
                "Accuracy": f"{metrics.hallucination.accuracy:.1%}",
                "Precision": f"{metrics.hallucination.precision:.1%}",
                "Recall": f"{metrics.hallucination.recall:.1%}",
                "F1 Score": f"{metrics.hallucination.f1_score:.1%}",
                "Halluc. Pass Rate": f"{metrics.hallucination.hallucination_pass_rate:.1%}",
                "TruthfulQA": f"{metrics.truthfulqa.accuracy:.1%}",
                "FActScore": f"{metrics.factscore.avg_factscore:.1%}",
                "P50 Latency": f"{metrics.latency.p50:.0f}ms",
                "P95 Latency": f"{metrics.latency.p95:.0f}ms",
                "Throughput": f"{metrics.latency.throughput:.1f} req/s",
            }
            rows.append(row)
        
        return rows
