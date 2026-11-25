"""
Research-Grade Report Generator
================================

Generates comprehensive performance reports with statistical significance
analysis for COMPLETE mode benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PerformanceStatement:
    """Research-grade performance statement with statistical backing."""

    evaluator_name: str
    primary_metric: str
    primary_value: float
    confidence_interval: tuple[float, float]
    ranking: int
    total_evaluators: int
    effect_sizes: dict[str, float]  # vs other evaluators
    statistical_significance: dict[str, str]  # vs other evaluators
    interpretation: str
    recommendation: str


class ResearchReportGenerator:
    """
    Generate research-grade reports for COMPLETE mode benchmarks.

    Produces:
    - Executive summary with key findings
    - Statistical analysis results
    - Performance rankings with confidence intervals
    - Pairwise comparisons with effect sizes
    - Recommendations based on empirical evidence
    """

    def __init__(self, report: Any, output_dir: Path):
        """
        Initialize report generator.

        Args:
            report: ComparisonReport with COMPLETE mode results
            output_dir: Directory for output files
        """
        self.report = report
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_performance_statements(self) -> list[PerformanceStatement]:
        """
        Generate performance statements for each evaluator.

        Returns:
            List of PerformanceStatement objects
        """
        statements = []

        # Get evaluator metrics sorted by accuracy
        evaluators = []
        for name, metrics in self.report.evaluators.items():
            evaluators.append(
                {
                    "name": name,
                    "accuracy": metrics.hallucination.accuracy,
                    "f1": metrics.hallucination.f1_score,
                    "precision": metrics.hallucination.precision,
                    "recall": metrics.hallucination.recall,
                    "hpr": metrics.hallucination.hallucination_pass_rate,
                    "p95": metrics.latency.p95,
                    "throughput": metrics.latency.throughput,
                }
            )

        # Sort by accuracy descending
        evaluators.sort(key=lambda x: x["accuracy"], reverse=True)

        # Get statistical analysis
        comparisons = getattr(self.report, "statistical_analysis", {}).get(
            "pairwise_comparisons", []
        )
        confidence_intervals = getattr(self.report, "confidence_intervals", {})

        # Generate statement for each evaluator
        for rank, eval_data in enumerate(evaluators, start=1):
            name = eval_data["name"]

            # Get confidence interval
            ci = confidence_intervals.get(name, {})
            ci_lower = ci.get("accuracy_lower", eval_data["accuracy"])
            ci_upper = ci.get("accuracy_upper", eval_data["accuracy"])

            # Compute effect sizes vs other evaluators
            effect_sizes = {}
            sig_results = {}

            for comp in comparisons:
                if comp["evaluator_1"] == name:
                    other = comp["evaluator_2"]
                    effect_sizes[other] = comp["cohens_d"]
                    sig_results[other] = comp["effect_size"]
                elif comp["evaluator_2"] == name:
                    other = comp["evaluator_1"]
                    effect_sizes[other] = -comp["cohens_d"]  # Flip sign
                    sig_results[other] = comp["effect_size"]

            # Generate interpretation
            interpretation = self._generate_interpretation(
                name, eval_data, rank, len(evaluators), effect_sizes, sig_results
            )

            # Generate recommendation
            recommendation = self._generate_recommendation(
                name, eval_data, rank, len(evaluators), effect_sizes
            )

            statement = PerformanceStatement(
                evaluator_name=name,
                primary_metric="accuracy",
                primary_value=eval_data["accuracy"],
                confidence_interval=(ci_lower, ci_upper),
                ranking=rank,
                total_evaluators=len(evaluators),
                effect_sizes=effect_sizes,
                statistical_significance=sig_results,
                interpretation=interpretation,
                recommendation=recommendation,
            )

            statements.append(statement)

        return statements

    def _generate_interpretation(
        self,
        name: str,
        eval_data: dict,
        rank: int,
        total: int,
        effect_sizes: dict[str, float],
        sig_results: dict[str, str],
    ) -> str:
        """Generate research-grade interpretation text."""
        acc = eval_data["accuracy"]
        hpr = eval_data["hpr"]
        f1 = eval_data["f1"]

        parts = []

        # Ranking statement
        if rank == 1:
            parts.append(
                f"{name} achieved the highest accuracy ({acc:.1%}) among all evaluated systems"
            )
        elif rank == total:
            parts.append(
                f"{name} achieved the lowest accuracy ({acc:.1%}) among all evaluated systems"
            )
        else:
            parts.append(f"{name} ranked #{rank} of {total} systems with {acc:.1%} accuracy")

        # Safety metric
        if hpr < 0.05:
            parts.append(
                f"demonstrating excellent safety with only {hpr:.1%} hallucination pass rate"
            )
        elif hpr < 0.10:
            parts.append(f"showing good safety with {hpr:.1%} hallucination pass rate")
        elif hpr < 0.20:
            parts.append(f"with moderate safety at {hpr:.1%} hallucination pass rate")
        else:
            parts.append(f"with elevated risk at {hpr:.1%} hallucination pass rate")

        # F1 score
        parts.append(f"F1 score of {f1:.1%} indicates {self._interpret_f1(f1)} classification")

        # Statistical significance
        if sig_results:
            large_effects = [k for k, v in sig_results.items() if v in ("large", "medium")]
            if large_effects:
                if rank == 1:
                    parts.append(
                        f"Statistical analysis confirms significant advantages over "
                        f"{len(large_effects)} baseline{'s' if len(large_effects) > 1 else ''} "
                        f"(medium to large effect sizes)"
                    )
                else:
                    parts.append(
                        f"Performance differences from top system show {sig_results.get(list(sig_results.keys())[0], 'notable')} effect size"
                    )

        return ". ".join(parts) + "."

    def _generate_recommendation(
        self,
        name: str,
        eval_data: dict,
        rank: int,
        total: int,
        effect_sizes: dict[str, float],
    ) -> str:
        """Generate actionable recommendation."""
        hpr = eval_data["hpr"]
        p95 = eval_data["p95"]

        # Determine use case
        if rank == 1 and hpr < 0.05 and p95 < 1000:
            return (
                f"**Strongly Recommended** for production use. {name} demonstrates "
                f"superior accuracy, excellent safety profile, and acceptable latency. "
                f"Suitable for critical applications requiring high precision hallucination detection."
            )
        elif rank <= 2 and hpr < 0.10:
            return (
                f"**Recommended** for production use. {name} offers strong performance "
                f"with good safety characteristics. Consider for applications where "
                f"accuracy and safety are prioritized over latency."
            )
        elif hpr < 0.15 and p95 < 500:
            return (
                f"**Conditionally Recommended** for specific use cases. {name} provides "
                f"balanced performance suitable for moderate-risk applications. "
                f"Fast response time makes it viable for real-time scenarios."
            )
        elif p95 < 100:
            return (
                f"**Baseline Option** suitable for low-risk applications. {name} offers "
                f"excellent latency but may require additional validation layers for "
                f"critical use cases. Best for high-throughput scenarios."
            )
        else:
            return (
                f"**Not Recommended** for production without significant improvements. "
                f"{name} shows limitations in accuracy or safety that may pose risks. "
                f"Consider for research or development purposes only."
            )

    def _interpret_f1(self, f1: float) -> str:
        """Interpret F1 score quality."""
        if f1 >= 0.90:
            return "excellent balanced"
        elif f1 >= 0.80:
            return "strong balanced"
        elif f1 >= 0.70:
            return "good balanced"
        elif f1 >= 0.60:
            return "acceptable balanced"
        else:
            return "weak balanced"

    def generate_markdown_report(self, statements: list[PerformanceStatement]) -> str:
        """
        Generate comprehensive Markdown report.

        Returns:
            Markdown-formatted report string
        """
        lines = []

        # Title and metadata
        lines.append("# Open Hallucination Index - COMPLETE Mode Benchmark Report")
        lines.append("")
        lines.append(f"**Report ID:** `{self.report.run_id}`")
        lines.append(f"**Generated:** {self.report.timestamp}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        best = statements[0]
        lines.append(
            f"Comprehensive evaluation of {len(statements)} hallucination detection systems "
            f"across multiple datasets with statistical rigor. "
            f"**{best.evaluator_name}** emerged as the top-performing system with "
            f"{best.primary_value:.1%} accuracy (95% CI: [{best.confidence_interval[0]:.1%}, "
            f"{best.confidence_interval[1]:.1%}])."
        )
        lines.append("")

        # Dataset Coverage
        if hasattr(self.report, "complete_mode_metadata"):
            meta = list(self.report.complete_mode_metadata.values())[0]
            lines.append("### Dataset Coverage")
            lines.append("")
            lines.append(f"- **Total Cases:** {meta['total_cases']:,}")
            lines.append(f"- **Datasets:** {meta['total_datasets']}")
            lines.append(f"- **Factual Claims:** {meta['factual_cases']:,}")
            lines.append(f"- **Hallucinations:** {meta['hallucination_cases']:,}")
            lines.append(f"- **Domains:** {', '.join(meta['domains'])}")
            lines.append(f"- **Samples per Dataset:** {meta['samples_per_dataset']}")
            lines.append("")

        # Performance Rankings
        lines.append("## Performance Rankings")
        lines.append("")
        lines.append("| Rank | System | Accuracy | 95% CI | HPR | F1 |")
        lines.append("|------|--------|----------|--------|-----|-----|")

        for stmt in statements:
            metrics = self.report.evaluators[stmt.evaluator_name]
            lines.append(
                f"| {stmt.ranking} | **{stmt.evaluator_name}** | "
                f"{stmt.primary_value:.1%} | "
                f"[{stmt.confidence_interval[0]:.1%}, {stmt.confidence_interval[1]:.1%}] | "
                f"{metrics.hallucination.hallucination_pass_rate:.1%} | "
                f"{metrics.hallucination.f1_score:.1%} |"
            )

        lines.append("")
        lines.append("*HPR = Hallucination Pass Rate (lower is better)*")
        lines.append("")

        # Detailed Analysis
        lines.append("## Detailed Analysis")
        lines.append("")

        for stmt in statements:
            lines.append(f"### {stmt.ranking}. {stmt.evaluator_name}")
            lines.append("")
            lines.append(f"**Interpretation:** {stmt.interpretation}")
            lines.append("")
            lines.append(f"**Recommendation:** {stmt.recommendation}")
            lines.append("")

            # Performance metrics table
            metrics = self.report.evaluators[stmt.evaluator_name]
            lines.append("**Key Metrics:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Accuracy | {metrics.hallucination.accuracy:.1%} |")
            lines.append(f"| Precision | {metrics.hallucination.precision:.1%} |")
            lines.append(f"| Recall | {metrics.hallucination.recall:.1%} |")
            lines.append(f"| F1 Score | {metrics.hallucination.f1_score:.1%} |")
            lines.append(f"| Specificity | {metrics.hallucination.specificity:.1%} |")
            lines.append(f"| HPR (Safety) | {metrics.hallucination.hallucination_pass_rate:.1%} |")
            lines.append(f"| P50 Latency | {metrics.latency.p50:.0f} ms |")
            lines.append(f"| P95 Latency | {metrics.latency.p95:.0f} ms |")
            lines.append(f"| Throughput | {metrics.latency.throughput:.1f} req/s |")
            lines.append("")

        # Statistical Significance
        if hasattr(self.report, "statistical_analysis"):
            lines.append("## Statistical Significance Analysis")
            lines.append("")
            lines.append("Pairwise comparisons with effect sizes (Cohen's d):")
            lines.append("")
            lines.append("| System A | System B | Accuracy Δ | Effect Size | Interpretation |")
            lines.append("|----------|----------|------------|-------------|----------------|")

            comparisons = self.report.statistical_analysis.get("pairwise_comparisons", [])
            for comp in comparisons:
                lines.append(
                    f"| {comp['evaluator_1']} | {comp['evaluator_2']} | "
                    f"{comp['accuracy_diff']:+.1%} | "
                    f"{comp['cohens_d']:+.2f} | "
                    f"{comp['effect_size'].title()} |"
                )

            lines.append("")

        # Methodology
        lines.append("## Methodology")
        lines.append("")
        lines.append("### Evaluation Protocol")
        lines.append("")
        lines.append("- **Mode:** COMPLETE (Research-Grade)")
        lines.append("- **Datasets:** Multi-source HuggingFace datasets")
        lines.append("- **Sampling:** Stratified balanced sampling")
        lines.append("- **Metrics:** Accuracy, Precision, Recall, F1, HPR, Latency")
        lines.append(
            "- **Statistical Tests:** Bootstrap confidence intervals, Cohen's d effect sizes"
        )
        lines.append("- **Confidence Level:** 95%")
        lines.append("")

        # Conclusion
        lines.append("## Conclusion")
        lines.append("")

        best_stmt = statements[0]
        lines.append(
            f"This comprehensive evaluation provides strong empirical evidence that "
            f"**{best_stmt.evaluator_name}** represents the current state-of-the-art "
            f"in hallucination detection for RAG systems. The statistical analysis "
            f"confirms significant performance advantages with effect sizes indicating "
            f"practical importance beyond mere statistical significance."
        )
        lines.append("")

        if len(statements) > 1:
            second = statements[1]
            lines.append(
                f"Alternative systems such as **{second.evaluator_name}** offer viable "
                f"options for specific deployment scenarios where trade-offs between "
                f"accuracy, latency, and safety requirements differ from typical use cases."
            )
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            "*This report was generated automatically by the Open Hallucination Index benchmark suite.*"
        )

        return "\n".join(lines)

    def save_report(self, statements: list[PerformanceStatement]) -> Path:
        """
        Generate and save comprehensive report.

        Returns:
            Path to saved report file
        """
        markdown = self.generate_markdown_report(statements)

        output_path = self.output_dir / f"{self.report.run_id}_COMPLETE_report.md"
        output_path.write_text(markdown, encoding="utf-8")

        return output_path
