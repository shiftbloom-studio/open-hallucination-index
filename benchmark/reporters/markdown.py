"""
Markdown Reporter
=================

Generate publication-ready Markdown reports for benchmark results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


class MarkdownReporter(BaseReporter):
    """
    Markdown report generator.

    Creates publication-ready reports with tables and analysis.
    """

    @property
    def file_extension(self) -> str:
        return "md"

    def generate(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
    ) -> str:
        """Generate Markdown report content."""
        lines = [
            "# OHI Benchmark Report",
            "",
            "## Executive Summary",
            "",
            f"**Run ID:** `{report.run_id}`  ",
            f"**Timestamp:** {report.timestamp}  ",
            f"**Dataset:** {report.dataset_path} ({report.dataset_size} cases)  ",
            f"**Threshold:** {report.threshold_used}  ",
            f"**Runtime:** {report.total_runtime_seconds:.1f}s  ",
            "",
        ]

        # Best performers summary
        lines.extend(self._generate_summary_table(report))

        # Strategy comparison
        lines.extend(self._generate_comparison_table(report))

        # Stratified analysis
        lines.extend(self._generate_stratified_analysis(report))

        # Calibration analysis
        lines.extend(self._generate_calibration_section(report))

        # Statistical significance
        lines.extend(self._generate_significance_section(report))

        # Error analysis
        lines.extend(self._generate_error_analysis(report))

        return "\n".join(lines)

    def _generate_summary_table(self, report: BenchmarkReport) -> list[str]:
        """Generate summary of best performers."""
        lines = [
            "## Best Performers",
            "",
            "| Metric | Best Strategy | Value |",
            "|--------|---------------|-------|",
        ]

        def get_value(strat: str, metric: str) -> str:
            if not strat or strat not in report.strategy_reports:
                return "N/A"
            sr = report.strategy_reports[strat]
            if metric == "accuracy":
                return f"{sr.confusion_matrix.accuracy:.1%}"
            if metric == "f1_score":
                return f"{sr.confusion_matrix.f1_score:.3f}"
            if metric == "mcc":
                return f"{sr.confusion_matrix.mcc:.3f}"
            if metric == "auc":
                return f"{sr.roc.auc:.3f}" if sr.roc else "N/A"
            if metric == "hallucination_rate":
                return f"{sr.confusion_matrix.hallucination_pass_rate:.1%}"
            return "N/A"

        lines.append(
            f"| Accuracy | {report.best_strategy_accuracy} | "
            f"{get_value(report.best_strategy_accuracy, 'accuracy')} |"
        )
        lines.append(
            f"| F1 Score | {report.best_strategy_f1} | "
            f"{get_value(report.best_strategy_f1, 'f1_score')} |"
        )
        lines.append(
            f"| MCC | {report.best_strategy_mcc} | "
            f"{get_value(report.best_strategy_mcc, 'mcc')} |"
        )
        lines.append(
            f"| AUC-ROC | {report.best_strategy_auc} | "
            f"{get_value(report.best_strategy_auc, 'auc')} |"
        )
        lines.append(
            f"| Lowest Hallucination Rate | {report.lowest_hallucination_rate} | "
            f"{get_value(report.lowest_hallucination_rate, 'hallucination_rate')} |"
        )
        lines.append("")

        return lines

    def _generate_comparison_table(self, report: BenchmarkReport) -> list[str]:
        """Generate full strategy comparison table."""
        lines = [
            "## Strategy Comparison",
            "",
            "| Strategy | Accuracy | Precision | Recall | F1 | MCC | AUC | Halluc. Rate | P95 Latency |",
            "|----------|----------|-----------|--------|-----|-----|-----|--------------|-------------|",
        ]

        for strat, sr in report.strategy_reports.items():
            cm = sr.confusion_matrix
            auc = sr.roc.auc if sr.roc else 0.0
            p95 = sr.latency.p95_ms if sr.latency else 0.0

            lines.append(
                f"| {strat} | {cm.accuracy:.1%} | {cm.precision:.1%} | "
                f"{cm.recall:.1%} | {cm.f1_score:.3f} | {cm.mcc:.3f} | "
                f"{auc:.3f} | {cm.hallucination_pass_rate:.1%} | {p95:.0f}ms |"
            )

        lines.append("")
        return lines

    def _generate_stratified_analysis(self, report: BenchmarkReport) -> list[str]:
        """Generate stratified analysis by domain and difficulty."""
        lines = ["## Stratified Analysis", ""]

        for strat, sr in report.strategy_reports.items():
            lines.append(f"### {strat}")
            lines.append("")

            # By Domain
            if sr.by_domain:
                lines.append("#### By Domain")
                lines.append("")
                lines.append("| Domain | Accuracy | F1 | Halluc. Rate | n |")
                lines.append("|--------|----------|-----|--------------|---|")

                for domain in sorted(sr.by_domain.keys()):
                    cm = sr.by_domain[domain]
                    lines.append(
                        f"| {domain} | {cm.accuracy:.1%} | {cm.f1_score:.3f} | "
                        f"{cm.hallucination_pass_rate:.1%} | {cm.total} |"
                    )
                lines.append("")

            # By Difficulty
            if sr.by_difficulty:
                lines.append("#### By Difficulty")
                lines.append("")
                lines.append("| Difficulty | Accuracy | F1 | Halluc. Rate | n |")
                lines.append("|------------|----------|-----|--------------|---|")

                for diff in ["easy", "medium", "hard", "critical"]:
                    if diff in sr.by_difficulty:
                        cm = sr.by_difficulty[diff]
                        lines.append(
                            f"| {diff} | {cm.accuracy:.1%} | {cm.f1_score:.3f} | "
                            f"{cm.hallucination_pass_rate:.1%} | {cm.total} |"
                        )
                lines.append("")

            # Multi-claim vs Single-claim
            if sr.multi_claim_cm and sr.single_claim_cm:
                lines.append("#### Multi-claim vs Single-claim")
                lines.append("")
                lines.append("| Type | Accuracy | F1 | Halluc. Rate | n |")
                lines.append("|------|----------|-----|--------------|---|")
                lines.append(
                    f"| Single | {sr.single_claim_cm.accuracy:.1%} | "
                    f"{sr.single_claim_cm.f1_score:.3f} | "
                    f"{sr.single_claim_cm.hallucination_pass_rate:.1%} | "
                    f"{sr.single_claim_cm.total} |"
                )
                lines.append(
                    f"| Multi | {sr.multi_claim_cm.accuracy:.1%} | "
                    f"{sr.multi_claim_cm.f1_score:.3f} | "
                    f"{sr.multi_claim_cm.hallucination_pass_rate:.1%} | "
                    f"{sr.multi_claim_cm.total} |"
                )
                lines.append("")

        return lines

    def _generate_calibration_section(self, report: BenchmarkReport) -> list[str]:
        """Generate calibration analysis section."""
        lines = [
            "## Calibration Analysis",
            "",
            "| Strategy | Brier Score | ECE | MCE | Optimal Threshold | Youden's J |",
            "|----------|-------------|-----|-----|-------------------|------------|",
        ]

        for strat, sr in report.strategy_reports.items():
            brier = sr.calibration.brier_score if sr.calibration else 0.0
            ece = sr.calibration.ece if sr.calibration else 0.0
            mce = sr.calibration.mce if sr.calibration else 0.0
            opt_thresh = sr.roc.optimal_threshold if sr.roc else 0.5
            youden = sr.roc.youden_j if sr.roc else 0.0

            lines.append(
                f"| {strat} | {brier:.4f} | {ece:.4f} | {mce:.4f} | "
                f"{opt_thresh:.3f} | {youden:.3f} |"
            )

        lines.append("")

        # Add interpretation
        lines.extend([
            "> **Interpretation:**",
            "> - **Brier Score**: Lower is better (0 = perfect, 0.25 = random)",
            "> - **ECE (Expected Calibration Error)**: Lower is better",
            "> - **Optimal Threshold**: Threshold that maximizes Youden's J",
            "> - **Youden's J**: Higher is better (max = 1)",
            "",
        ])

        return lines

    def _generate_significance_section(self, report: BenchmarkReport) -> list[str]:
        """Generate statistical significance section."""
        if not report.comparisons and not report.mcnemar_results:
            return []

        lines = [
            "## Statistical Significance",
            "",
        ]

        # New format with full comparison objects
        if report.comparisons:
            lines.extend([
                "| Comparison | McNemar χ² | McNemar p | DeLong Z | DeLong p | AUC Diff | Significant |",
                "|------------|------------|-----------|----------|----------|----------|-------------|",
            ])

            for _, comp in report.comparisons.items():
                sig = "✓" if comp.is_significant else "✗"
                lines.append(
                    f"| {comp.strategy_a} vs {comp.strategy_b} | "
                    f"{comp.mcnemar_chi2:.2f} | {comp.mcnemar_p_value:.4f} | "
                    f"{comp.delong_z:.2f} | {comp.delong_p_value:.4f} | "
                    f"{comp.auc_difference:+.3f} | {sig} |"
                )
        # Legacy format
        elif report.mcnemar_results:
            lines.extend([
                "| Comparison | χ² | p-value | Significant (α=0.05) |",
                "|------------|-----|---------|---------------------|",
            ])

            for comp, res in report.mcnemar_results.items():
                sig = "✓" if res.get("significant_at_05") else "✗"
                lines.append(
                    f"| {comp} | {res.get('chi2', 0):.2f} | "
                    f"{res.get('p_value', 1.0):.4f} | {sig} |"
                )

        lines.append("")
        return lines

    def _generate_error_analysis(self, report: BenchmarkReport) -> list[str]:
        """Generate error analysis section with worst cases."""
        lines = ["## Error Analysis", ""]

        for strat, sr in report.strategy_reports.items():
            if not sr.worst_fp_cases and not sr.worst_fn_cases:
                continue

            lines.append(f"### {strat}")
            lines.append("")

            if sr.worst_fp_cases:
                lines.append("#### Worst False Positives (Hallucinations Believed)")
                lines.append("")
                for i, fp in enumerate(sr.worst_fp_cases[:5], 1):
                    lines.append(
                        f"{i}. **Case {fp['case_id']}** (score: {fp['trust_score']:.2f})"
                    )
                    text = fp.get("text", "")[:100]
                    lines.append(f'   > "{text}..."')
                    lines.append("")

            if sr.worst_fn_cases:
                lines.append("#### Worst False Negatives (Facts Rejected)")
                lines.append("")
                for i, fn in enumerate(sr.worst_fn_cases[:5], 1):
                    lines.append(
                        f"{i}. **Case {fn['case_id']}** (score: {fn['trust_score']:.2f})"
                    )
                    text = fn.get("text", "")[:100]
                    lines.append(f'   > "{text}..."')
                    lines.append("")

        return lines
