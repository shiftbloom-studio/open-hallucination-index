"""
Markdown Reporter
=================

Generate publication-ready Markdown reports for benchmark results.

Improvements:
- Robust against missing fields (getattr everywhere)
- Table of contents + meta block
- Sorted strategy table + highlight best
- Auto-embed generated charts if present in output_dir/charts
- Better stratified analysis layout
- Significance as clean tables + short interpretation
- Error cases using <details> blocks for readability
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


class MarkdownReporter(BaseReporter):
    """
    Markdown report generator.

    Creates publication-ready reports with tables and analysis.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        embed_charts: bool = True,
        max_worst_cases: int = 5,
        snippet_chars: int = 220,
    ) -> None:
        super().__init__(output_dir)
        self.embed_charts = embed_charts
        self.max_worst_cases = max_worst_cases
        self.snippet_chars = snippet_chars

    @property
    def file_extension(self) -> str:
        return "md"

    def generate(self, report: BenchmarkReport, results: list[ResultMetric]) -> str:
        run_id = getattr(report, "run_id", "n/a")
        timestamp = getattr(report, "timestamp", "n/a")
        dataset_path = getattr(report, "dataset_path", "n/a")
        dataset_size = getattr(report, "dataset_size", "n/a")
        threshold = getattr(report, "threshold_used", "n/a")
        runtime = getattr(report, "total_runtime_seconds", None)

        lines: list[str] = []
        lines.append(f"# OHI Benchmark Report — `{run_id}`")
        lines.append("")
        lines.extend(self._toc())

        # Meta block
        lines.append("## Run Metadata")
        lines.append("")
        lines.append("```text")
        lines.append(f"run_id: {run_id}")
        lines.append(f"timestamp: {timestamp}")
        lines.append(f"dataset: {dataset_path} ({dataset_size} cases)")
        lines.append(f"threshold_used: {threshold}")
        if runtime is not None:
            lines.append(f"total_runtime_seconds: {float(runtime):.2f}")
        lines.append(f"results_total: {len(results)}")
        lines.append(f"results_errors: {sum(1 for r in results if getattr(r, 'has_error', False))}")
        lines.append("```")
        lines.append("")

        lines.extend(self._generate_key_insights(report))
        lines.extend(self._generate_summary_table(report))
        lines.extend(self._generate_comparison_table(report))

        if self.embed_charts:
            lines.extend(self._generate_charts_section(report))

        lines.extend(self._generate_stratified_analysis(report))
        lines.extend(self._generate_calibration_section(report))
        lines.extend(self._generate_significance_section(report))
        lines.extend(self._generate_error_analysis(report))

        return "\n".join(lines)

    # ----------------------------
    # TOC / Insights
    # ----------------------------

    def _toc(self) -> list[str]:
        return [
            "## Contents",
            "",
            "- [Run Metadata](#run-metadata)",
            "- [Key Insights](#key-insights)",
            "- [Best Performers](#best-performers)",
            "- [Strategy Comparison](#strategy-comparison)",
            "- [Charts](#charts)",
            "- [Stratified Analysis](#stratified-analysis)",
            "- [Calibration Analysis](#calibration-analysis)",
            "- [Statistical Significance](#statistical-significance)",
            "- [Error Analysis](#error-analysis)",
            "",
        ]

    def _generate_key_insights(self, report: BenchmarkReport) -> list[str]:
        srs = getattr(report, "strategy_reports", None) or {}
        if not srs:
            return ["## Key Insights", "", "_No strategy reports available._", ""]

        # Build sortable list by F1
        rows = []
        for strat, sr in srs.items():
            cm = getattr(sr, "confusion_matrix", None)
            lat = getattr(sr, "latency", None)
            if not cm:
                continue
            f1 = getattr(cm, "f1_score", None)
            acc = getattr(cm, "accuracy", None)
            halluc = getattr(cm, "hallucination_pass_rate", None)
            p95 = getattr(lat, "p95_ms", None) if lat else None
            p50 = getattr(lat, "p50_ms", None) if lat else None
            rows.append((strat, f1, acc, halluc, p50, p95, getattr(sr, "error_count", 0)))

        rows = [r for r in rows if r[1] is not None]
        if not rows:
            return ["## Key Insights", "", "_No valid metrics found._", ""]

        rows.sort(key=lambda r: float(r[1]), reverse=True)
        best = rows[0]
        second = rows[1] if len(rows) > 1 else None

        best_name, best_f1, best_acc, best_hall, best_p50, best_p95, best_err = best
        best_line = (
            f"- 🏆 **Best overall (F1):** `{best_name}` "
            f"(F1={self._fmt(best_f1, '.3f')}, Acc={self._pct(best_acc)}, "
            f"Halluc={self._pct(best_hall)}, P95={self._ms(best_p95)}, Errors={best_err})"
        )

        tradeoff_line = ""
        if second:
            _, f1b, _, _, _, p95b, _ = second
            delta = float(best_f1) - float(f1b)
            tradeoff_line = f"- 📌 **Lead vs #2:** ΔF1={delta:+.3f}"
            if best_p95 is not None and p95b is not None:
                faster = "faster" if float(best_p95) < float(p95b) else "slower"
                tradeoff_line += f", tail latency is **{faster}** (P95 {self._ms(best_p95)} vs {self._ms(p95b)})"

        # Best latency strategy (by P95)
        p95_rows = [(s, p95) for (s, _f1, _a, _h, _p50, p95, _e) in rows if p95 is not None]
        fastest_line = ""
        if p95_rows:
            p95_rows.sort(key=lambda x: float(x[1]))
            fastest_s, fastest_p95 = p95_rows[0]
            fastest_line = f"- ⚡ **Lowest tail latency (P95):** `{fastest_s}` (P95={self._ms(fastest_p95)})"

        return [
            "## Key Insights",
            "",
            best_line,
            tradeoff_line if tradeoff_line else "- 📌 **Lead vs #2:** N/A",
            fastest_line if fastest_line else "- ⚡ **Lowest tail latency (P95):** N/A",
            "",
        ]

    # ----------------------------
    # Best performers
    # ----------------------------

    def _generate_summary_table(self, report: BenchmarkReport) -> list[str]:
        lines = [
            "## Best Performers",
            "",
            "| Metric | Best Strategy | Value |",
            "|--------|---------------|-------|",
        ]

        def get_value(strat: str, metric: str) -> str:
            srs = getattr(report, "strategy_reports", None) or {}
            if not strat or strat not in srs:
                return "N/A"
            sr = srs[strat]
            cm = getattr(sr, "confusion_matrix", None)
            if not cm:
                return "N/A"
            if metric == "accuracy":
                return self._pct(getattr(cm, "accuracy", None))
            if metric == "f1_score":
                return self._fmt(getattr(cm, "f1_score", None), ".3f")
            if metric == "mcc":
                return self._fmt(getattr(cm, "mcc", None), ".3f")
            if metric == "auc":
                roc = getattr(sr, "roc", None)
                return self._fmt(getattr(roc, "auc", None), ".3f")
            if metric == "hallucination_rate":
                return self._pct(getattr(cm, "hallucination_pass_rate", None))
            if metric == "p95":
                lat = getattr(sr, "latency", None)
                return self._ms(getattr(lat, "p95_ms", None))
            return "N/A"

        lines.append(
            f"| Accuracy | `{getattr(report, 'best_strategy_accuracy', None)}` | "
            f"{get_value(getattr(report, 'best_strategy_accuracy', None), 'accuracy')} |"
        )
        lines.append(
            f"| F1 Score | `{getattr(report, 'best_strategy_f1', None)}` | "
            f"{get_value(getattr(report, 'best_strategy_f1', None), 'f1_score')} |"
        )
        lines.append(
            f"| MCC | `{getattr(report, 'best_strategy_mcc', None)}` | "
            f"{get_value(getattr(report, 'best_strategy_mcc', None), 'mcc')} |"
        )
        lines.append(
            f"| AUC-ROC | `{getattr(report, 'best_strategy_auc', None)}` | "
            f"{get_value(getattr(report, 'best_strategy_auc', None), 'auc')} |"
        )
        lines.append(
            f"| Lowest Hallucination Rate | `{getattr(report, 'lowest_hallucination_rate', None)}` | "
            f"{get_value(getattr(report, 'lowest_hallucination_rate', None), 'hallucination_rate')} |"
        )
        lines.append(
            f"| Lowest P95 Latency | `{self._best_latency_strategy(report)}` | "
            f"{get_value(self._best_latency_strategy(report), 'p95')} |"
        )
        lines.append("")
        return lines

    def _best_latency_strategy(self, report: BenchmarkReport) -> str | None:
        srs = getattr(report, "strategy_reports", None) or {}
        best_s = None
        best_p = None
        for s, sr in srs.items():
            lat = getattr(sr, "latency", None)
            p95 = getattr(lat, "p95_ms", None) if lat else None
            if p95 is None:
                continue
            if best_p is None or float(p95) < float(best_p):
                best_p = p95
                best_s = s
        return best_s

    # ----------------------------
    # Strategy comparison
    # ----------------------------

    def _generate_comparison_table(self, report: BenchmarkReport) -> list[str]:
        srs = getattr(report, "strategy_reports", None) or {}
        lines = [
            "## Strategy Comparison",
            "",
            "| Rank | Strategy | Acc | Prec | Rec | F1 | MCC | AUC | Halluc. | P50 | P95 | Errors |",
            "|------|----------|-----|------|-----|----|-----|-----|---------|-----|-----|--------|",
        ]

        rows = []
        for strat, sr in srs.items():
            cm = getattr(sr, "confusion_matrix", None)
            if not cm:
                continue
            f1 = getattr(cm, "f1_score", None)
            rows.append((strat, float(f1) if f1 is not None else -1.0))
        rows.sort(key=lambda x: x[1], reverse=True)

        best = getattr(report, "best_strategy_f1", None)
        for i, (strat, _f1) in enumerate(rows, 1):
            sr = srs[strat]
            cm = sr.confusion_matrix
            roc = getattr(sr, "roc", None)
            lat = getattr(sr, "latency", None)

            auc = getattr(roc, "auc", None)
            p50 = getattr(lat, "p50_ms", None) if lat else None
            p95 = getattr(lat, "p95_ms", None) if lat else None

            badge = "🏆" if strat == best else ""
            lines.append(
                f"| {i} | {badge} `{strat}` | {self._pct(getattr(cm,'accuracy',None))} | {self._pct(getattr(cm,'precision',None))} | "
                f"{self._pct(getattr(cm,'recall',None))} | {self._fmt(getattr(cm,'f1_score',None),'.3f')} | {self._fmt(getattr(cm,'mcc',None),'.3f')} | "
                f"{self._fmt(auc,'.3f')} | {self._pct(getattr(cm,'hallucination_pass_rate',None))} | {self._ms(p50)} | {self._ms(p95)} | {getattr(sr,'error_count',0)} |"
            )

        lines.append("")
        return lines

    # ----------------------------
    # Charts embedding
    # ----------------------------

    def _generate_charts_section(self, report: BenchmarkReport) -> list[str]:
        charts_dir = self.output_dir / "charts"
        if not charts_dir.exists():
            return ["## Charts", "", "_No charts directory found._", ""]

        # Prefer run-prefixed images if present, otherwise include all pngs
        run_id = getattr(report, "run_id", "run")
        preferred_prefix = f"ohi_benchmark_{run_id}_"
        pngs = sorted([p for p in charts_dir.glob("*.png")])
        if not pngs:
            return ["## Charts", "", "_No chart PNGs found._", ""]

        preferred = [p for p in pngs if p.name.startswith(preferred_prefix)]
        selected = preferred if preferred else pngs

        lines = ["## Charts", ""]
        lines.append("> Tip: These images are embedded if your Markdown renderer supports local relative paths.")
        lines.append("")

        for p in selected:
            title = p.stem.replace("_", " ").title()
            rel = Path("charts") / p.name
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({rel.as_posix()})")
            lines.append("")

        return lines

    # ----------------------------
    # Stratified analysis
    # ----------------------------

    def _generate_stratified_analysis(self, report: BenchmarkReport) -> list[str]:
        srs = getattr(report, "strategy_reports", None) or {}
        lines = ["## Stratified Analysis", ""]

        if not srs:
            return lines + ["_No strategy reports available._", ""]

        for strat, sr in srs.items():
            lines.append(f"### `{strat}`")
            lines.append("")

            # By Domain
            by_domain = getattr(sr, "by_domain", None)
            if by_domain:
                lines.append("#### By Domain")
                lines.append("")
                lines.append("| Domain | Acc | F1 | Halluc. | n |")
                lines.append("|--------|-----|----|---------|---|")

                # sort by n desc
                domains = sorted(by_domain.keys(), key=lambda d: getattr(by_domain[d], "total", 0), reverse=True)
                for domain in domains:
                    cm = by_domain[domain]
                    lines.append(
                        f"| `{domain}` | {self._pct(getattr(cm,'accuracy',None))} | {self._fmt(getattr(cm,'f1_score',None),'.3f')} | "
                        f"{self._pct(getattr(cm,'hallucination_pass_rate',None))} | {getattr(cm,'total',0)} |"
                    )
                lines.append("")

            # By Difficulty
            by_diff = getattr(sr, "by_difficulty", None)
            if by_diff:
                lines.append("#### By Difficulty")
                lines.append("")
                lines.append("| Difficulty | Acc | F1 | Halluc. | n |")
                lines.append("|------------|-----|----|---------|---|")
                for diff in ["easy", "medium", "hard", "critical"]:
                    if diff in by_diff:
                        cm = by_diff[diff]
                        lines.append(
                            f"| `{diff}` | {self._pct(getattr(cm,'accuracy',None))} | {self._fmt(getattr(cm,'f1_score',None),'.3f')} | "
                            f"{self._pct(getattr(cm,'hallucination_pass_rate',None))} | {getattr(cm,'total',0)} |"
                        )
                lines.append("")

            # Multi-claim vs Single-claim
            multi = getattr(sr, "multi_claim_cm", None)
            single = getattr(sr, "single_claim_cm", None)
            if multi and single:
                lines.append("#### Multi-claim vs Single-claim")
                lines.append("")
                lines.append("| Type | Acc | F1 | Halluc. | n |")
                lines.append("|------|-----|----|---------|---|")
                lines.append(
                    f"| `single` | {self._pct(getattr(single,'accuracy',None))} | {self._fmt(getattr(single,'f1_score',None),'.3f')} | "
                    f"{self._pct(getattr(single,'hallucination_pass_rate',None))} | {getattr(single,'total',0)} |"
                )
                lines.append(
                    f"| `multi` | {self._pct(getattr(multi,'accuracy',None))} | {self._fmt(getattr(multi,'f1_score',None),'.3f')} | "
                    f"{self._pct(getattr(multi,'hallucination_pass_rate',None))} | {getattr(multi,'total',0)} |"
                )
                lines.append("")

        return lines

    # ----------------------------
    # Calibration
    # ----------------------------

    def _generate_calibration_section(self, report: BenchmarkReport) -> list[str]:
        srs = getattr(report, "strategy_reports", None) or {}
        lines = [
            "## Calibration Analysis",
            "",
            "| Strategy | Brier | ECE | MCE | Optimal Threshold | Youden's J |",
            "|----------|-------|-----|-----|-------------------|------------|",
        ]

        for strat, sr in srs.items():
            cal = getattr(sr, "calibration", None)
            roc = getattr(sr, "roc", None)
            lines.append(
                f"| `{strat}` | {self._fmt(getattr(cal,'brier_score',None),'.4f')} | {self._fmt(getattr(cal,'ece',None),'.4f')} | "
                f"{self._fmt(getattr(cal,'mce',None),'.4f')} | {self._fmt(getattr(roc,'optimal_threshold',None),'.3f')} | {self._fmt(getattr(roc,'youden_j',None),'.3f')} |"
            )

        lines.append("")
        lines.extend(
            [
                "> **Interpretation:**",
                "> - **Brier Score**: Lower is better (0 = perfect; ~0.25 can indicate uninformative probabilities for balanced binary tasks)",
                "> - **ECE/MCE**: Calibration error (lower is better)",
                "> - **Optimal Threshold**: Threshold maximizing Youden's J",
                "> - **Youden's J**: Higher is better (max = 1)",
                "",
            ]
        )
        return lines

    # ----------------------------
    # Significance
    # ----------------------------

    def _generate_significance_section(self, report: BenchmarkReport) -> list[str]:
        comparisons = getattr(report, "comparisons", None)
        legacy = getattr(report, "mcnemar_results", None)

        if not comparisons and not legacy:
            return []

        lines = ["## Statistical Significance", ""]

        if comparisons:
            lines.extend(
                [
                    "| Comparison | McNemar χ² | McNemar p | DeLong Z | DeLong p | AUC Diff | Significant |",
                    "|------------|------------:|----------:|---------:|----------:|---------:|:-----------:|",
                ]
            )
            for _, comp in comparisons.items():
                sig = "✓" if getattr(comp, "is_significant", False) else "✗"
                lines.append(
                    f"| `{getattr(comp,'strategy_a',None)}` vs `{getattr(comp,'strategy_b',None)}` | "
                    f"{self._fmt(getattr(comp,'mcnemar_chi2',None),'.2f')} | {self._fmt(getattr(comp,'mcnemar_p_value',None),'.4f')} | "
                    f"{self._fmt(getattr(comp,'delong_z',None),'.2f')} | {self._fmt(getattr(comp,'delong_p_value',None),'.4f')} | "
                    f"{self._fmt(getattr(comp,'auc_difference',None),'+.3f')} | {sig} |"
                )
        else:
            lines.extend(
                [
                    "| Comparison | χ² | p-value | Significant (α=0.05) |",
                    "|------------|---:|--------:|:--------------------:|",
                ]
            )
            for comp, res in (legacy or {}).items():
                sig = "✓" if res.get("significant_at_05") else "✗"
                lines.append(
                    f"| `{comp}` | {res.get('chi2', 0):.2f} | {res.get('p_value', 1.0):.4f} | {sig} |"
                )

        lines.append("")
        lines.extend(
            [
                "> Notes:",
                "> - **McNemar** tests paired classification differences (same cases).",
                "> - **DeLong** compares ROC-AUC differences (probabilistic outputs required).",
                "",
            ]
        )
        return lines

    # ----------------------------
    # Errors / worst cases
    # ----------------------------

    def _generate_error_analysis(self, report: BenchmarkReport) -> list[str]:
        srs = getattr(report, "strategy_reports", None) or {}
        lines = ["## Error Analysis", ""]

        any_cases = False
        for strat, sr in srs.items():
            worst_fp = getattr(sr, "worst_fp_cases", None) or []
            worst_fn = getattr(sr, "worst_fn_cases", None) or []
            if not worst_fp and not worst_fn:
                continue

            any_cases = True
            lines.append(f"### `{strat}`")
            lines.append("")

            if worst_fp:
                lines.append("#### Worst False Positives (Hallucinations Believed)")
                lines.append("")
                lines.extend(self._details_cases(worst_fp[: self.max_worst_cases]))

            if worst_fn:
                lines.append("#### Worst False Negatives (Facts Rejected)")
                lines.append("")
                lines.extend(self._details_cases(worst_fn[: self.max_worst_cases]))

        if not any_cases:
            lines.append("_No worst-case lists available._")
            lines.append("")

        return lines

    def _details_cases(self, cases: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for i, c in enumerate(cases, 1):
            case_id = c.get("case_id", "n/a")
            score = c.get("trust_score", None)
            domain = c.get("domain", None)
            text = (c.get("text", "") or "").strip().replace("\r", "")
            snippet = text.replace("\n", " ")
            if len(snippet) > self.snippet_chars:
                snippet = snippet[: self.snippet_chars] + "…"

            title_bits = [f"#{i}", f"case {case_id}"]
            if domain:
                title_bits.append(f"domain {domain}")
            if score is not None:
                try:
                    title_bits.append(f"trust_score {float(score):.3f}")
                except Exception:
                    title_bits.append(f"trust_score {score}")

            lines.append("<details>")
            lines.append(f"<summary><strong>{' • '.join(title_bits)}</strong></summary>")
            lines.append("")
            lines.append("```text")
            lines.append(snippet)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        return lines

    # ----------------------------
    # Formatting helpers
    # ----------------------------

    def _fmt(self, v: Any, fmt: str) -> str:
        if v is None:
            return "N/A"
        try:
            return format(float(v), fmt)
        except Exception:
            return "N/A"

    def _pct(self, v: Any) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.1%}"
        except Exception:
            return "N/A"

    def _ms(self, v: Any) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.0f}ms"
        except Exception:
            return "N/A"
