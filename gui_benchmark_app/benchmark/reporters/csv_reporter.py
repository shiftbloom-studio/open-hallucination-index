"""
CSV Reporter
============

Generate CSV exports of benchmark results.

Improvements:
- Robust record extraction (dataclass / dict / object)
- Flatten nested fields via pandas.json_normalize (sep="__")
- Optional run-metadata columns for easy joins
- Stable column ordering + JSON serialization for complex object columns
- Writes extra CSVs (strategy summary, domain summary, comparisons, run metadata)
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import json
import pandas as pd

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


_DROP_COLUMNS_DEFAULT = {
    "raw_response",  # too large for CSV
}


class CSVReporter(BaseReporter):
    """
    CSV report generator.

    Exports raw results for analysis in spreadsheets or pandas.
    Also writes additional summary CSVs for quicker analysis.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        include_run_columns: bool = True,
        drop_columns: set[str] | None = None,
        flatten_sep: str = "__",
    ) -> None:
        super().__init__(output_dir)
        self.include_run_columns = include_run_columns
        self.drop_columns = drop_columns or set(_DROP_COLUMNS_DEFAULT)
        self.flatten_sep = flatten_sep

    @property
    def file_extension(self) -> str:
        return "csv"

    # ----------------------------
    # Public API
    # ----------------------------

    def generate(self, report: BenchmarkReport, results: list[ResultMetric]) -> str:
        """Generate CSV content from raw results (primary results export)."""
        if not results:
            return ""

        df = self._results_df(report, results)
        return df.to_csv(index=False, encoding="utf-8")

    def save(
        self, report: BenchmarkReport, results: list[ResultMetric], filename: str | None = None
    ) -> Path:
        """
        Save results to CSV files.

        Writes:
        - <filename>_results.csv (primary)
        - <filename>_strategy_summary.csv
        - <filename>_domain_summary.csv
        - <filename>_comparisons.csv
        - <filename>_run_metadata.csv
        """
        run_id = getattr(report, "run_id", "run")
        base = filename or f"ohi_benchmark_{run_id}"

        # Primary results CSV
        results_path = self.output_dir / f"{base}_results.{self.file_extension}"
        df_results = self._results_df(report, results)
        df_results.to_csv(results_path, index=False, encoding="utf-8", newline="")

        # Extra exports (best effort; never fail overall save)
        try:
            df_run = self._run_metadata_df(report, results)
            df_run.to_csv(
                self.output_dir / f"{base}_run_metadata.{self.file_extension}",
                index=False,
                encoding="utf-8",
                newline="",
            )
        except Exception:
            pass

        try:
            df_strat = self._strategy_summary_df(report)
            if not df_strat.empty:
                df_strat.to_csv(
                    self.output_dir / f"{base}_strategy_summary.{self.file_extension}",
                    index=False,
                    encoding="utf-8",
                    newline="",
                )
        except Exception:
            pass

        try:
            df_dom = self._domain_summary_df(report)
            if not df_dom.empty:
                df_dom.to_csv(
                    self.output_dir / f"{base}_domain_summary.{self.file_extension}",
                    index=False,
                    encoding="utf-8",
                    newline="",
                )
        except Exception:
            pass

        try:
            df_comp = self._comparisons_df(report)
            if not df_comp.empty:
                df_comp.to_csv(
                    self.output_dir / f"{base}_comparisons.{self.file_extension}",
                    index=False,
                    encoding="utf-8",
                    newline="",
                )
        except Exception:
            pass

        return results_path

    # ----------------------------
    # DataFrames
    # ----------------------------

    def _results_df(self, report: BenchmarkReport, results: list[ResultMetric]) -> pd.DataFrame:
        records = [self._to_record(r) for r in results]

        # Flatten nested dicts into columns
        df = pd.json_normalize(records, sep=self.flatten_sep)

        # Drop large / unwanted columns if present
        drop = [c for c in self.drop_columns if c in df.columns]
        if drop:
            df = df.drop(columns=drop)

        # Optional run columns (helpful for joins across multiple runs)
        if self.include_run_columns:
            run_cols = self._run_columns(report)
            for k, v in run_cols.items():
                if k not in df.columns:
                    df[k] = v

        # Normalize object columns: dict/list -> JSON strings
        df = self._stringify_complex_cells(df)

        # Order columns: key metrics first, then rest alphabetical
        df = self._reorder_columns(df)

        return df

    def _run_metadata_df(
        self, report: BenchmarkReport, results: list[ResultMetric]
    ) -> pd.DataFrame:
        total = len(results)
        errors = sum(1 for r in results if getattr(r, "has_error", False))
        valid = total - errors

        row = {
            "run_id": getattr(report, "run_id", None),
            "dataset_size": getattr(report, "dataset_size", None),
            "threshold_used": getattr(report, "threshold_used", None),
            "total_runtime_seconds": getattr(report, "total_runtime_seconds", None),
            "best_strategy_f1": getattr(report, "best_strategy_f1", None),
            "results_total": total,
            "results_valid": valid,
            "results_errors": errors,
        }
        return pd.DataFrame([row])

    def _strategy_summary_df(self, report: BenchmarkReport) -> pd.DataFrame:
        srs = getattr(report, "strategy_reports", None) or {}
        rows: list[dict[str, Any]] = []

        for strategy, sr in srs.items():
            cm = getattr(sr, "confusion_matrix", None)
            lat = getattr(sr, "latency", None)
            roc = getattr(sr, "roc", None)
            cal = getattr(sr, "calibration", None)

            row = {
                "strategy": strategy,
                # Confusion matrix metrics
                "total": getattr(cm, "total", None),
                "accuracy": getattr(cm, "accuracy", None),
                "precision": getattr(cm, "precision", None),
                "recall": getattr(cm, "recall", None),
                "f1": getattr(cm, "f1_score", None),
                "mcc": getattr(cm, "mcc", None),
                "hallucination_pass_rate": getattr(cm, "hallucination_pass_rate", None),
                # Latency
                "latency_avg_ms": getattr(lat, "avg_ms", None),
                "latency_p50_ms": getattr(lat, "p50_ms", None),
                "latency_p95_ms": getattr(lat, "p95_ms", None),
                "latency_p99_ms": getattr(lat, "p99_ms", None),
                # Errors
                "error_count": getattr(sr, "error_count", None),
                # ROC / Calibration
                "roc_auc": getattr(roc, "auc", None),
                "roc_optimal_threshold": getattr(roc, "optimal_threshold", None),
                "roc_youden_j": getattr(roc, "youden_j", None),
                "cal_brier_score": getattr(cal, "brier_score", None),
                "cal_ece": getattr(cal, "ece", None),
            }

            # Confidence interval if present
            f1_ci = getattr(sr, "f1_ci", None)
            if f1_ci is not None:
                row["f1_ci_low"] = getattr(f1_ci, "low", None)
                row["f1_ci_high"] = getattr(f1_ci, "high", None)
                row["f1_ci_margin"] = getattr(f1_ci, "margin_of_error", None)

            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort by F1 desc if present
        if "f1" in df.columns:
            df = df.sort_values(by="f1", ascending=False, na_position="last")

        return df

    def _domain_summary_df(self, report: BenchmarkReport) -> pd.DataFrame:
        srs = getattr(report, "strategy_reports", None) or {}
        rows: list[dict[str, Any]] = []

        # Export all strategies by domain if available (more useful than only best)
        for strategy, sr in srs.items():
            by_domain = getattr(sr, "by_domain", None) or {}
            for domain, cm in by_domain.items():
                rows.append(
                    {
                        "strategy": strategy,
                        "domain": domain,
                        "total": getattr(cm, "total", None),
                        "accuracy": getattr(cm, "accuracy", None),
                        "precision": getattr(cm, "precision", None),
                        "recall": getattr(cm, "recall", None),
                        "f1": getattr(cm, "f1_score", None),
                        "mcc": getattr(cm, "mcc", None),
                        "hallucination_pass_rate": getattr(cm, "hallucination_pass_rate", None),
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Nice ordering: strategy then largest domains first
        if "total" in df.columns:
            df = df.sort_values(
                by=["strategy", "total"], ascending=[True, False], na_position="last"
            )

        return df

    def _comparisons_df(self, report: BenchmarkReport) -> pd.DataFrame:
        comparisons = getattr(report, "comparisons", None)
        legacy = getattr(report, "mcnemar_results", None)

        rows: list[dict[str, Any]] = []

        if comparisons:
            for _name, comp in comparisons.items():
                rows.append(
                    {
                        "strategy_a": getattr(comp, "strategy_a", None),
                        "strategy_b": getattr(comp, "strategy_b", None),
                        "mcnemar_p_value": getattr(comp, "mcnemar_p_value", None),
                        "delong_p_value": getattr(comp, "delong_p_value", None),
                        "is_significant": getattr(comp, "is_significant", None),
                    }
                )
        elif legacy:
            for comp_name, res in legacy.items():
                rows.append(
                    {
                        "comparison": str(comp_name),
                        "mcnemar_p_value": res.get("p_value", None),
                        "significant_at_05": res.get("significant_at_05", None),
                    }
                )

        return pd.DataFrame(rows)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _to_record(self, obj: Any) -> dict[str, Any]:
        """Convert ResultMetric-like object to a dict without mutating it."""
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return dict(obj)
        # Pydantic / attrs / plain objects
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return {"value": str(obj)}

    def _run_columns(self, report: BenchmarkReport) -> dict[str, Any]:
        return {
            "run_id": getattr(report, "run_id", None),
            "dataset_size": getattr(report, "dataset_size", None),
            "threshold_used": getattr(report, "threshold_used", None),
        }

    def _stringify_complex_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert dict/list cells to compact JSON strings for CSV safety."""
        if df.empty:
            return df

        def conv(v: Any) -> Any:
            if isinstance(v, (dict, list)):
                try:
                    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    return str(v)
            return v

        # Only apply to object dtype columns
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        for c in obj_cols:
            df[c] = df[c].map(conv)

        return df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Put commonly-used columns first; rest alphabetical."""
        if df.empty:
            return df

        preferred = [
            # identifiers / grouping
            "run_id",
            "case_id",
            "id",
            "strategy",
            "domain",
            # labels / predictions
            "label",
            "ground_truth",
            "expected",
            "prediction",
            "predicted",
            "is_correct",
            "correct",
            "trust_score",
            "score",
            # timing / errors
            "latency_ms",
            "has_error",
            "error_type",
            "error_message",
            # dataset meta
            "dataset_size",
            "threshold_used",
        ]

        cols = list(df.columns)
        first = [c for c in preferred if c in cols]
        rest = sorted([c for c in cols if c not in first])
        return df[first + rest]
