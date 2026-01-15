"""
CSV Reporter
============

Generate CSV exports of raw benchmark results.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


class CSVReporter(BaseReporter):
    """
    CSV report generator.

    Exports raw results for analysis in spreadsheets or pandas.
    """

    @property
    def file_extension(self) -> str:
        return "csv"

    def generate(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
    ) -> str:
        """Generate CSV content from raw results."""
        if not results:
            return ""

        df = pd.DataFrame([asdict(r) for r in results])

        # Remove raw_response column if present (too large for CSV)
        if "raw_response" in df.columns:
            df = df.drop(columns=["raw_response"])

        return df.to_csv(index=False)

    def save(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
        filename: str | None = None,
    ) -> Path:
        """Save results to CSV file."""
        if filename is None:
            filename = f"ohi_benchmark_{report.run_id}"

        content = self.generate(report, results)
        filepath = self.output_dir / f"{filename}.{self.file_extension}"

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(content)

        return filepath
