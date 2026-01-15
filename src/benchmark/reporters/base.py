"""
Base Reporter Interface
=======================

Abstract base class for all benchmark report generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


class BaseReporter(ABC):
    """
    Abstract base class for benchmark reporters.

    All reporters must implement the generate and save methods.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize reporter.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this report format."""
        ...

    @abstractmethod
    def generate(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
    ) -> str:
        """
        Generate report content.

        Args:
            report: The benchmark report.
            results: Raw result metrics.

        Returns:
            Report content as string.
        """
        ...

    def save(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
        filename: str | None = None,
    ) -> Path:
        """
        Generate and save report to file.

        Args:
            report: The benchmark report.
            results: Raw result metrics.
            filename: Optional custom filename (without extension).

        Returns:
            Path to saved file.
        """
        if filename is None:
            filename = f"ohi_benchmark_{report.run_id}"

        content = self.generate(report, results)
        filepath = self.output_dir / f"{filename}.{self.file_extension}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath
