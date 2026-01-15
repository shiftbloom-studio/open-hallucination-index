"""
JSON Reporter
=============

Generate machine-readable JSON reports for benchmark results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


class JSONReporter(BaseReporter):
    """
    JSON report generator.

    Creates machine-readable reports for integration with other tools.
    """

    @property
    def file_extension(self) -> str:
        return "json"

    def generate(
        self,
        report: BenchmarkReport,
        results: list[ResultMetric],
    ) -> str:
        """Generate JSON report content."""
        return json.dumps(
            asdict(report),
            indent=2,
            default=self._serialize,
            ensure_ascii=False,
        )

    def _serialize(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)
