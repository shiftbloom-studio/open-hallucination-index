"""
JSON Reporter
=============

Generate machine-readable JSON reports for benchmark results.

Improvements:
- Strict JSON: allow_nan=False (no NaN/Infinity)
- Robust serialization: numpy, dataclasses, datetime, Path, sets, bytes, enums
- Optional deterministic output: sort_keys=True
- Optional include raw results in JSON (with field drops like raw_response)
- Adds top-level envelope with schema_version + meta for easier integration
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from benchmark.reporters.base import BaseReporter

if TYPE_CHECKING:
    from benchmark.models import BenchmarkReport, ResultMetric


class JSONReporter(BaseReporter):
    """
    JSON report generator.

    Creates machine-readable reports for integration with other tools.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        include_results: bool = True,
        drop_result_fields: set[str] | None = None,
        sort_keys: bool = True,
        indent: int = 2,
        schema_version: str = "1.0",
    ) -> None:
        super().__init__(output_dir)
        self.include_results = include_results
        self.drop_result_fields = drop_result_fields or {"raw_response"}
        self.sort_keys = sort_keys
        self.indent = indent
        self.schema_version = schema_version

    @property
    def file_extension(self) -> str:
        return "json"

    def generate(self, report: BenchmarkReport, results: list[ResultMetric]) -> str:
        """Generate JSON report content."""
        payload = {
            "schema_version": self.schema_version,
            "meta": {
                "run_id": getattr(report, "run_id", None),
                "dataset_size": getattr(report, "dataset_size", None),
                "threshold_used": getattr(report, "threshold_used", None),
                "total_runtime_seconds": getattr(report, "total_runtime_seconds", None),
                "best_strategy_f1": getattr(report, "best_strategy_f1", None),
                "results_total": len(results),
                "results_errors": sum(1 for r in results if getattr(r, "has_error", False)),
            },
            "report": self._to_jsonable(report),
        }

        if self.include_results:
            payload["results"] = [self._to_jsonable(self._strip_fields(r)) for r in results]

        # Important: allow_nan=False => strict JSON
        return json.dumps(
            payload,
            indent=self.indent,
            ensure_ascii=False,
            sort_keys=self.sort_keys,
            default=self._serialize,
            allow_nan=False,
        )

    # ----------------------------
    # Helpers
    # ----------------------------

    def _strip_fields(self, r: ResultMetric) -> Any:
        """Return a shallow copy of the result-like object without huge fields (no mutation)."""
        d = self._to_mapping(r)
        for k in list(d.keys()):
            if k in self.drop_result_fields:
                d.pop(k, None)
        return d

    def _to_mapping(self, obj: Any) -> dict[str, Any]:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return dict(obj)
        # Pydantic v2
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        # fallback to attrs/plain objects
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return {"value": str(obj)}

    def _to_jsonable(self, obj: Any) -> Any:
        """
        Convert to JSON-safe structures early, reducing reliance on default serializer.
        """
        # Dataclasses / dict-like
        if (
            is_dataclass(obj)
            or isinstance(obj, dict)
            or hasattr(obj, "__dict__")
            or hasattr(obj, "model_dump")
        ):
            return self._to_mapping(obj)

        # Collections
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(x) for x in obj]

        return obj

    def _serialize(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        # Dataclasses
        if is_dataclass(obj):
            return asdict(obj)

        # Numpy scalars / arrays
        if isinstance(obj, np.floating):
            v = float(obj)
            # json.dumps with allow_nan=False will raise if nan/inf; keep as None
            if not np.isfinite(v):
                return None
            return v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Python floats that might be nan/inf
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
            return obj

        # pathlib.Path
        if isinstance(obj, Path):
            return str(obj)

        # Enums
        if hasattr(obj, "value") and obj.__class__.__name__.endswith("Enum"):
            try:
                return obj.value
            except Exception:
                return str(obj)

        # datetime/date/time
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                pass

        # bytes
        if isinstance(obj, (bytes, bytearray)):
            return base64.b64encode(bytes(obj)).decode("ascii")

        # Sets / tuples / other iterables (avoid treating strings as iterables)
        if isinstance(obj, Iterable) and not isinstance(obj, (str, dict)):
            try:
                return [self._serialize(x) for x in obj]
            except Exception:
                pass

        # Fallback
        return str(obj)
