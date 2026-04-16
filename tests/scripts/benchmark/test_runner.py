"""Integration tests for scripts/benchmark/runner.py.

Uses a MockEngine (no network, no model) to verify JSONL round-trip and
concurrency-bounded execution.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.benchmark.runner import run_benchmark
from scripts.benchmark.types import (
    BenchmarkExample,
    BenchmarkResult,
    VerificationEngine,
)


class MockEngine(VerificationEngine):
    """Deterministic mock engine that echoes predictions from a dict."""

    name = "mock-v0"

    def __init__(
        self,
        predictions: dict[str, str],
        *,
        delay_ms: float = 0.0,
        fail_ids: set[str] | None = None,
    ) -> None:
        self._predictions = predictions
        self._delay_ms = delay_ms
        self._fail_ids = fail_ids or set()

    async def verify(self, example: BenchmarkExample) -> BenchmarkResult:
        if self._delay_ms:
            await asyncio.sleep(self._delay_ms / 1000.0)
        if example.id in self._fail_ids:
            return BenchmarkResult(
                example_id=example.id,
                predicted_label="",
                p_true=None,
                interval=None,
                raw_response={},
                latency_ms=self._delay_ms,
                error="simulated failure",
            )
        return BenchmarkResult(
            example_id=example.id,
            predicted_label=self._predictions[example.id],
            p_true=0.9 if self._predictions[example.id] == "true" else 0.1,
            interval=None,
            raw_response={"mock": True},
            latency_ms=self._delay_ms,
            error=None,
        )


@pytest.mark.asyncio
async def test_run_benchmark_writes_jsonl(tmp_path: Path) -> None:
    examples = [
        BenchmarkExample(
            id=str(i), text=f"claim {i}", expected_label="true", metadata={}
        )
        for i in range(5)
    ]
    engine = MockEngine({str(i): "true" for i in range(5)})
    output_path = tmp_path / "results.jsonl"

    result_path = await run_benchmark(
        engine, examples, concurrency=2, output_path=output_path
    )

    assert result_path == output_path
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 5
    assert {line["example_id"] for line in lines} == {"0", "1", "2", "3", "4"}
    assert all(line["predicted_label"] == "true" for line in lines)
    assert all(line["error"] is None for line in lines)


@pytest.mark.asyncio
async def test_run_benchmark_captures_engine_errors(tmp_path: Path) -> None:
    examples = [
        BenchmarkExample(id="ok", text="a", expected_label="true", metadata={}),
        BenchmarkExample(id="bad", text="b", expected_label="true", metadata={}),
    ]
    engine = MockEngine({"ok": "true"}, fail_ids={"bad"})
    output_path = tmp_path / "results.jsonl"

    await run_benchmark(engine, examples, concurrency=1, output_path=output_path)

    with output_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    by_id = {line["example_id"]: line for line in lines}
    assert by_id["ok"]["error"] is None
    assert by_id["bad"]["error"] == "simulated failure"


@pytest.mark.asyncio
async def test_run_benchmark_atomic_write(tmp_path: Path) -> None:
    """The runner writes to a temp file then renames — so a crashed run
    never leaves a half-complete output file at the target path."""
    examples = [
        BenchmarkExample(id=str(i), text=f"c{i}", expected_label="true", metadata={})
        for i in range(3)
    ]
    engine = MockEngine({str(i): "true" for i in range(3)})
    output_path = tmp_path / "results.jsonl"

    await run_benchmark(engine, examples, concurrency=1, output_path=output_path)

    # Check no .tmp file left behind
    assert output_path.exists()
    assert not (tmp_path / "results.jsonl.tmp").exists()


@pytest.mark.asyncio
async def test_run_benchmark_concurrency_bounded(tmp_path: Path) -> None:
    """Verify the semaphore actually bounds concurrency."""
    examples = [
        BenchmarkExample(id=str(i), text=f"c{i}", expected_label="true", metadata={})
        for i in range(10)
    ]
    # 50ms per request × 10 requests / concurrency=2 ≈ 250ms lower bound,
    # concurrency=10 ≈ 50ms. We'll assert > 100ms at concurrency=2.
    engine = MockEngine({str(i): "true" for i in range(10)}, delay_ms=50.0)
    output_path = tmp_path / "results.jsonl"

    import time

    start = time.monotonic()
    await run_benchmark(engine, examples, concurrency=2, output_path=output_path)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    # With concurrency=2 and 50ms per op × 10 ops, expect ≥ 200ms (5 batches of 2)
    assert elapsed_ms >= 150.0, f"Expected >= 150ms but got {elapsed_ms}"
