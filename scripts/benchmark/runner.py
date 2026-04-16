"""Engine-agnostic benchmark runner.

Runs a `VerificationEngine` over a list of `BenchmarkExample` with bounded
concurrency, writes JSONL atomically to a target path.

Any engine exception is captured as `BenchmarkResult.error` so a single
bad example never kills the whole run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from scripts.benchmark.types import (
    BenchmarkExample,
    BenchmarkResult,
    VerificationEngine,
)

logger = logging.getLogger(__name__)


async def _run_one(
    engine: VerificationEngine,
    example: BenchmarkExample,
    semaphore: asyncio.Semaphore,
) -> BenchmarkResult:
    async with semaphore:
        start = time.perf_counter()
        try:
            result = await engine.verify(example)
            # Normalise latency to runner-measured wall time if engine left it 0
            if result.latency_ms <= 0:
                elapsed = (time.perf_counter() - start) * 1000.0
                result = BenchmarkResult(
                    example_id=result.example_id,
                    predicted_label=result.predicted_label,
                    p_true=result.p_true,
                    interval=result.interval,
                    raw_response=result.raw_response,
                    latency_ms=elapsed,
                    error=result.error,
                )
            return result
        except Exception as exc:  # broad on purpose: engines are third-party
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "engine=%s example=%s failed: %s", engine.name, example.id, exc
            )
            return BenchmarkResult(
                example_id=example.id,
                predicted_label="",
                p_true=None,
                interval=None,
                raw_response={},
                latency_ms=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )


def _result_to_jsonable(r: BenchmarkResult) -> dict:
    d = asdict(r)
    # asdict leaves the tuple as-is; JSON list-encoding is fine. Tuple is
    # retained on read via Pydantic / manual coercion.
    return d


async def run_benchmark(
    engine: VerificationEngine,
    examples: list[BenchmarkExample],
    *,
    concurrency: int = 4,
    output_path: Path,
) -> Path:
    """Run `engine` over `examples`, write JSONL to `output_path`.

    Concurrency is bounded by `asyncio.Semaphore(concurrency)` so the engine
    never sees more than N in-flight requests.

    Atomic write: results are streamed to `<output_path>.tmp` and renamed
    on completion, so a crashed run never leaves a half-written file at the
    target path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [_run_one(engine, ex, sem) for ex in examples]

    # We stream writes to keep memory bounded on long benchmarks (1k+ examples).
    with tmp_path.open("w", encoding="utf-8") as f:
        # Use `asyncio.as_completed` so results stream out in completion order
        # rather than submission order.
        for coro in asyncio.as_completed(tasks):
            result: BenchmarkResult = await coro
            f.write(json.dumps(_result_to_jsonable(result), ensure_ascii=False))
            f.write("\n")
            f.flush()

    tmp_path.replace(output_path)
    return output_path
