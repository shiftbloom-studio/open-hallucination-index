"""Benchmark dataset loaders.

Each loader is an async generator yielding `BenchmarkExample` instances.
Datasets are fetched via the `datasets` (Hugging Face) library where
possible and cached locally under `~/.cache/ohi-benchmarks/`. For
benchmarks with specialised formats (e.g. FActScore atomic-fact splits),
the loader wraps the benchmark's own preprocessing.

Phase 0 ships a minimal set of loaders backed by stub/synthetic data so
the benchmark runner can be wired end-to-end without requiring network.
Real loaders are filled in as each phase needs them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from scripts.benchmark.types import BenchmarkExample

_CACHE_DIR = Path.home() / ".cache" / "ohi-benchmarks"


# ---------------------------------------------------------------------------
# Registered datasets: name → async loader.
# Filled in as each phase needs them. Phase 0 just registers the names so
# `run_baseline.py --datasets factscore truthfulqa` can dispatch.
# ---------------------------------------------------------------------------


class DatasetNotAvailable(RuntimeError):
    """Raised when a dataset loader is registered but not yet implemented."""


async def _not_yet_implemented(name: str) -> AsyncIterator[BenchmarkExample]:
    raise DatasetNotAvailable(
        f"Dataset '{name}' loader is not yet implemented. This is a Phase 0 "
        f"placeholder — fill in when the phase that needs it begins."
    )
    # unreachable; keeps type checker happy
    yield  # type: ignore[unreachable]


async def load_factscore(
    *, limit: int | None = None
) -> AsyncIterator[BenchmarkExample]:
    """Atomic-fact factuality benchmark (Min et al., 2023).

    Phase 0 stub — returns a small synthetic fixture so the runner can be
    wired end-to-end without a HF token. Real loader (downloading the
    FActScore atomic-fact split) lands alongside the v1 baseline run.
    """
    fixtures = [
        BenchmarkExample(
            id=f"factscore-syn-{i}",
            text=(
                "Barack Obama was the 44th president of the United States."
                if i % 2 == 0
                else "Barack Obama was the 45th president of the United States."
            ),
            expected_label="true" if i % 2 == 0 else "false",
            metadata={"benchmark": "factscore", "split": "synthetic"},
        )
        for i in range(10)
    ]
    n = min(limit, len(fixtures)) if limit else len(fixtures)
    for example in fixtures[:n]:
        yield example


async def load_truthfulqa(
    *, limit: int | None = None
) -> AsyncIterator[BenchmarkExample]:
    """TruthfulQA (Lin et al., 2022) — adversarial-prompt truthfulness."""
    async for ex in _not_yet_implemented("truthfulqa"):
        yield ex


async def load_halueval(*, limit: int | None = None) -> AsyncIterator[BenchmarkExample]:
    """HaluEval — general hallucination detection."""
    async for ex in _not_yet_implemented("halueval"):
        yield ex


async def load_pubmedqa(*, limit: int | None = None) -> AsyncIterator[BenchmarkExample]:
    """PubMedQA — biomedical factoid Q&A."""
    async for ex in _not_yet_implemented("pubmedqa"):
        yield ex


async def load_legalbench_entailment(
    *, limit: int | None = None
) -> AsyncIterator[BenchmarkExample]:
    """LegalBench entailment subset — contract/statute entailment."""
    async for ex in _not_yet_implemented("legalbench-entailment"):
        yield ex


async def load_liar(*, limit: int | None = None) -> AsyncIterator[BenchmarkExample]:
    """LIAR — political claim truthfulness (social domain benchmark)."""
    async for ex in _not_yet_implemented("liar"):
        yield ex


REGISTRY: dict[str, callable] = {  # type: ignore[type-arg]
    "factscore": load_factscore,
    "truthfulqa": load_truthfulqa,
    "halueval": load_halueval,
    "pubmedqa": load_pubmedqa,
    "legalbench-entailment": load_legalbench_entailment,
    "liar": load_liar,
}


def list_datasets() -> list[str]:
    return sorted(REGISTRY.keys())


def cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR
