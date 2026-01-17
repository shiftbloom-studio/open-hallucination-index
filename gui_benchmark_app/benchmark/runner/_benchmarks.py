"""
Individual benchmark implementations.

Each benchmark method:
1. Loads its specific dataset
2. Processes items concurrently with semaphore control
3. Updates the live display during execution
4. Returns metrics and latencies

Benchmarks:
- Hallucination Detection: Binary classification of claims
- TruthfulQA: Truthfulness evaluation
- FActScore: Factual accuracy scoring with claim decomposition

Note: Error logging uses display.log_error() instead of logger.error()
to prevent corrupting the Rich Live display.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any

from benchmark.comparison_metrics import (
    FActScoreMetrics,
    HallucinationMetrics,
    TruthfulQAMetrics,
)
from benchmark.datasets import HallucinationLoader, TruthfulQALoader
from benchmark.runner._display import LiveBenchmarkDisplay
from benchmark.runner._types import LiveStats

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.evaluators import BaseEvaluator, EvaluatorResult, FActScoreResult

logger = logging.getLogger(__name__)

# Context/evidence cap per sample for RAG evaluation metrics
MAX_CONTEXTS_PER_SAMPLE = 8  # Balance between coverage and performance


class StopRequested(RuntimeError):
    pass


def _maybe_record_verification(display: Any, payload: dict[str, Any]) -> None:
    record_fn = getattr(display, "record_verification", None)
    if callable(record_fn):
        record_fn(payload)


def _safe_evidence(result: EvaluatorResult, limit: int = 5) -> list[dict[str, Any]]:
    evidence_payload: list[dict[str, Any]] = []
    for ev in result.evidence[:limit]:
        evidence_payload.append(
            {
                "text": (ev.text or "")[:300],
                "source": ev.source,
                "similarity_score": ev.similarity_score,
                "metadata": ev.metadata,
            }
        )
    return evidence_payload


def _extract_response_text(result: Any, *, max_chars: int = 6000) -> str:
    """Best-effort extraction of a textual model response.

    Some evaluators keep the full raw model output (e.g. `raw_response`) or
    a cleaned explanation/rationale. We keep this optional and capped.
    """
    for attr in (
        "response_text",
        "raw_response",
        "explanation",
        "rationale",
        "analysis",
        "text",
    ):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            return text[:max_chars]
    return ""


def _parse_relevant_sources(value: Any) -> list[str]:
    """Parse one or many relevant source IDs from dataset fields."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    # common separators for multi-source fields
    for sep in (";", "|", ","):
        if sep in text:
            parts = [p.strip() for p in text.split(sep)]
            return [p for p in parts if p]
    return [text]


def _should_stop(display: Any) -> bool:
    checker = getattr(display, "should_stop", None)
    return bool(callable(checker) and checker())


# =============================================================================
# Timeout Wrappers
# =============================================================================


async def verify_with_timeout(
    evaluator: BaseEvaluator,
    claim: str,
    timeout_seconds: float,
    label: str = "verify",
) -> EvaluatorResult:
    """
    Call evaluator.verify() with timeout protection.
    
    Args:
        evaluator: Evaluator instance
        claim: Claim text to verify
        timeout_seconds: Maximum time to wait
        label: Label for error messages
        
    Returns:
        EvaluatorResult from the evaluator
        
    Raises:
        RuntimeError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(
            evaluator.verify(claim),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Latency exceeded {timeout_seconds:.0f}s for {evaluator.name} ({label}). "
            f"Aborting benchmark."
        ) from exc


async def decompose_with_timeout(
    evaluator: BaseEvaluator,
    text: str,
    timeout_seconds: float,
    label: str = "decompose",
) -> FActScoreResult:
    """
    Call evaluator.decompose_and_verify() with timeout protection.
    
    Args:
        evaluator: Evaluator instance
        text: Text to decompose and verify
        timeout_seconds: Maximum time to wait
        label: Label for error messages
        
    Returns:
        FActScoreResult from the evaluator
        
    Raises:
        RuntimeError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(
            evaluator.decompose_and_verify(text),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Latency exceeded {timeout_seconds:.0f}s for {evaluator.name} ({label}). "
            f"Aborting benchmark."
        ) from exc


def guard_latency(
    latency_ms: float,
    max_latency_ms: float,
    evaluator_name: str,
    label: str,
) -> None:
    """
    Check if latency exceeds maximum threshold.
    
    Args:
        latency_ms: Measured latency
        max_latency_ms: Maximum allowed latency
        evaluator_name: Name for error message
        label: Context label for error message
        
    Raises:
        RuntimeError: If latency exceeds threshold
    """
    if latency_ms >= max_latency_ms:
        raise RuntimeError(
            f"Latency exceeded {max_latency_ms / 1000:.0f}s "
            f"for {evaluator_name} ({label}). Aborting benchmark."
        )


# =============================================================================
# Hallucination Detection Benchmark
# =============================================================================


async def run_hallucination_benchmark(
    evaluator: BaseEvaluator,
    display: LiveBenchmarkDisplay,
    stats: LiveStats,
    dataset_path: Path | None,
    extended_dataset_path: Path | None,
    max_samples: int,
    concurrency: int,
    max_latency_ms: float,
) -> tuple[HallucinationMetrics, list[float]]:
    """
    Run hallucination detection benchmark with live display.
    
    Args:
        evaluator: Evaluator to benchmark
        display: Live display for progress updates
        stats: Shared statistics object
        dataset_path: Path to hallucination CSV dataset
        extended_dataset_path: Path to extended dataset (optional)
        max_samples: Maximum samples to process
        concurrency: Maximum concurrent requests
        max_latency_ms: Maximum allowed latency per request
        
    Returns:
        Tuple of (HallucinationMetrics, latencies list)
    """
    loader = HallucinationLoader(dataset_path)
    timeout_seconds = max_latency_ms / 1000
    
    # Load dataset with fallback
    try:
        if extended_dataset_path and extended_dataset_path.exists():
            dataset = loader.load_combined(
                csv_path=dataset_path,
                include_huggingface=False,
                hf_max_samples=max_samples,
            )
        elif dataset_path and dataset_path.exists():
            dataset = loader.load_csv()
        else:
            dataset = loader.load_from_huggingface(max_samples=max_samples)
    except FileNotFoundError:
        dataset = loader.load_from_huggingface(max_samples=max_samples)
    
    # Track metrics only for the samples we actually evaluated.
    # (This stays accurate even with max_samples or filtered loaders.)
    metrics = HallucinationMetrics()
    latencies: list[float] = []
    
    display.start_task(
        f"Hallucination Detection ({evaluator.name})",
        total=len(dataset.cases),
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _process_case(case: Any) -> tuple[EvaluatorResult, Any]:
        async with semaphore:
            result = await verify_with_timeout(
                evaluator,
                case.text,
                timeout_seconds,
                "hallucination",
            )
            return result, case
    
    tasks = [_process_case(case) for case in dataset.cases]
    pending: set[asyncio.Task[tuple[EvaluatorResult, Any]]] = set(
        asyncio.ensure_future(t) for t in tasks
    )
    
    while pending:
        if _should_stop(display):
            for task in pending:
                task.cancel()
            raise StopRequested("Benchmark stopped by user")
        done, pending = await asyncio.wait(
            pending,
            timeout=2.0,  # Increased to reduce display jitter
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        display.force_refresh()
        
        for future in done:
            result, case = future.result()
            
            latencies.append(result.latency_ms)
            guard_latency(result.latency_ms, max_latency_ms, evaluator.name, "hallucination")
            
            # Compare prediction to ground truth
            predicted = result.predicted_label
            expected = case.label

            # Collect auxiliary signals for AURC / BEIR-style retrieval / ALCE-style citation stats
            confidence = float(result.trust_score) if result.trust_score is not None else 0.0
            if math.isnan(confidence):  # NaN guard
                confidence = 0.0

            retrieved_sources = [str(ev.source) for ev in (result.evidence or []) if getattr(ev, "source", None)]
            relevant_sources = _parse_relevant_sources(getattr(case, "source", None))
            response_text = _extract_response_text(result)
            evidence_count = len(result.evidence or [])

            # RAG signals (optional): question/answer/contexts/ground-truth text
            # These fields are extracted and normalized (with type coercion and fallbacks)
            # from the evaluation result and case. They do not mutate the original objects.
            question = str(getattr(case, "question", None) or getattr(case, "query", None) or case.text)
            answer = str(getattr(result, "answer", None) or getattr(result, "generated_answer", None) or response_text)
            contexts = [
                str(text)
                for ev in (result.evidence or [])
                if (text := (getattr(ev, "text", None) or getattr(ev, "snippet", None)))
            ]
            # Cap contexts for speed + memory (balance between coverage and performance)
            contexts = contexts[:MAX_CONTEXTS_PER_SAMPLE]
            ground_truth = str(
                getattr(case, "ground_truth", None)
                or getattr(case, "reference", None)
                or getattr(case, "answer", None)
                or ""
            )

            metrics.add_sample(
                expected_is_fact=bool(expected),
                predicted_is_fact=bool(predicted),
                confidence=confidence,
                retrieved_sources=retrieved_sources,
                relevant_sources=relevant_sources,
                response_text=response_text,
                evidence_count=evidence_count,
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

            is_correct = bool(predicted == expected)
            
            display.advance(1, latency_ms=result.latency_ms)
            has_error = result.error is not None
            if has_error:
                # Use display.log_error to prevent corrupting the Live display
                claim_preview = case.text[:50] if len(case.text) > 50 else case.text
                display.log_error(f"Hallucination error: {result.error} (claim: '{claim_preview}...')")
            display.add_result(correct=is_correct, error=has_error)

            _maybe_record_verification(
                display,
                {
                    "metric": "hallucination",
                    "evaluator": evaluator.name,
                    "tier": getattr(evaluator, "tier", "baseline"),
                    "claim": case.text,
                    "expected": expected,
                    "predicted": predicted,
                    "verdict": getattr(result.verdict, "value", str(result.verdict)),
                    "trust_score": result.trust_score,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "response_text": _extract_response_text(result),
                    "meta": {
                        "case_id": case.id,
                        "domain": case.domain,
                        "difficulty": case.difficulty,
                        "source": case.source,
                        "hallucination_type": case.hallucination_type,
                    },
                    "evidence": _safe_evidence(result),
                },
            )
    
    return metrics, latencies


# =============================================================================
# TruthfulQA Benchmark
# =============================================================================


async def run_truthfulqa_benchmark(
    evaluator: BaseEvaluator,
    display: LiveBenchmarkDisplay,
    stats: LiveStats,
    max_samples: int | None,
    categories: list[str] | None,
    concurrency: int,
    max_latency_ms: float,
) -> tuple[TruthfulQAMetrics, list[float]]:
    """
    Run TruthfulQA benchmark with live display.
    
    Args:
        evaluator: Evaluator to benchmark
        display: Live display for progress updates
        stats: Shared statistics object
        max_samples: Maximum samples to process (None for all)
        categories: Optional category filter
        concurrency: Maximum concurrent requests
        max_latency_ms: Maximum allowed latency per request
        
    Returns:
        Tuple of (TruthfulQAMetrics, latencies list)
    """
    loader = TruthfulQALoader()
    timeout_seconds = max_latency_ms / 1000
    
    try:
        claims = loader.load_for_verification(
            max_samples=max_samples,
            categories=categories,
        )
    except Exception as e:
        # Log warning - this happens before display is active, so logger is safe
        logger.warning(f"Failed to load TruthfulQA: {e}")
        return TruthfulQAMetrics(), []
    
    metrics = TruthfulQAMetrics(total_questions=len(claims))
    latencies: list[float] = []
    
    display.start_task(
        f"TruthfulQA ({evaluator.name})",
        total=len(claims),
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _process_item(item: tuple[str, bool]) -> tuple[EvaluatorResult, bool]:
        claim_text, is_correct = item
        async with semaphore:
            result = await verify_with_timeout(
                evaluator,
                claim_text,
                timeout_seconds,
                "truthfulqa",
            )
            return result, is_correct
    
    tasks = [_process_item(item) for item in claims]
    pending: set[asyncio.Task[tuple[EvaluatorResult, bool]]] = set(
        asyncio.ensure_future(t) for t in tasks
    )
    
    while pending:
        if _should_stop(display):
            for task in pending:
                task.cancel()
            raise StopRequested("Benchmark stopped by user")
        done, pending = await asyncio.wait(
            pending,
            timeout=2.0,  # Increased to reduce display jitter
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        display.force_refresh()
        
        for future in done:
            result, is_correct = future.result()
            
            latencies.append(result.latency_ms)
            guard_latency(result.latency_ms, max_latency_ms, evaluator.name, "truthfulqa")
            
            predicted = result.predicted_label
            correct = predicted == is_correct
            if correct:
                metrics.correct_predictions += 1
            
            display.advance(1, latency_ms=result.latency_ms)
            is_error = result.error is not None
            if is_error:
                # Use display.log_error to prevent corrupting the Live display
                display.log_error(f"TruthfulQA error: {result.error}")
            display.add_result(correct=correct, error=is_error)

            _maybe_record_verification(
                display,
                {
                    "metric": "truthfulqa",
                    "evaluator": evaluator.name,
                    "tier": getattr(evaluator, "tier", "baseline"),
                    "claim": result.claim,
                    "expected": is_correct,
                    "predicted": result.predicted_label,
                    "verdict": getattr(result.verdict, "value", str(result.verdict)),
                    "trust_score": result.trust_score,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "meta": {
                        "source": "truthfulqa",
                    },
                    "evidence": _safe_evidence(result),
                },
            )
    
    return metrics, latencies


# =============================================================================
# FActScore Benchmark
# =============================================================================

# Default sample texts for FActScore evaluation
FACTSCORE_SAMPLE_TEXTS = [
    "Albert Einstein was born in Germany in 1879. He developed the theory of relativity and won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
    "The Eiffel Tower is located in Paris, France. It was constructed in 1889 for the World's Fair and stands at 324 meters tall. It is made of iron and was designed by Gustave Eiffel.",
    "Python is a programming language created by Guido van Rossum in 1991. It is known for its simple syntax and is widely used in web development, data science, and artificial intelligence.",
    "The human heart has four chambers and pumps blood throughout the body. It beats approximately 100,000 times per day and is located in the chest cavity.",
    "World War II ended in 1945. It involved most of the world's nations and resulted in significant geopolitical changes including the formation of the United Nations.",
]


async def run_factscore_benchmark(
    evaluator: BaseEvaluator,
    display: LiveBenchmarkDisplay,
    stats: LiveStats,
    max_samples: int | None,
    concurrency: int,
    max_latency_ms: float,
) -> tuple[FActScoreMetrics, list[float]]:
    """
    Run FActScore benchmark with live display.
    
    Args:
        evaluator: Evaluator to benchmark
        display: Live display for progress updates
        stats: Shared statistics object
        max_samples: Maximum samples (defaults to len of sample texts)
        concurrency: Maximum concurrent requests
        max_latency_ms: Maximum allowed latency per request
        
    Returns:
        Tuple of (FActScoreMetrics, latencies list)
    """
    timeout_seconds = max_latency_ms / 1000
    
    sample_limit = max_samples or len(FACTSCORE_SAMPLE_TEXTS)
    sample_texts = FACTSCORE_SAMPLE_TEXTS[:sample_limit]
    
    metrics = FActScoreMetrics(total_texts=len(sample_texts))
    latencies: list[float] = []
    
    display.start_task(
        f"FActScore ({evaluator.name})",
        total=len(sample_texts),
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _process_text(text: str) -> FActScoreResult:
        async with semaphore:
            return await decompose_with_timeout(
                evaluator,
                text,
                timeout_seconds,
                "factscore",
            )
    
    tasks = [_process_text(text) for text in sample_texts]
    pending: set[asyncio.Task[FActScoreResult]] = set(
        asyncio.ensure_future(t) for t in tasks
    )
    
    while pending:
        if _should_stop(display):
            for task in pending:
                task.cancel()
            raise StopRequested("Benchmark stopped by user")
        done, pending = await asyncio.wait(
            pending,
            timeout=2.0,  # Increased to reduce display jitter
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        display.force_refresh()
        
        for future in done:
            result = future.result()
            
            latencies.append(result.latency_ms)
            guard_latency(result.latency_ms, max_latency_ms, evaluator.name, "factscore")
            
            metrics.total_facts += result.total_facts
            metrics.supported_facts += result.supported_facts
            metrics.facts_per_text.append(result.total_facts)
            
            if result.total_facts > 0:
                metrics.scores.append(result.factscore)
            
            display.advance(1, latency_ms=result.latency_ms)
            display.add_result(correct=True)  # FActScore doesn't have pass/fail

            _maybe_record_verification(
                display,
                {
                    "metric": "factscore",
                    "evaluator": evaluator.name,
                    "tier": getattr(evaluator, "tier", "baseline"),
                    "claim": result.original_text,
                    "expected": None,
                    "predicted": None,
                    "verdict": "factscore",
                    "trust_score": result.factscore,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "meta": {
                        "total_facts": result.total_facts,
                        "supported_facts": result.supported_facts,
                        "factscore": result.factscore,
                    },
                    "evidence": [
                        {
                            "text": (fact.text or "")[:300],
                            "source": "atomic_fact",
                            "similarity_score": 0.0,
                            "metadata": {"verified": fact.verified},
                        }
                        for fact in result.atomic_facts[:6]
                    ],
                },
            )
    
    return metrics, latencies


# =============================================================================
# Warmup
# =============================================================================


async def warmup_evaluator(
    evaluator: BaseEvaluator,
    warmup_requests: int,
    include_factscore: bool = False,
) -> None:
    """
    Warm up evaluator API and model before measuring metrics.
    
    Args:
        evaluator: Evaluator to warm up
        warmup_requests: Number of warmup requests
        include_factscore: Whether to also warm up FActScore
    """
    if warmup_requests <= 0:
        return
    
    warmup_samples = [
        "The Eiffel Tower is in Paris.",
        "Albert Einstein was born in 1879.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Python was created by Guido van Rossum.",
        "The Earth orbits the Sun.",
    ]
    warmup_claims = warmup_samples[:min(warmup_requests, len(warmup_samples))]
    
    try:
        if hasattr(evaluator, "verify_batch"):
            await evaluator.verify_batch(warmup_claims, concurrency=len(warmup_claims))
        else:
            for claim in warmup_claims:
                await evaluator.verify(claim)
        
        if include_factscore:
            warmup_text = "The Moon orbits the Earth. It reflects sunlight and affects tides."
            await evaluator.decompose_and_verify(warmup_text)
    
    except Exception as e:
        logger.warning(f"Warmup failed for {evaluator.name}: {e}")
