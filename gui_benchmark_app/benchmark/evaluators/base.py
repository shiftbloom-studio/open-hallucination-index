"""
Evaluator Base Classes
======================

Abstract base class and result models for claim evaluators.
Each evaluator verifies claims and returns structured results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationVerdict(str, Enum):
    """Verification result verdict."""

    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"
    PARTIAL = "partial"


@dataclass
class EvidenceItem:
    """A piece of evidence supporting or refuting a claim."""

    text: str
    source: str
    similarity_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorResult:
    """
    Result from a single claim evaluation.

    Attributes:
        claim: The original claim text.
        verdict: Verification verdict (supported/refuted/unverifiable).
        trust_score: Confidence score (0.0 to 1.0).
        latency_ms: Time taken for evaluation in milliseconds.
        evidence: List of evidence items used.
        evaluator: Name of the evaluator that produced this result.
        error: Error message if evaluation failed.
        metadata: Additional evaluator-specific data.
    """

    claim: str
    verdict: VerificationVerdict
    trust_score: float
    latency_ms: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    evaluator: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_supported(self) -> bool:
        """Whether the claim is considered supported/factual."""
        return self.verdict == VerificationVerdict.SUPPORTED

    @property
    def has_error(self) -> bool:
        """Whether an error occurred during evaluation."""
        return self.error is not None

    @property
    def predicted_label(self) -> bool:
        """
        Convert verdict to binary label.

        Returns:
            True if claim is considered factual, False otherwise.
        """
        return self.verdict in (
            VerificationVerdict.SUPPORTED,
            VerificationVerdict.PARTIAL,
        )


@dataclass
class AtomicFact:
    """An atomic fact extracted from a longer text."""

    text: str
    source_text: str
    index: int
    verified: bool | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass
class FActScoreResult:
    """
    Result from FActScore evaluation.

    FActScore = (# supported facts) / (# total facts)
    """

    original_text: str
    atomic_facts: list[AtomicFact]
    latency_ms: float
    evaluator: str = ""
    error: str | None = None

    @property
    def total_facts(self) -> int:
        """Total number of atomic facts extracted."""
        return len(self.atomic_facts)

    @property
    def supported_facts(self) -> int:
        """Number of supported/verified facts."""
        return sum(1 for f in self.atomic_facts if f.verified is True)

    @property
    def factscore(self) -> float:
        """
        Compute FActScore (atomic fact precision).

        Returns:
            Score between 0.0 and 1.0.
        """
        if not self.atomic_facts:
            return 0.0
        return self.supported_facts / self.total_facts

    @property
    def factscore_with_penalty(self, gamma: float = 10.0) -> float:
        """
        FActScore with length penalty for short responses.

        Args:
            gamma: Penalty parameter (default 10).
        """
        if self.total_facts >= gamma:
            return self.factscore
        # Apply penalty for responses with few facts
        penalty = self.total_facts / gamma
        return self.factscore * penalty


class BaseEvaluator(ABC):
    """
    Abstract base class for claim evaluators.

    Each evaluator must implement:
    - verify(): Single claim verification
    - verify_batch(): Batch claim verification (optional optimization)
    - decompose_and_verify(): FActScore-style atomic fact evaluation
    """

    name: str = "base"

    @abstractmethod
    async def verify(self, claim: str) -> EvaluatorResult:
        """
        Verify a single claim.

        Args:
            claim: The claim text to verify.

        Returns:
            EvaluatorResult with verdict, score, and evidence.
        """
        ...

    async def verify_batch(
        self,
        claims: list[str],
        concurrency: int = 5,
    ) -> list[EvaluatorResult]:
        """
        Verify multiple claims in batch.

        Default implementation runs sequentially.
        Subclasses can override for parallel execution.

        Args:
            claims: List of claim texts to verify.
            concurrency: Max parallel requests.

        Returns:
            List of EvaluatorResult objects.
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def verify_with_limit(claim: str) -> EvaluatorResult:
            async with semaphore:
                return await self.verify(claim)

        return await asyncio.gather(*[verify_with_limit(c) for c in claims])

    @abstractmethod
    async def decompose_and_verify(self, text: str) -> FActScoreResult:
        """
        Decompose text into atomic facts and verify each.

        Used for FActScore evaluation.

        Args:
            text: Longer text to decompose and verify.

        Returns:
            FActScoreResult with atomic facts and verification status.
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the evaluator is operational.

        Returns:
            True if evaluator is ready, False otherwise.
        """
        return True

    async def close(self) -> None:
        """Clean up resources."""
        pass
