"""
VerificationOracle Port
=======================

Abstract interface for claim verification against knowledge stores.
Supports pluggable verification strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.domain.results import CitationTrace, VerificationStatus


class VerificationStrategy(StrEnum):
    """Available verification strategies."""

    GRAPH_EXACT = auto()  # Exact matching in knowledge graph
    VECTOR_SEMANTIC = auto()  # Semantic similarity in vector store
    HYBRID = auto()  # Combination of graph + vector
    CASCADING = auto()  # Try graph first, fall back to vector
    MCP_ENHANCED = auto()  # MCP sources first, fallback to local stores
    ADAPTIVE = auto()  # Intelligent tiered collection with early-exit


class VerificationOracle(ABC):
    """
    Port for verifying claims against knowledge sources.

    Orchestrates the verification process using configurable
    strategies that combine graph and vector lookups.

    Responsibilities:
    - Apply verification strategy to each claim
    - Aggregate evidence from multiple sources
    - Determine verification status
    - Generate citation traces for transparency
    """

    @abstractmethod
    async def verify_claim(
        self,
        claim: Claim,
        strategy: VerificationStrategy = VerificationStrategy.HYBRID,
        *,
        target_sources: int | None = None,
    ) -> tuple[VerificationStatus, CitationTrace]:
        """
        Verify a single claim against knowledge sources.

        Args:
            claim: The claim to verify.
            strategy: Verification strategy to use.

        Returns:
            Tuple of (verification status, citation trace with evidence).

        Raises:
            VerificationError: If verification cannot be performed.
        """
        ...

    @abstractmethod
    async def verify_claims(
        self,
        claims: list[Claim],
        strategy: VerificationStrategy = VerificationStrategy.HYBRID,
        *,
        target_sources: int | None = None,
    ) -> list[tuple[VerificationStatus, CitationTrace]]:
        """
        Verify multiple claims (may parallelize internally).

        Args:
            claims: List of claims to verify.
            strategy: Verification strategy to use.

        Returns:
            List of (status, trace) tuples, one per claim.
        """
        ...

    @abstractmethod
    async def set_strategy(self, strategy: VerificationStrategy) -> None:
        """
        Change the default verification strategy.

        Args:
            strategy: New default strategy.
        """
        ...

    @property
    @abstractmethod
    def current_strategy(self) -> VerificationStrategy:
        """Return the current default verification strategy."""
        ...

    async def health_check(self) -> bool:
        """Check if the oracle and its dependencies are operational."""
        return True
