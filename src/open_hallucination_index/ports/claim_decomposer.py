"""
ClaimDecomposer Port
====================

Abstract interface for decomposing unstructured text into atomic claims.
Implementations may use LLMs, rule-based parsers, or hybrid approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim


class ClaimDecomposer(ABC):
    """
    Port for claim extraction from unstructured text.

    Responsibilities:
    - Parse input text into sentence/clause units
    - Extract atomic, verifiable claims
    - Structure claims as subject-predicate-object triplets where possible
    - Assign claim types and confidence scores

    Implementations might use:
    - LLM-based extraction (GPT, Claude, local models)
    - Dependency parsing + rule-based extraction
    - Hybrid approaches with LLM refinement
    """

    @abstractmethod
    async def decompose(self, text: str) -> list[Claim]:
        """
        Decompose text into a list of atomic claims.

        Args:
            text: Unstructured input text to analyze.

        Returns:
            List of extracted claims with structured representation.

        Raises:
            DecompositionError: If text cannot be processed.
        """
        ...

    @abstractmethod
    async def decompose_with_context(
        self,
        text: str,
        context: str | None = None,
        max_claims: int | None = None,
    ) -> list[Claim]:
        """
        Decompose text with additional context for disambiguation.

        Args:
            text: Unstructured input text to analyze.
            context: Optional context (e.g., document title, topic).
            max_claims: Optional limit on number of claims to extract.

        Returns:
            List of extracted claims.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the decomposer is operational."""
        return True
