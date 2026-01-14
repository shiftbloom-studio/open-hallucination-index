"""
Scorer Port
===========

Abstract interface for computing trust scores from verification results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_hallucination_index.domain.results import (
        ClaimVerification,
        TrustScore,
    )


class Scorer(ABC):
    """
    Port for computing trust scores from claim verifications.

    Responsibilities:
    - Aggregate individual claim scores
    - Weight claims by importance/confidence
    - Compute overall trust score (0.0 - 1.0)
    - Calculate meta-confidence in the score

    Scoring considerations:
    - Supported claims increase trust
    - Refuted claims decrease trust heavily
    - Unverifiable claims may be neutral or slightly negative
    - Claim importance may vary (key facts vs. minor details)
    """

    @abstractmethod
    async def compute_score(
        self,
        verifications: list[ClaimVerification],
    ) -> TrustScore:
        """
        Compute aggregate trust score from claim verifications.

        Args:
            verifications: List of individual claim verification results.

        Returns:
            Aggregated trust score with breakdown.
        """
        ...

    @abstractmethod
    async def compute_claim_contribution(
        self,
        verification: ClaimVerification,
    ) -> float:
        """
        Compute a single claim's contribution to the overall score.

        Args:
            verification: Single claim verification result.

        Returns:
            Score contribution (0.0 - 1.0).
        """
        ...

    @property
    @abstractmethod
    def scoring_method(self) -> str:
        """Return the name of the scoring algorithm."""
        ...
