"""
Weighted Scorer
===============

Computes trust scores from claim verification results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from interfaces.scoring import Scorer
from models.results import (
    ClaimVerification,
    TrustScore,
    VerificationStatus,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Status weights for scoring
# These weights represent the base trust contribution of each status
STATUS_WEIGHTS = {
    VerificationStatus.SUPPORTED: 1.0,
    VerificationStatus.PARTIALLY_SUPPORTED: 0.85,  # Mostly trustworthy
    VerificationStatus.UNCERTAIN: 0.50,  # True uncertainty
    # Neutral - no evidence either way, slight benefit of doubt
    VerificationStatus.UNVERIFIABLE: 0.60,
    VerificationStatus.REFUTED: 0.0,
}

# Penalties for refuted claims (they hurt trust more than supported helps)
REFUTED_PENALTY_MULTIPLIER = 1.3  # Reduced from 1.5 - less harsh


class WeightedScorer(Scorer):
    """
    Weighted scoring implementation.

    Computes trust scores using:
    - Status-based weights (supported=1.0, refuted=0.0)
    - Confidence weighting from verification traces
    - Penalty multiplier for refuted claims
    """

    def __init__(
        self,
        refuted_penalty: float = REFUTED_PENALTY_MULTIPLIER,
        unverifiable_weight: float = 0.5,
    ) -> None:
        """
        Initialize the scorer.

        Args:
            refuted_penalty: Multiplier for refuted claim impact (>1 = harsher).
            unverifiable_weight: Base score for unverifiable claims.
        """
        self._refuted_penalty = refuted_penalty
        self._unverifiable_weight = unverifiable_weight

    @property
    def scoring_method(self) -> str:
        """Return the name of the scoring algorithm."""
        return "weighted_average"

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
        if not verifications:
            return TrustScore(
                overall=1.0,  # No claims = nothing to distrust
                claims_total=0,
                claims_supported=0,
                claims_refuted=0,
                claims_unverifiable=0,
                confidence=0.0,
                scoring_method=self.scoring_method,
            )

        # Count by status
        total = len(verifications)
        supported = sum(1 for v in verifications if v.status == VerificationStatus.SUPPORTED)
        refuted = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
        unverifiable = sum(1 for v in verifications if v.status == VerificationStatus.UNVERIFIABLE)

        # Compute weighted score
        weighted_sum = 0.0
        weight_sum = 0.0

        for v in verifications:
            contribution = await self.compute_claim_contribution(v)
            weight = v.trace.confidence

            # Apply penalty multiplier for refuted claims
            if v.status == VerificationStatus.REFUTED:
                weight *= self._refuted_penalty

            weighted_sum += contribution * weight
            weight_sum += weight

        # Calculate overall score
        overall = weighted_sum / weight_sum if weight_sum > 0 else 0.5

        # Clamp to [0, 1]
        overall = max(0.0, min(1.0, overall))

        # Meta-confidence: how confident are we in this score?
        # Higher if more claims, more verified (not unverifiable)
        verified_ratio = (total - unverifiable) / total if total > 0 else 0
        avg_confidence = sum(v.trace.confidence for v in verifications) / total if total > 0 else 0
        meta_confidence = verified_ratio * 0.5 + avg_confidence * 0.5

        return TrustScore(
            overall=round(overall, 4),
            claims_total=total,
            claims_supported=supported,
            claims_refuted=refuted,
            claims_unverifiable=unverifiable,
            confidence=round(meta_confidence, 4),
            scoring_method=self.scoring_method,
        )

    async def compute_claim_contribution(
        self,
        verification: ClaimVerification,
    ) -> float:
        """
        Compute a single claim's contribution to the overall score.

        Uses a more balanced approach:
        - Base weight from verification status (SUPPORTED=1.0, PARTIALLY_SUPPORTED=0.85, etc.)
        - Light adjustment based on claim extraction confidence
        - Light adjustment based on verification trace confidence

        The formula is designed to NOT over-penalize:
        - Uses additive blending instead of pure multiplication
        - High confidence evidence gets full weight
        - Low confidence only slightly reduces contribution

        Args:
            verification: Single claim verification result.

        Returns:
            Score contribution (0.0 - 1.0).
        """
        # If the oracle used the LLM plausibility fallback (no evidence),
        # use its bounded confidence directly as the base contribution.
        if "LLM plausibility prior used" in verification.trace.reasoning:
            return max(0.0, min(1.0, verification.trace.confidence))

        status = verification.status
        base_weight = STATUS_WEIGHTS.get(status, 0.5)

        # Get confidence values
        claim_confidence = verification.claim.confidence
        trace_confidence = verification.trace.confidence

        # Blend confidence values (average, with floor)
        # This prevents low extraction confidence from tanking good verification
        blended_confidence = max(0.5, (claim_confidence + trace_confidence) / 2)

        # Apply a gentle adjustment based on confidence
        # High confidence (1.0) = full base_weight
        # Medium confidence (0.5) = 85% of base_weight
        # Low confidence (0.0) = 70% of base_weight
        confidence_factor = 0.7 + 0.3 * blended_confidence

        adjusted = base_weight * confidence_factor

        return max(0.0, min(1.0, adjusted))


class StrictScorer(Scorer):
    """
    Strict scoring - any refuted claim results in low trust.

    Suitable for high-stakes verification where accuracy is critical.
    """

    @property
    def scoring_method(self) -> str:
        """Return the name of the scoring algorithm."""
        return "strict"

    async def compute_score(
        self,
        verifications: list[ClaimVerification],
    ) -> TrustScore:
        """Compute strict trust score."""
        if not verifications:
            return TrustScore(
                overall=1.0,
                claims_total=0,
                claims_supported=0,
                claims_refuted=0,
                claims_unverifiable=0,
                confidence=0.0,
                scoring_method=self.scoring_method,
            )

        total = len(verifications)
        supported = sum(1 for v in verifications if v.status == VerificationStatus.SUPPORTED)
        refuted = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
        unverifiable = sum(1 for v in verifications if v.status == VerificationStatus.UNVERIFIABLE)

        # Any refuted claim = max 0.3 trust
        if refuted > 0:
            # More refuted = lower score
            penalty = min(0.3, 0.3 / refuted)
            overall = penalty
        else:
            # Score based on supported ratio
            verified = total - unverifiable
            overall = supported / verified if verified > 0 else 0.5

        avg_confidence = sum(v.trace.confidence for v in verifications) / total

        return TrustScore(
            overall=round(overall, 4),
            claims_total=total,
            claims_supported=supported,
            claims_refuted=refuted,
            claims_unverifiable=unverifiable,
            confidence=round(avg_confidence, 4),
            scoring_method=self.scoring_method,
        )

    async def compute_claim_contribution(
        self,
        verification: ClaimVerification,
    ) -> float:
        """Compute claim contribution for strict scoring."""
        if verification.status == VerificationStatus.REFUTED:
            return 0.0
        if verification.status == VerificationStatus.SUPPORTED:
            return 1.0
        return 0.5
