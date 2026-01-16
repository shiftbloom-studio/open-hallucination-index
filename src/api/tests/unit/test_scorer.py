"""
Tests for Scorer Implementation
===============================
"""

from uuid import uuid4

import pytest

from models.entities import Claim, ClaimType
from models.results import (
    CitationTrace,
    ClaimVerification,
    VerificationStatus,
)
from pipeline.scorer import StrictScorer, WeightedScorer


def create_verification(
    status: VerificationStatus,
    claim_confidence: float = 0.9,
    trace_confidence: float = 0.8,
) -> ClaimVerification:
    """Helper to create a verification with specific status."""
    claim = Claim(
        id=uuid4(),
        text="Test claim",
        claim_type=ClaimType.UNCLASSIFIED,
        confidence=claim_confidence,
    )

    trace = CitationTrace(
        claim_id=claim.id,
        status=status,
        reasoning="Test",
        supporting_evidence=[],
        refuting_evidence=[],
        confidence=trace_confidence,
        verification_strategy="hybrid",
    )

    return ClaimVerification(
        claim=claim,
        status=status,
        trace=trace,
        score_contribution=0.0,
    )


class TestWeightedScorer:
    """Tests for the WeightedScorer."""

    @pytest.fixture
    def scorer(self) -> WeightedScorer:
        """Create a scorer instance."""
        return WeightedScorer()

    @pytest.mark.asyncio
    async def test_empty_verifications(self, scorer: WeightedScorer) -> None:
        """Test scoring with no claims."""
        score = await scorer.compute_score([])

        assert score.overall == 1.0  # No claims = nothing to distrust
        assert score.claims_total == 0
        assert score.confidence == 0.0

    @pytest.mark.asyncio
    async def test_all_supported(self, scorer: WeightedScorer) -> None:
        """Test scoring when all claims are supported."""
        verifications = [
            create_verification(VerificationStatus.SUPPORTED),
            create_verification(VerificationStatus.SUPPORTED),
            create_verification(VerificationStatus.SUPPORTED),
        ]

        score = await scorer.compute_score(verifications)

        assert score.overall > 0.7  # High trust
        assert score.claims_supported == 3
        assert score.claims_refuted == 0

    @pytest.mark.asyncio
    async def test_all_refuted(self, scorer: WeightedScorer) -> None:
        """Test scoring when all claims are refuted."""
        verifications = [
            create_verification(VerificationStatus.REFUTED),
            create_verification(VerificationStatus.REFUTED),
        ]

        score = await scorer.compute_score(verifications)

        assert score.overall < 0.3  # Low trust
        assert score.claims_refuted == 2

    @pytest.mark.asyncio
    async def test_mixed_results(self, scorer: WeightedScorer) -> None:
        """Test scoring with mixed verification results."""
        verifications = [
            create_verification(VerificationStatus.SUPPORTED),
            create_verification(VerificationStatus.REFUTED),
            create_verification(VerificationStatus.UNVERIFIABLE),
        ]

        score = await scorer.compute_score(verifications)

        assert 0.3 < score.overall < 0.7  # Medium trust
        assert score.claims_total == 3
        assert score.claims_supported == 1
        assert score.claims_refuted == 1
        assert score.claims_unverifiable == 1

    @pytest.mark.asyncio
    async def test_claim_contribution_supported(self, scorer: WeightedScorer) -> None:
        """Test individual claim contribution for supported status."""
        verification = create_verification(VerificationStatus.SUPPORTED)

        contribution = await scorer.compute_claim_contribution(verification)

        assert 0.5 < contribution <= 1.0

    @pytest.mark.asyncio
    async def test_claim_contribution_refuted(self, scorer: WeightedScorer) -> None:
        """Test individual claim contribution for refuted status."""
        verification = create_verification(VerificationStatus.REFUTED)

        contribution = await scorer.compute_claim_contribution(verification)

        assert 0.0 <= contribution < 0.3

    @pytest.mark.asyncio
    async def test_scoring_method_name(self, scorer: WeightedScorer) -> None:
        """Test scoring method is correctly reported."""
        assert scorer.scoring_method == "weighted_average"


class TestStrictScorer:
    """Tests for the StrictScorer."""

    @pytest.fixture
    def scorer(self) -> StrictScorer:
        """Create a strict scorer instance."""
        return StrictScorer()

    @pytest.mark.asyncio
    async def test_any_refuted_tanks_score(self, scorer: StrictScorer) -> None:
        """Test that any refuted claim results in low trust."""
        verifications = [
            create_verification(VerificationStatus.SUPPORTED),
            create_verification(VerificationStatus.SUPPORTED),
            create_verification(VerificationStatus.REFUTED),  # One bad claim
        ]

        score = await scorer.compute_score(verifications)

        assert score.overall <= 0.3  # Strict penalty

    @pytest.mark.asyncio
    async def test_all_supported_high_score(self, scorer: StrictScorer) -> None:
        """Test high score when all claims are supported."""
        verifications = [
            create_verification(VerificationStatus.SUPPORTED),
            create_verification(VerificationStatus.SUPPORTED),
        ]

        score = await scorer.compute_score(verifications)

        assert score.overall == 1.0

    @pytest.mark.asyncio
    async def test_scoring_method_name(self, scorer: StrictScorer) -> None:
        """Test scoring method is correctly reported."""
        assert scorer.scoring_method == "strict"
