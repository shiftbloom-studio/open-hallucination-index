"""
Tests for Domain Results
========================
"""

from uuid import uuid4

from open_hallucination_index.domain.entities import Claim, ClaimType
from open_hallucination_index.domain.results import (
    CitationTrace,
    ClaimVerification,
    TrustScore,
    VerificationResult,
    VerificationStatus,
)


class TestVerificationStatus:
    """Tests for the VerificationStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all expected statuses are defined."""
        statuses = {s.name for s in VerificationStatus}
        expected = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "REFUTED",
            "UNCERTAIN",
            "UNVERIFIABLE",
        }
        assert statuses == expected

    def test_status_values(self) -> None:
        """Test status string values."""
        assert VerificationStatus.SUPPORTED.value == "supported"
        assert VerificationStatus.REFUTED.value == "refuted"


class TestTrustScore:
    """Tests for the TrustScore result."""

    def test_trust_score_creation(self) -> None:
        """Test creating a trust score."""
        score = TrustScore(
            overall=0.85,
            claims_total=10,
            claims_supported=7,
            claims_refuted=1,
            claims_unverifiable=2,
            confidence=0.9,
            scoring_method="weighted_average",
        )

        assert score.overall == 0.85
        assert score.claims_total == 10
        assert score.claims_supported == 7
        assert score.claims_refuted == 1
        assert score.claims_unverifiable == 2
        assert score.confidence == 0.9

    def test_trust_score_bounds(self) -> None:
        """Test trust score must be in [0, 1]."""
        # Valid scores
        score_low = TrustScore(
            overall=0.0,
            claims_total=1,
            claims_supported=0,
            claims_refuted=1,
            claims_unverifiable=0,
            confidence=1.0,
            scoring_method="test",
        )
        assert score_low.overall == 0.0

        score_high = TrustScore(
            overall=1.0,
            claims_total=1,
            claims_supported=1,
            claims_refuted=0,
            claims_unverifiable=0,
            confidence=1.0,
            scoring_method="test",
        )
        assert score_high.overall == 1.0


class TestCitationTrace:
    """Tests for the CitationTrace result."""

    def test_citation_trace_creation(self) -> None:
        """Test creating a citation trace."""
        claim_id = uuid4()
        trace = CitationTrace(
            claim_id=claim_id,
            status=VerificationStatus.SUPPORTED,
            reasoning="Found matching evidence.",
            supporting_evidence=[],
            refuting_evidence=[],
            confidence=0.85,
            verification_strategy="hybrid",
        )

        assert trace.claim_id == claim_id
        assert trace.status == VerificationStatus.SUPPORTED
        assert trace.reasoning == "Found matching evidence."
        assert trace.confidence == 0.85


class TestClaimVerification:
    """Tests for the ClaimVerification result."""

    def test_claim_verification(self) -> None:
        """Test creating a claim verification."""
        claim = Claim(
            id=uuid4(),
            text="Test claim",
            claim_type=ClaimType.UNCLASSIFIED,
        )

        trace = CitationTrace(
            claim_id=claim.id,
            status=VerificationStatus.SUPPORTED,
            reasoning="Test",
            supporting_evidence=[],
            refuting_evidence=[],
            confidence=0.8,
            verification_strategy="hybrid",
        )

        verification = ClaimVerification(
            claim=claim,
            status=VerificationStatus.SUPPORTED,
            trace=trace,
            score_contribution=0.8,
        )

        assert verification.claim == claim
        assert verification.status == VerificationStatus.SUPPORTED
        assert verification.trace == trace


class TestVerificationResult:
    """Tests for the overall VerificationResult."""

    def test_verification_result(self) -> None:
        """Test creating a full verification result."""
        result = VerificationResult(
            input_hash="abc123",
            input_length=31,
            trust_score=TrustScore(
                overall=1.0,
                claims_total=0,
                claims_supported=0,
                claims_refuted=0,
                claims_unverifiable=0,
                confidence=0.0,
                scoring_method="weighted_average",
            ),
            claim_verifications=[],
            summary="No claims extracted.",
            processing_time_ms=10.0,
            cached=False,
        )

        assert result.input_hash == "abc123"
        assert result.summary == "No claims extracted."
