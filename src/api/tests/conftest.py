"""
Pytest Fixtures
===============

Shared fixtures for all test modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from interfaces.verification import VerificationStrategy
from models.entities import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
)
from models.results import (
    CitationTrace,
    ClaimVerification,
    TrustScore,
    VerificationStatus,
)

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Domain Entity Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_claim() -> Claim:
    """Create a sample claim for testing."""
    return Claim(
        id=uuid4(),
        text="Paris is the capital of France.",
        claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
        subject="Paris",
        predicate="is the capital of",
        object="France",
        confidence=0.95,
        normalized_form="paris is the capital of france",
    )


@pytest.fixture
def sample_claim_refuted() -> Claim:
    """Create a claim that should be refuted."""
    return Claim(
        id=uuid4(),
        text="Berlin is the capital of France.",
        claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
        subject="Berlin",
        predicate="is the capital of",
        object="France",
        confidence=0.9,
        normalized_form="berlin is the capital of france",
    )


@pytest.fixture
def sample_claims(sample_claim: Claim, sample_claim_refuted: Claim) -> list[Claim]:
    """Create a list of sample claims."""
    return [
        sample_claim,
        sample_claim_refuted,
        Claim(
            id=uuid4(),
            text="Water boils at 100 degrees Celsius at sea level.",
            claim_type=ClaimType.QUANTITATIVE,
            subject="Water",
            predicate="boils at",
            object="100 degrees Celsius",
            confidence=0.98,
        ),
    ]


@pytest.fixture
def sample_evidence() -> Evidence:
    """Create sample evidence for testing."""
    return Evidence(
        id=uuid4(),
        source=EvidenceSource.GRAPH_EXACT,
        content="Paris is the capital city of France.",
        similarity_score=0.92,
        structured_data={
            "subject": "Paris",
            "predicate": "capital_of",
            "object": "France",
        },
    )


@pytest.fixture
def sample_evidence_list(sample_evidence: Evidence) -> list[Evidence]:
    """Create a list of sample evidence."""
    return [
        sample_evidence,
        Evidence(
            id=uuid4(),
            source=EvidenceSource.VECTOR_SEMANTIC,
            content="Paris serves as the capital and largest city of France.",
            similarity_score=0.88,
        ),
    ]


# -----------------------------------------------------------------------------
# Verification Result Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def supported_trace(sample_claim: Claim, sample_evidence_list: list[Evidence]) -> CitationTrace:
    """Create a trace for a supported claim."""
    return CitationTrace(
        claim_id=sample_claim.id,
        status=VerificationStatus.SUPPORTED,
        reasoning="Found 2 pieces of supporting evidence.",
        supporting_evidence=sample_evidence_list,
        refuting_evidence=[],
        confidence=0.9,
        verification_strategy=VerificationStrategy.HYBRID.value,
    )


@pytest.fixture
def refuted_trace(sample_claim_refuted: Claim) -> CitationTrace:
    """Create a trace for a refuted claim."""
    return CitationTrace(
        claim_id=sample_claim_refuted.id,
        status=VerificationStatus.REFUTED,
        reasoning="Found contradicting evidence.",
        supporting_evidence=[],
        refuting_evidence=[
            Evidence(
                id=uuid4(),
                source=EvidenceSource.GRAPH_EXACT,
                content="Paris is the capital of France, not Berlin.",
                similarity_score=0.85,
            ),
        ],
        confidence=0.85,
        verification_strategy=VerificationStrategy.HYBRID.value,
    )


@pytest.fixture
def sample_verification(sample_claim: Claim, supported_trace: CitationTrace) -> ClaimVerification:
    """Create a sample verification result."""
    return ClaimVerification(
        claim=sample_claim,
        status=VerificationStatus.SUPPORTED,
        trace=supported_trace,
    )


@pytest.fixture
def sample_verifications(
    sample_claim: Claim,
    sample_claim_refuted: Claim,
    supported_trace: CitationTrace,
    refuted_trace: CitationTrace,
) -> list[ClaimVerification]:
    """Create a list of verification results."""
    return [
        ClaimVerification(
            claim=sample_claim,
            status=VerificationStatus.SUPPORTED,
            trace=supported_trace,
        ),
        ClaimVerification(
            claim=sample_claim_refuted,
            status=VerificationStatus.REFUTED,
            trace=refuted_trace,
        ),
    ]


@pytest.fixture
def sample_trust_score() -> TrustScore:
    """Create a sample trust score."""
    return TrustScore(
        overall=0.75,
        claims_total=4,
        claims_supported=2,
        claims_refuted=1,
        claims_unverifiable=1,
        confidence=0.8,
        scoring_method="weighted_average",
    )
