"""
Domain Layer
============

Core business entities and value objects.
These are persistence-agnostic and contain no infrastructure dependencies.
"""

from open_hallucination_index.domain.entities import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
)
from open_hallucination_index.domain.results import (
    CitationTrace,
    ClaimVerification,
    TrustScore,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    # Entities
    "Claim",
    "ClaimType",
    "Evidence",
    "EvidenceSource",
    # Results
    "CitationTrace",
    "ClaimVerification",
    "TrustScore",
    "VerificationResult",
    "VerificationStatus",
]
