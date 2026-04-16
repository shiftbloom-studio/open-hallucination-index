"""
OHI Models - Domain Entities
============================

Core data structures for the v2 verification pipeline.

v1 result models (CitationTrace, ClaimVerification, TrustScore,
VerificationResult, VerificationStatus, EvidenceClassification) have been
removed. v2 equivalents live in `models.results` (ClaimVerdict,
DocumentVerdict, ClaimEdge, EdgeType).

The `models.track` module is kept for now — it powers the in-memory
KnowledgeMesh used by L1 retrieval (pipeline.mesh). Its EdgeType enum is
distinct from the v2 PCG EdgeType in `models.results`; callers importing
from this package get the v2 one by default. Import directly from
`models.track` when you need the track/mesh-specific version.
"""

from models.entities import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
)
from models.nli import NLIDistribution
from models.pcg import PosteriorBelief
from models.results import (
    ClaimEdge,
    ClaimVerdict,
    DocumentVerdict,
    EdgeType,
)

__all__ = [
    # Entities
    "Claim",
    "ClaimType",
    "Evidence",
    "EvidenceSource",
    # v2 results
    "ClaimEdge",
    "ClaimVerdict",
    "DocumentVerdict",
    "EdgeType",
    # v2 NLI
    "NLIDistribution",
    # v2 PCG
    "PosteriorBelief",
]
