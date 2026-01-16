"""
OHI Models - Domain Entities
============================

Core data structures for verification: claims, evidence, trust scores.
"""

from models.entities import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
)
from models.results import (
    CitationTrace,
    ClaimVerification,
    EvidenceClassification,
    TrustScore,
    VerificationResult,
    VerificationStatus,
)
from models.track import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeMesh,
    KnowledgeNode,
    KnowledgeTrackResult,
    MCPSourceInfo,
    NodeType,
    SourceReference,
    TraceData,
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
    "EvidenceClassification",
    "TrustScore",
    "VerificationResult",
    "VerificationStatus",
    # Track
    "EdgeType",
    "KnowledgeEdge",
    "KnowledgeMesh",
    "KnowledgeNode",
    "KnowledgeTrackResult",
    "MCPSourceInfo",
    "NodeType",
    "SourceReference",
    "TraceData",
]
