"""
Ports Layer (Hexagonal Architecture)
====================================

Abstract interfaces defining the contracts between the application core
and external systems. These are the "ports" that adapters plug into.

Primary Ports (driving):
- API endpoints drive the application

Secondary Ports (driven):
- ClaimDecomposer: Text → Claims transformation
- KnowledgeStore: Fact retrieval (graph + vector)
- LLMProvider: Language model inference
- CacheProvider: Semantic caching
- KnowledgeTracker: Provenance tracking for knowledge-track
"""

from open_hallucination_index.ports.cache import CacheProvider
from open_hallucination_index.ports.claim_decomposer import ClaimDecomposer
from open_hallucination_index.ports.knowledge_store import (
    GraphKnowledgeStore,
    KnowledgeStore,
    VectorKnowledgeStore,
)
from open_hallucination_index.ports.knowledge_tracker import KnowledgeTracker
from open_hallucination_index.ports.llm_provider import LLMProvider
from open_hallucination_index.ports.scorer import Scorer
from open_hallucination_index.ports.verification_oracle import (
    VerificationOracle,
    VerificationStrategy,
)

__all__ = [
    "CacheProvider",
    "ClaimDecomposer",
    "GraphKnowledgeStore",
    "KnowledgeStore",
    "KnowledgeTracker",
    "LLMProvider",
    "Scorer",
    "VectorKnowledgeStore",
    "VerificationOracle",
    "VerificationStrategy",
]
