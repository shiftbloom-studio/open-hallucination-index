"""
OHI Pipeline - v2 Verification Flow Stages
===========================================

The v2 pipeline: L1 decompose + retrieve → L2 domain route → L3 NLI →
L4 PCG → L5 conformal → L6 active-learning hook → L7 assembly.

Task 1.4 moved the 4 retrieval modules (router, collector, selector,
mesh) into `pipeline.retrieval/`, and added `source_credibility.py`
there. L2-L7 land in subsequent Phase 1/2/3/4 tasks.
"""

from pipeline.decomposer import LLMClaimDecomposer
from pipeline.retrieval import (
    AdaptiveEvidenceCollector,
    ClaimRouter,
    KnowledgeMeshBuilder,
    SmartMCPSelector,
)

__all__ = [
    # L1 — Decomposition
    "LLMClaimDecomposer",
    # L1 — Retrieval primitives (live under pipeline.retrieval)
    "ClaimRouter",
    "SmartMCPSelector",
    "AdaptiveEvidenceCollector",
    "KnowledgeMeshBuilder",
]
