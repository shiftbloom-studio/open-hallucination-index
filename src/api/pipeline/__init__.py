"""
OHI Pipeline - v2 Verification Flow Stages
===========================================

The v2 pipeline: L1 decompose + retrieve → L2 domain route → L3 NLI →
L4 PCG → L5 conformal → L6 active-learning hook → L7 assembly.

v1 classes (HybridVerificationOracle, WeightedScorer) have been removed.
v2 layers land in Phase 1-4 tasks. This __init__ re-exports only the
primitives that survive the rewrite (decomposer, retrieval router,
collector, mesh, selector).
"""

from pipeline.collector import AdaptiveEvidenceCollector
from pipeline.decomposer import LLMClaimDecomposer
from pipeline.mesh import KnowledgeMeshBuilder
from pipeline.router import ClaimRouter
from pipeline.selector import SmartMCPSelector

__all__ = [
    # L1 — Decomposition
    "LLMClaimDecomposer",
    # L1 — Retrieval primitives (moved into pipeline/retrieval/ in Task 1.4)
    "ClaimRouter",
    "SmartMCPSelector",
    "AdaptiveEvidenceCollector",
    "KnowledgeMeshBuilder",
]
