"""L1 retrieval package — moved from pipeline/{router,collector,selector,mesh}.py.

The four retrieval primitives (claim routing, adaptive evidence collection,
MCP source selection, knowledge-mesh building) live here. Task 1.4 move.

Added in this package:
  - source_credibility.py — per-source credibility priors + temporal decay
                            + provenance fingerprinting. Used by the
                            collector to stamp Evidence objects with
                            weight metadata the L4 PCG will consume.
"""

from __future__ import annotations

from pipeline.retrieval.collector import AdaptiveEvidenceCollector
from pipeline.retrieval.mesh import KnowledgeMeshBuilder
from pipeline.retrieval.router import ClaimRouter
from pipeline.retrieval.selector import SmartMCPSelector
from pipeline.retrieval.source_credibility import (
    DEFAULT_PRIORS,
    credibility_for,
    fingerprint,
    temporal_decay,
)

__all__ = [
    "AdaptiveEvidenceCollector",
    "ClaimRouter",
    "KnowledgeMeshBuilder",
    "SmartMCPSelector",
    "DEFAULT_PRIORS",
    "credibility_for",
    "fingerprint",
    "temporal_decay",
]
