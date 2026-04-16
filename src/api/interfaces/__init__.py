"""
OHI Interfaces - Abstract Contracts
====================================

Port interfaces for dependency injection and loose coupling.

v1 ports (verification.VerificationOracle, scoring.Scorer) have been
removed. v2 ports (NLI, Domain, PCG, Conformal, Feedback) are added in
Phase 1 Task 1.2.
"""

from interfaces.conformal import CalibratedVerdict, ConformalCalibrator
from interfaces.decomposition import ClaimDecomposer
from interfaces.domain import DomainAdapter, DomainRouter
from interfaces.feedback import FeedbackStore
from interfaces.llm import LLMMessage, LLMProvider, LLMResponse
from interfaces.mcp import (
    MCPConnectionError,
    MCPKnowledgeSource,
    MCPQueryError,
    reset_mcp_call_cache,
    set_mcp_call_cache,
)
from interfaces.nli import NLIService
from interfaces.pcg import PCGInferenceService
from interfaces.stores import (
    GraphKnowledgeStore,
    GraphQuery,
    KnowledgeStore,
    VectorKnowledgeStore,
    VectorQuery,
)
from interfaces.tracking import KnowledgeTracker, KnowledgeTrackerError

__all__ = [
    # Decomposition
    "ClaimDecomposer",
    # L2 Domain routing
    "DomainAdapter",
    "DomainRouter",
    # L3 NLI
    "NLIService",
    # L4 PCG
    "PCGInferenceService",
    # L5 Conformal
    "ConformalCalibrator",
    "CalibratedVerdict",
    # L6 Feedback
    "FeedbackStore",
    # LLM
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    # MCP
    "MCPKnowledgeSource",
    "MCPConnectionError",
    "MCPQueryError",
    "reset_mcp_call_cache",
    "set_mcp_call_cache",
    # Stores
    "GraphKnowledgeStore",
    "VectorKnowledgeStore",
    "KnowledgeStore",
    "GraphQuery",
    "VectorQuery",
    # Tracking
    "KnowledgeTracker",
    "KnowledgeTrackerError",
]
