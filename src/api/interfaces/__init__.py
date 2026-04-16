"""
OHI Interfaces - Abstract Contracts
====================================

Port interfaces for dependency injection and loose coupling.

v1 ports (verification.VerificationOracle, scoring.Scorer) have been
removed. v2 ports (NLI, Domain, PCG, Conformal, Feedback) are added in
Phase 1 Task 1.2.
"""

from interfaces.decomposition import ClaimDecomposer
from interfaces.llm import LLMMessage, LLMProvider, LLMResponse
from interfaces.mcp import (
    MCPConnectionError,
    MCPKnowledgeSource,
    MCPQueryError,
    reset_mcp_call_cache,
    set_mcp_call_cache,
)
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
