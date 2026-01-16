"""
OHI Interfaces - Abstract Contracts
====================================

Port interfaces for dependency injection and loose coupling.
"""

from interfaces.cache import CacheProvider
from interfaces.decomposition import ClaimDecomposer
from interfaces.llm import LLMProvider, LLMMessage, LLMResponse
from interfaces.mcp import MCPKnowledgeSource, MCPConnectionError, MCPQueryError, reset_mcp_call_cache, set_mcp_call_cache
from interfaces.scoring import Scorer
from interfaces.stores import GraphKnowledgeStore, VectorKnowledgeStore, KnowledgeStore, GraphQuery, VectorQuery
from interfaces.tracking import KnowledgeTracker, KnowledgeTrackerError
from interfaces.verification import EvidenceTier, VerificationOracle, VerificationStrategy

__all__ = [
    # Cache
    "CacheProvider",
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
    # Scoring
    "Scorer",
    # Stores
    "GraphKnowledgeStore",
    "VectorKnowledgeStore",
    "KnowledgeStore",
    "GraphQuery",
    "VectorQuery",
    # Tracking
    "KnowledgeTracker",
    "KnowledgeTrackerError",
    # Verification
    "EvidenceTier",
    "VerificationOracle",
    "VerificationStrategy",
]
