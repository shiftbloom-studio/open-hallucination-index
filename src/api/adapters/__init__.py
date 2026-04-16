"""
OHI Adapters - Infrastructure Integrations
===========================================

Concrete implementations connecting to external services.
"""

from adapters.embeddings import LocalEmbeddingAdapter
from adapters.mcp_client import HTTPMCPAdapter
from adapters.mcp_ohi import OHIMCPAdapter, TargetedOHISource
from adapters.neo4j import Neo4jGraphAdapter
from adapters.openai import OpenAILLMAdapter
from adapters.qdrant import QdrantVectorAdapter
from adapters.redis_trace import RedisTraceAdapter

# RedisCacheAdapter (v1) has been removed. v2 caches DocumentVerdict by
# sha256(text+options) — wired in Phase 1 Task 1.8/1.10.

__all__ = [
    # Graph Store
    "Neo4jGraphAdapter",
    # Vector Store
    "QdrantVectorAdapter",
    # Trace
    "RedisTraceAdapter",
    # LLM
    "OpenAILLMAdapter",
    # Embeddings
    "LocalEmbeddingAdapter",
    # MCP
    "HTTPMCPAdapter",
    "OHIMCPAdapter",
    "TargetedOHISource",
]
