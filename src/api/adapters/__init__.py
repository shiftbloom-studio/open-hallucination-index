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
from adapters.redis_cache import RedisCacheAdapter
from adapters.redis_trace import RedisTraceAdapter

__all__ = [
    # Graph Store
    "Neo4jGraphAdapter",
    # Vector Store
    "QdrantVectorAdapter",
    # Cache
    "RedisCacheAdapter",
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
