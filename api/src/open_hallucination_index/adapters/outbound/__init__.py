"""Outbound adapters for external services."""

from open_hallucination_index.adapters.outbound.cache_redis import RedisCacheAdapter
from open_hallucination_index.adapters.outbound.graph_neo4j import Neo4jGraphAdapter
from open_hallucination_index.adapters.outbound.llm_openai import OpenAILLMAdapter
from open_hallucination_index.adapters.outbound.mcp_http_base import HTTPMCPAdapter
from open_hallucination_index.adapters.outbound.mcp_ohi import OHIMCPAdapter
from open_hallucination_index.adapters.outbound.vector_qdrant import QdrantVectorAdapter

__all__ = [
    # Core adapters
    "Neo4jGraphAdapter",
    "OpenAILLMAdapter",
    "QdrantVectorAdapter",
    "RedisCacheAdapter",
    # MCP adapters (HTTP-based, recommended)
    "HTTPMCPAdapter",
    "OHIMCPAdapter",
]
