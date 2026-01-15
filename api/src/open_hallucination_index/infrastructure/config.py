"""
Configuration Management
========================

Pydantic-settings based configuration for all external services.
Reads from environment variables with sensible defaults.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Configuration for LLM inference engine (vLLM/OpenAI-compatible)."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for OpenAI-compatible API",
    )
    api_key: SecretStr = Field(
        default=SecretStr("no-key-required"),
        description="API key (some OpenAI-compatible servers require a value)",
    )
    model: str = Field(
        default="Qwen/Qwen2.5-14B-Instruct-AWQ",
        description="Model name/ID to use",
    )
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    # OpenAI API key for embeddings (separate from vLLM)
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key for embeddings (reads OPENAI_API_KEY env var)",
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        # Also read OPENAI_API_KEY without prefix
        extra="ignore",
    )

    def __init__(self, **kwargs: object) -> None:
        import os

        # Allow OPENAI_API_KEY to be read directly
        if "openai_api_key" not in kwargs and os.environ.get("OPENAI_API_KEY"):
            kwargs["openai_api_key"] = SecretStr(os.environ["OPENAI_API_KEY"])
        super().__init__(**kwargs)


class Neo4jSettings(BaseSettings):
    """Configuration for Neo4j graph database."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_")

    uri: str = Field(
        default="bolt://localhost:7687",
        description="Bolt URI for Neo4j connection",
    )
    http_port: int = Field(default=7474)
    bolt_port: int = Field(default=7687)
    username: str = Field(default="neo4j")
    password: SecretStr = Field(default=SecretStr("password"))
    database: str = Field(default="neo4j")
    max_connection_pool_size: int = Field(default=50, ge=1)


class QdrantSettings(BaseSettings):
    """Configuration for Qdrant vector database."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)
    api_key: SecretStr | None = Field(default=None)
    collection_name: str = Field(default="knowledge_base")
    vector_size: int = Field(
        default=384,
        description="Embedding dimension (384 for all-MiniLM-L6-v2)",
    )
    use_grpc: bool = Field(default=False)


class RedisSettings(BaseSettings):
    """Configuration for Redis semantic cache."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    enabled: bool = Field(default=True, description="Enable Redis caching")
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    socket_path: str | None = Field(default=None, description="Path to Unix socket")
    password: SecretStr | None = Field(default=None)
    db: int = Field(default=0, ge=0)
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Default TTL for cached results",
    )
    claim_cache_ttl_seconds: int = Field(
        default=2592000,
        description="Default TTL for claim-level cache entries (long-lived)",
    )
    max_connections: int = Field(default=10, ge=1)


class APISettings(BaseSettings):
    """Configuration for the FastAPI application."""

    model_config = SettingsConfigDict(env_prefix="API_")

    title: str = Field(default="Open Hallucination Index API")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    # API key for public access (optional, leave empty to disable)
    api_key: str = Field(
        default="",
        description="API key for authentication. Leave empty to disable auth.",
    )
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1)


class VerificationSettings(BaseSettings):
    """Configuration for verification pipeline behavior."""

    model_config = SettingsConfigDict(env_prefix="VERIFY_")

    default_strategy: Literal[
        "graph_exact", "vector_semantic", "hybrid", "cascading", "mcp_enhanced", "adaptive"
    ] = Field(
        default="adaptive",
        description="Verification strategy. 'adaptive' uses intelligent tiered collection.",
    )
    max_claims_per_request: int = Field(default=100, ge=1)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    graph_max_hops: int = Field(default=2, ge=1)
    enable_caching: bool = Field(default=True)
    persist_mcp_evidence: bool = Field(
        default=True,
        description="Persist evidence from MCP sources to Neo4j graph",
    )
    persist_to_vector: bool = Field(
        default=True,
        description="Also persist MCP evidence to Qdrant for semantic fallback",
    )

    # === Adaptive Evidence Collection Settings ===
    min_evidence_count: int = Field(
        default=3,
        ge=1,
        description="Minimum evidence pieces before early exit",
    )
    min_weighted_value: float = Field(
        default=2.0,
        ge=0.0,
        description="Minimum quality-weighted value for sufficiency",
    )
    high_confidence_threshold: int = Field(
        default=2,
        ge=1,
        description="High-confidence evidence count for early exit",
    )

    # === Timeout Settings (milliseconds) ===
    local_timeout_ms: float = Field(
        default=50.0,
        ge=10.0,
        description="Timeout for local sources (Neo4j + Qdrant)",
    )
    mcp_timeout_ms: float = Field(
        default=500.0,
        ge=100.0,
        description="Timeout for MCP sources per claim",
    )
    total_timeout_ms: float = Field(
        default=2000.0,
        ge=500.0,
        description="Total timeout for all evidence collection",
    )

    # === Source Selection ===
    max_mcp_sources_per_claim: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum MCP sources to query per claim",
    )
    min_source_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to include MCP source",
    )

    # === Background Completion ===
    enable_background_completion: bool = Field(
        default=True,
        description="Allow slow MCP tasks to complete in background for caching",
    )


class MCPSettings(BaseSettings):
    """Configuration for MCP knowledge sources (Wikipedia, Context7, OHI)."""

    model_config = SettingsConfigDict(env_prefix="MCP_")

    # Wikipedia MCP
    wikipedia_enabled: bool = Field(default=True)
    wikipedia_url: str = Field(
        default="http://ohi-mcp-server:8080",
        description="URL of MCP server providing Wikipedia tools",
    )

    # Context7 MCP
    context7_enabled: bool = Field(default=True)
    context7_url: str = Field(
        default="http://ohi-mcp-server:8080",
        description="URL of MCP server providing Context7 tools",
    )
    context7_api_key: str = Field(
        default="",
        description="Context7 API key for higher rate limits",
    )

    # OHI Unified MCP Server (13+ knowledge sources)
    ohi_enabled: bool = Field(default=True)
    ohi_url: str = Field(
        default="http://ohi-mcp-server:8080",
        description="URL of unified OHI MCP server",
    )

    def __init__(self, **kwargs: object) -> None:
        import os

        # Read CONTEXT7_API_KEY from environment
        if "context7_api_key" not in kwargs and os.environ.get("CONTEXT7_API_KEY"):
            kwargs["context7_api_key"] = os.environ["CONTEXT7_API_KEY"]
        super().__init__(**kwargs)


class EmbeddingSettings(BaseSettings):
    """Configuration for local embedding generation."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    # Model choices:
    # - all-MiniLM-L6-v2: 384 dim, fast, good quality (default)
    # - all-mpnet-base-v2: 768 dim, higher quality, slower
    # - BAAI/bge-small-en-v1.5: 384 dim, excellent quality
    # - BAAI/bge-base-en-v1.5: 768 dim, best quality
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name",
    )
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = Field(default=True, description="Normalize embeddings to unit length")


class Settings(BaseSettings):
    """
    Root configuration aggregating all service settings.

    Usage:
        settings = get_settings()
        neo4j_uri = settings.neo4j.uri
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings (manually instantiated due to pydantic-settings behavior)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    # Environment
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns singleton instance, reading from environment on first call.
    """
    return Settings()
