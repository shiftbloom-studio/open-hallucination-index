"""
Configuration Management
========================

Pydantic-settings based configuration for all external services.
Reads from environment variables with sensible defaults.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
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
        default="TheBloke/openinstruct-mistral-7B-AWQ",
        description="Model name/ID to use",
    )
    timeout_seconds: float = Field(default=240.0, ge=1.0)
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
    collection_name: str = Field(default="wikipedia_hybrid")
    vector_size: int = Field(
        default=384,
        description="Embedding dimension (384 for all-MiniLM-L12-v2)",
    )
    use_grpc: bool = Field(default=False)
    https: bool = Field(default=False)
    tls_ca_cert: str | None = Field(default=None)


class RedisSettings(BaseSettings):
    """Configuration for Redis semantic cache."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    enabled: bool = Field(default=True, description="Enable Redis caching")
    flush_on_startup: bool = Field(
        default=False,
        description="Flush all keys in Redis DB on startup",
    )
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
    # API key for public access and admin dashboard (optional, leave empty to disable)
    api_key: str = Field(
        default="",
        description="API key for authentication and admin access. Leave empty to disable auth.",
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
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
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
        default=2,
        ge=1,
        description="Minimum evidence pieces before early exit",
    )
    min_weighted_value: float = Field(
        default=1.5,
        ge=0.0,
        description="Minimum quality-weighted value for sufficiency",
    )
    high_confidence_threshold: int = Field(
        default=2,
        ge=1,
        description="High-confidence evidence count for early exit",
    )

    # === Timeout Settings (milliseconds) - INCREASED for MCP sources ===
    local_timeout_ms: float = Field(
        default=2000.0,
        ge=10.0,
        description="Timeout for local sources (Neo4j + Qdrant) - needs ~500ms for complex queries",
    )
    mcp_timeout_ms: float = Field(
        default=10000.0,
        ge=100.0,
        description="Timeout for MCP sources per claim - MCP can take 5-10s",
    )
    total_timeout_ms: float = Field(
        default=15000.0,
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

    # === Evidence Classification Settings ===
    classification_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for evidence classification (0.1=conservative, 0.3-0.5=balanced)",
    )
    enable_two_pass_classification: bool = Field(
        default=False,
        description="Enable two-pass classification to reduce false NEUTRAL classifications",
    )
    enable_confidence_scoring: bool = Field(
        default=False,
        description="Use granular confidence-weighted classifications (5-level scale)",
    )
    classification_batch_size: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of evidence items to classify in a single LLM call",
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


class NLISettings(BaseSettings):
    """Configuration for the Phase 2 LLM-based NLI classifier layer.

    Injected into :class:`adapters.nli_gemini.NliGeminiAdapter` via
    :func:`config.dependencies._initialize_adapters`. The adapter runs
    against a dedicated :class:`adapters.gemini.GeminiLLMAdapter` whose
    model is taken from :attr:`llm_model` (not the decomposer's model),
    so L1 decomposition and L3 NLI can evolve their model choices
    independently.
    """

    model_config = SettingsConfigDict(env_prefix="NLI_")

    llm_model: str = Field(
        default="gemini-3-pro-preview",
        description=(
            "Gemini model id for NLI classification. Default gemini-3-pro-preview;"
            " fallback to gemini-2.5-pro GA via NLI_LLM_MODEL if preview is flaky."
        ),
    )
    thinking_level: str = Field(
        default="HIGH",
        description=(
            "Gemini 3 thinking budget for NLI. Plumbed for future tuning;"
            " GeminiLLMAdapter intrinsically sets thinkingLevel=HIGH for any"
            " gemini-3* model, so this value is not consumed by the adapter today."
        ),
    )
    self_consistency_k: int = Field(
        default=1,
        ge=1,
        description=(
            "Majority-vote samples per (claim, evidence) classification."
            " K=1 disables self-consistency. Flipping to K>1 multiplies Gemini"
            " spend by K — plan §6.2 G6 gate required."
        ),
    )


class PCGSettings(BaseSettings):
    """Configuration for the Wave 3 Stream P probabilistic claim graph
    inference layer (TRW-BP primary, damped LBP fallback, Gibbs MCMC
    sanity).

    All fields read ``PCG_*`` env vars. Defaults match the Wave 3 spec
    §4.2 and the TF-plumbed Lambda env values in
    ``infra/terraform/compute/lambda.tf``. Consumed by
    :class:`adapters.pcg_belief_propagation.PCGBeliefPropagationAdapter`.
    """

    model_config = SettingsConfigDict(env_prefix="PCG_")

    max_iters: int = Field(default=200, ge=1)
    convergence_tol: float = Field(default=1e-4, gt=0.0)
    damping_factor: float = Field(default=0.8, gt=0.0, le=1.0)
    rigor_default: Literal["fast", "balanced", "maximum"] = Field(default="balanced")
    entity_overlap_threshold: int = Field(default=1, ge=0)
    gibbs_burn_in: int = Field(default=500, ge=0)
    gibbs_samples: int = Field(default=2000, ge=1)
    gibbs_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    claim_claim_max_pairs: int = Field(default=200, ge=1)


class CCNliSettings(BaseSettings):
    """Configuration for the Wave 3 Stream P claim-claim NLI dispatcher.

    OpenAI ``gpt-5.4`` with ``reasoning.effort=xhigh`` is the primary
    adapter (Decision H); Gemini 3 Pro preview via
    :class:`adapters.nli_gemini.NliGeminiAdapter` is the one-shot
    fallback on terminal primary failure.

    ``llm_model`` encodes both model name + reasoning effort via a
    suffix convention (``gpt-5.4-xhigh``); the parser at adapter-ctor
    splits it into a model ID + effort kwarg.
    """

    model_config = SettingsConfigDict(env_prefix="CC_NLI_")

    llm_provider: Literal["openai", "gemini"] = Field(
        default="openai",
        description=(
            "Primary cc-NLI provider. ``gemini`` forces fallback-only "
            "(cost-cap emergency lever)."
        ),
    )
    llm_model: str = Field(
        default="gpt-5.4-xhigh",
        description=(
            "Primary cc-NLI model spec; suffix -xhigh/-high/-medium/-low "
            "parses as reasoning.effort."
        ),
    )
    llm_fallback_model: str = Field(
        default="gemini-3-pro-preview",
        description="Fallback cc-NLI model (Gemini). Used via NliGeminiAdapter.",
    )
    openai_max_retries: int = Field(default=3, ge=1)
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "OpenAI API key for cc-NLI. Read from OHI_OPENAI_API_KEY env var "
            "(which TF plumbs from the ohi/openai-api-key Secrets Manager entry)."
        ),
    )

    def __init__(self, **kwargs: object) -> None:
        import os

        # TF injects the raw key value as OHI_OPENAI_API_KEY at runtime
        # (mirrors the Gemini LLM_API_KEY pattern in compute/lambda.tf).
        if "openai_api_key" not in kwargs and os.environ.get("OHI_OPENAI_API_KEY"):
            kwargs["openai_api_key"] = SecretStr(os.environ["OHI_OPENAI_API_KEY"])
        super().__init__(**kwargs)


class EmbeddingSettings(BaseSettings):
    """Configuration for local embedding generation."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    # Model choices:
    # - all-MiniLM-L12-v2: 384 dim, fast, good quality (default)
    # - all-mpnet-base-v2: 768 dim, higher quality, slower
    # - BAAI/bge-small-en-v1.5: 384 dim, excellent quality
    # - BAAI/bge-base-en-v1.5: 768 dim, best quality
    model_name: str = Field(
        default="all-MiniLM-L12-v2",
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
    nli: NLISettings = Field(default_factory=NLISettings)
    pcg: PCGSettings = Field(default_factory=PCGSettings)
    cc_nli: CCNliSettings = Field(default_factory=CCNliSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    # Environment
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        validation_alias=AliasChoices("OHI_ENV", "ENVIRONMENT"),
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        validation_alias=AliasChoices("OHI_LOG_LEVEL", "LOG_LEVEL"),
    )

    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_environment(cls, value: object) -> object:
        if isinstance(value, str) and value.lower() == "prod":
            return "production"
        return value


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns singleton instance, reading from environment on first call.
    """
    return Settings()
