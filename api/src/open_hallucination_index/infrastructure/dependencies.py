"""
Dependency Injection Container
==============================

Provides FastAPI dependency functions for injecting ports.
Wires adapters to ports based on configuration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

from open_hallucination_index.adapters.outbound.cache_redis import RedisCacheAdapter
from open_hallucination_index.adapters.outbound.embeddings_local import LocalEmbeddingAdapter
from open_hallucination_index.adapters.outbound.graph_neo4j import Neo4jGraphAdapter
from open_hallucination_index.adapters.outbound.llm_openai import OpenAILLMAdapter
from open_hallucination_index.adapters.outbound.mcp_context7 import Context7MCPAdapter
from open_hallucination_index.adapters.outbound.mcp_ohi import OHIMCPAdapter
from open_hallucination_index.adapters.outbound.mcp_wikipedia import WikipediaMCPAdapter
from open_hallucination_index.adapters.outbound.vector_qdrant import QdrantVectorAdapter
from open_hallucination_index.application.verify_text import VerifyTextUseCase
from open_hallucination_index.domain.services.claim_decomposer import LLMClaimDecomposer
from open_hallucination_index.domain.services.evidence_collector import (
    AdaptiveEvidenceCollector,
)
from open_hallucination_index.domain.services.mcp_selector import SmartMCPSelector
from open_hallucination_index.domain.services.scorer import WeightedScorer
from open_hallucination_index.domain.services.verification_oracle import (
    HybridVerificationOracle,
)
from open_hallucination_index.infrastructure.config import get_settings
from open_hallucination_index.ports.cache import CacheProvider
from open_hallucination_index.ports.claim_decomposer import ClaimDecomposer
from open_hallucination_index.ports.knowledge_store import (
    GraphKnowledgeStore,
    VectorKnowledgeStore,
)
from open_hallucination_index.ports.llm_provider import LLMProvider
from open_hallucination_index.ports.mcp_source import MCPKnowledgeSource
from open_hallucination_index.ports.scorer import Scorer
from open_hallucination_index.ports.verification_oracle import (
    VerificationOracle,
    VerificationStrategy,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Singleton holders (initialized on app startup)
# -----------------------------------------------------------------------------

_llm_provider: LLMProvider | None = None
_embedding_adapter: LocalEmbeddingAdapter | None = None
_graph_store: GraphKnowledgeStore | None = None
_vector_store: VectorKnowledgeStore | None = None
_cache_provider: CacheProvider | None = None
_claim_decomposer: ClaimDecomposer | None = None
_verification_oracle: VerificationOracle | None = None
_scorer: Scorer | None = None
_verify_use_case: VerifyTextUseCase | None = None
_mcp_sources: list[MCPKnowledgeSource] = []


# -----------------------------------------------------------------------------
# Lifecycle management
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan_manager(app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    """
    Manage application lifecycle: initialize and cleanup adapters.

    Usage in FastAPI:
        app = FastAPI(lifespan=lifespan_manager)
    """
    await _initialize_adapters()
    try:
        yield {}
    finally:
        await _cleanup_adapters()


async def _initialize_adapters() -> None:
    """
    Initialize all adapters based on configuration.

    This is where concrete adapter implementations are wired to ports.
    """
    global _llm_provider, _embedding_adapter, _graph_store, _vector_store, _cache_provider
    global _claim_decomposer, _verification_oracle, _scorer, _verify_use_case
    global _mcp_sources

    settings = get_settings()

    # Add a small random delay to stagger worker startups and avoid
    # overwhelming MCP servers with simultaneous connection attempts
    worker_pid = os.getpid()
    stagger_delay = random.uniform(0.5, 3.0)
    logger.debug(f"Worker {worker_pid}: staggering startup by {stagger_delay:.1f}s")
    await asyncio.sleep(stagger_delay)

    logger.info(f"Initializing DI container - Environment: {settings.environment}")

    # Initialize local embedding adapter (loads sentence-transformer model)
    _embedding_adapter = LocalEmbeddingAdapter(settings.embedding)
    logger.info(f"Embedding adapter initialized: {settings.embedding.model_name}")

    # Initialize LLM provider (vLLM/OpenAI-compatible)
    _llm_provider = OpenAILLMAdapter(settings.llm)
    logger.info(f"LLM provider initialized: {settings.llm.model}")

    # Initialize Neo4j graph store
    _graph_store = Neo4jGraphAdapter(settings.neo4j)
    await _graph_store.connect()
    logger.info(f"Graph store connected: {settings.neo4j.uri}")

    # Initialize Qdrant vector store with local embeddings
    _vector_store = QdrantVectorAdapter(
        settings=settings.qdrant,
        embedding_func=_embedding_adapter.generate_embedding,  # Use local embeddings
    )
    await _vector_store.connect()
    logger.info(f"Vector store connected: {settings.qdrant.host}:{settings.qdrant.port}")

    # Initialize Redis cache (optional)
    if settings.redis.enabled:
        _cache_provider = RedisCacheAdapter(settings.redis)
        await _cache_provider.connect()
        logger.info(f"Cache connected: {settings.redis.host}:{settings.redis.port}")
    else:
        _cache_provider = None
        logger.info("Cache disabled")

    # Initialize MCP sources (Wikipedia, Context7)
    _mcp_sources = []

    if settings.mcp.wikipedia_enabled:
        try:
            wikipedia_adapter = WikipediaMCPAdapter(settings.mcp)
            await wikipedia_adapter.connect()
            _mcp_sources.append(wikipedia_adapter)
            logger.info(f"Wikipedia MCP connected: {settings.mcp.wikipedia_url}")
        except Exception as e:
            logger.warning(f"Wikipedia MCP unavailable (will use fallback): {e}")

    if settings.mcp.context7_enabled:
        try:
            context7_adapter = Context7MCPAdapter(settings.mcp)
            await context7_adapter.connect()
            _mcp_sources.append(context7_adapter)
            logger.info(f"Context7 MCP connected: {settings.mcp.context7_url}")
        except Exception as e:
            logger.warning(f"Context7 MCP unavailable (will use fallback): {e}")

    if settings.mcp.ohi_enabled:
        try:
            ohi_adapter = OHIMCPAdapter(settings.mcp)
            await ohi_adapter.connect()
            _mcp_sources.append(ohi_adapter)
            logger.info(f"OHI MCP connected: {settings.mcp.ohi_url}")
        except Exception as e:
            logger.warning(f"OHI MCP unavailable (will use fallback): {e}")

    # Initialize domain services
    _claim_decomposer = LLMClaimDecomposer(
        llm_provider=_llm_provider,
        max_claims=settings.verification.max_claims_per_request,
    )
    logger.info("Claim decomposer initialized")

    # Determine verification strategy from settings
    strategy_map = {
        "hybrid": VerificationStrategy.HYBRID,
        "graph_exact": VerificationStrategy.GRAPH_EXACT,
        "vector_semantic": VerificationStrategy.VECTOR_SEMANTIC,
        "cascading": VerificationStrategy.CASCADING,
        "mcp_enhanced": VerificationStrategy.MCP_ENHANCED,
        "adaptive": VerificationStrategy.ADAPTIVE,
    }
    strategy = strategy_map.get(
        settings.verification.default_strategy.lower(),
        VerificationStrategy.ADAPTIVE,  # Default to ADAPTIVE
    )

    # Initialize SmartMCPSelector for intelligent source selection
    mcp_selector = SmartMCPSelector(
        mcp_sources=_mcp_sources,
        max_sources_per_claim=settings.verification.max_mcp_sources_per_claim,
        min_relevance_threshold=settings.verification.min_source_relevance,
    )
    logger.info("SmartMCPSelector initialized")

    # Initialize AdaptiveEvidenceCollector for tiered collection
    evidence_collector = AdaptiveEvidenceCollector(
        graph_store=_graph_store,
        vector_store=_vector_store,
        mcp_selector=mcp_selector,
        min_evidence_count=settings.verification.min_evidence_count,
        min_weighted_value=settings.verification.min_weighted_value,
        high_confidence_threshold=settings.verification.high_confidence_threshold,
        local_timeout_ms=settings.verification.local_timeout_ms,
        mcp_timeout_ms=settings.verification.mcp_timeout_ms,
        total_timeout_ms=settings.verification.total_timeout_ms,
        enable_background_completion=settings.verification.enable_background_completion,
    )

    # Add persist callbacks for dual-write to Neo4j and Qdrant
    async def persist_to_graph(evidence_list: list) -> None:
        if _graph_store is not None:
            for ev in evidence_list:
                if hasattr(_graph_store, "persist_external_evidence"):
                    await _graph_store.persist_external_evidence(ev)

    async def persist_to_vector(evidence_list: list) -> None:
        if _vector_store is not None and hasattr(_vector_store, "persist_external_evidence"):
            await _vector_store.persist_external_evidence(evidence_list)

    if settings.verification.persist_mcp_evidence:
        evidence_collector.add_persist_callback(persist_to_graph)
    if settings.verification.persist_to_vector:
        evidence_collector.add_persist_callback(persist_to_vector)

    logger.info(
        f"AdaptiveEvidenceCollector initialized: "
        f"min_evidence={settings.verification.min_evidence_count}, "
        f"local_timeout={settings.verification.local_timeout_ms}ms, "
        f"mcp_timeout={settings.verification.mcp_timeout_ms}ms"
    )

    _verification_oracle = HybridVerificationOracle(
        graph_store=_graph_store,
        vector_store=_vector_store,
        mcp_sources=_mcp_sources,
        default_strategy=strategy,
        persist_mcp_evidence=settings.verification.persist_mcp_evidence,
        persist_to_vector=settings.verification.persist_to_vector,
        evidence_collector=evidence_collector,
        mcp_selector=mcp_selector,
    )
    logger.info(
        "Verification oracle initialized: strategy=%s, mcp_sources=%s",
        strategy.value,
        len(_mcp_sources),
    )

    _scorer = WeightedScorer()
    logger.info("Scorer initialized")

    # Initialize the main use case
    _verify_use_case = VerifyTextUseCase(
        decomposer=_claim_decomposer,
        oracle=_verification_oracle,
        scorer=_scorer,
        cache=_cache_provider,
    )
    logger.info("VerifyTextUseCase initialized - DI container ready")


async def _cleanup_adapters() -> None:
    """
    Cleanup all adapter connections on shutdown.

    Uses asyncio.shield() to protect cleanup operations from task cancellation.
    Each disconnect is wrapped in try/except to ensure all adapters get cleaned up.
    """
    import asyncio

    global _llm_provider, _graph_store, _vector_store, _cache_provider, _mcp_sources

    logger.info("Starting adapter cleanup...")

    # Disconnect MCP sources first (they may have session pools)
    for source in _mcp_sources:
        try:
            # Shield the disconnect from cancellation
            await asyncio.shield(asyncio.wait_for(source.disconnect(), timeout=5.0))
            logger.debug(f"Disconnected MCP source: {source.source_name}")
        except TimeoutError:
            logger.warning(f"MCP source disconnect timed out: {source.source_name}")
        except asyncio.CancelledError:
            logger.warning(f"MCP source disconnect cancelled: {source.source_name}")
        except Exception as e:
            logger.warning(f"MCP source disconnect failed: {e}")
    _mcp_sources = []

    # Disconnect cache provider
    if _cache_provider is not None:
        try:
            await asyncio.shield(asyncio.wait_for(_cache_provider.disconnect(), timeout=5.0))
            logger.debug("Disconnected cache provider")
        except TimeoutError:
            logger.warning("Cache provider disconnect timed out")
        except asyncio.CancelledError:
            logger.warning("Cache provider disconnect cancelled")
        except Exception as e:
            logger.warning(f"Cache provider disconnect failed: {e}")
        _cache_provider = None

    # Disconnect vector store
    if _vector_store is not None:
        try:
            await asyncio.shield(asyncio.wait_for(_vector_store.disconnect(), timeout=5.0))
            logger.debug("Disconnected vector store")
        except TimeoutError:
            logger.warning("Vector store disconnect timed out")
        except asyncio.CancelledError:
            logger.warning("Vector store disconnect cancelled")
        except Exception as e:
            logger.warning(f"Vector store disconnect failed: {e}")
        _vector_store = None

    # Disconnect graph store (Neo4j) - most likely to have cancellation issues
    if _graph_store is not None:
        try:
            # Use shield to protect from cancellation during shutdown
            await asyncio.shield(asyncio.wait_for(_graph_store.disconnect(), timeout=10.0))
            logger.debug("Disconnected graph store")
        except TimeoutError:
            logger.warning("Graph store disconnect timed out")
        except asyncio.CancelledError:
            logger.warning("Graph store disconnect cancelled (graceful shutdown)")
        except Exception as e:
            logger.warning(f"Graph store disconnect failed: {e}")
        _graph_store = None

    logger.info("Adapter cleanup complete")


# -----------------------------------------------------------------------------
# FastAPI Dependency providers
# -----------------------------------------------------------------------------


async def get_llm_provider() -> LLMProvider:
    """Dependency: Get LLM provider instance."""
    if _llm_provider is None:
        raise RuntimeError("LLM provider not initialized. Check adapter configuration.")
    return _llm_provider


async def get_graph_store() -> GraphKnowledgeStore:
    """Dependency: Get graph knowledge store instance."""
    if _graph_store is None:
        raise RuntimeError("Graph store not initialized. Check adapter configuration.")
    return _graph_store


async def get_vector_store() -> VectorKnowledgeStore:
    """Dependency: Get vector knowledge store instance."""
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized. Check adapter configuration.")
    return _vector_store


async def get_cache_provider() -> CacheProvider | None:
    """Dependency: Get cache provider instance (may be None if disabled)."""
    return _cache_provider


async def get_claim_decomposer() -> ClaimDecomposer:
    """Dependency: Get claim decomposer instance."""
    if _claim_decomposer is None:
        raise RuntimeError("Claim decomposer not initialized. Check adapter configuration.")
    return _claim_decomposer


async def get_verification_oracle() -> VerificationOracle:
    """Dependency: Get verification oracle instance."""
    if _verification_oracle is None:
        raise RuntimeError("Verification oracle not initialized. Check adapter configuration.")
    return _verification_oracle


async def get_scorer() -> Scorer:
    """Dependency: Get scorer instance."""
    if _scorer is None:
        raise RuntimeError("Scorer not initialized. Check adapter configuration.")
    return _scorer


async def get_verify_use_case() -> VerifyTextUseCase:
    """Dependency: Get the main verification use-case instance."""
    if _verify_use_case is None:
        raise RuntimeError("VerifyTextUseCase not initialized. Check adapter configuration.")
    return _verify_use_case
