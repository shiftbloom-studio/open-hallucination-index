"""
Dependency Injection Container
==============================

Wires infrastructure adapters to ports at lifespan boundaries and exposes
FastAPI dependency functions for handlers.

**This is the v2 skeleton.** v1 wiring (HybridVerificationOracle,
WeightedScorer, VerifyTextUseCase, KnowledgeTrackService,
RedisCacheAdapter) has been removed. v2 pipeline wiring (L1-L7) lands in
Phase 1 Task 1.7; v2 cache + feedback store wiring in Tasks 1.10 / 4.1.
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

from adapters.embeddings import LocalEmbeddingAdapter
from adapters.gemini import GeminiLLMAdapter
from adapters.mcp_ohi import OHIMCPAdapter, TargetedOHISource
from adapters.mcp_sources.mediawiki import MediaWikiAdapter
from adapters.neo4j import Neo4jGraphAdapter
from adapters.nli_gemini import NliGeminiAdapter
from adapters.openai import OpenAILLMAdapter
from adapters.qdrant import QdrantVectorAdapter
from adapters.redis_trace import RedisTraceAdapter
from adapters.verdict_store_memory import InMemoryVerdictStore
from config.settings import get_settings
from interfaces.decomposition import ClaimDecomposer
from interfaces.llm import LLMProvider
from interfaces.mcp import MCPKnowledgeSource
from interfaces.nli import NliAdapter
from interfaces.stores import (
    GraphKnowledgeStore,
    VectorKnowledgeStore,
)
from pipeline.conformal.calibration_store import InMemoryCalibrationStore
from pipeline.conformal.split_conformal import SplitConformalCalibrator
from pipeline.decomposer import LLMClaimDecomposer
from pipeline.pipeline import Pipeline
from pipeline.retrieval import (
    AdaptiveEvidenceCollector,
    KnowledgeMeshBuilder,
    SmartMCPSelector,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Singleton holders (initialized on app startup)
# -----------------------------------------------------------------------------

_llm_provider: LLMProvider | None = None
_nli_adapter: NliAdapter | None = None
_embedding_adapter: LocalEmbeddingAdapter | None = None
_graph_store: GraphKnowledgeStore | None = None
_vector_store: VectorKnowledgeStore | None = None
_trace_store: RedisTraceAdapter | None = None
_claim_decomposer: ClaimDecomposer | None = None
_mcp_sources: list[MCPKnowledgeSource] = []
_evidence_collector: AdaptiveEvidenceCollector | None = None
_mcp_selector: SmartMCPSelector | None = None
_mesh_builder: KnowledgeMeshBuilder | None = None
_pipeline: Pipeline | None = None
_verdict_store: VerdictStore | None = None

# TODO(Task 1.10): v2 DocumentVerdict cache (Redis + in-process LRU).
# TODO(Task 4.1): v2 Postgres feedback store.

_logging_configured = False


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan_manager(app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    """Initialize infrastructure adapters on startup; close on shutdown."""
    _configure_logging()
    await _initialize_adapters()
    try:
        yield {}
    finally:
        await _cleanup_adapters()


def _configure_logging() -> None:
    """Configure structured logging for the worker process (idempotent)."""
    global _logging_configured
    if _logging_configured:
        return

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    from config.logging import HealthLiveAccessFilter

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(HealthLiveAccessFilter(min_interval_seconds=120.0))
    access_logger.setLevel(logging.WARNING)

    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("ohi").setLevel(logging.INFO)
    logging.getLogger("ohi.audit").setLevel(logging.INFO)

    _logging_configured = True
    logging.getLogger(__name__).info(f"Logging configured for worker process {os.getpid()}")


async def _initialize_adapters() -> None:
    """Wire infrastructure adapters to ports."""
    global _llm_provider, _nli_adapter, _embedding_adapter, _graph_store, _vector_store
    global _trace_store, _claim_decomposer, _mcp_sources
    global _evidence_collector, _mcp_selector, _mesh_builder
    global _pipeline, _verdict_store

    settings = get_settings()

    # Stagger worker startups so MCP / vector/ graph services aren't hit
    # simultaneously with N connection bursts.
    if settings.api.workers > 1:
        await asyncio.sleep(random.uniform(0.5, 2.0))

    logger.info(f"Initializing DI container - Environment: {settings.environment}")

    # Embeddings (sentence-transformers, in-process)
    _embedding_adapter = LocalEmbeddingAdapter(settings.embedding)
    logger.info(f"Embedding adapter: {settings.embedding.model_name}")

    # LLM — native Gemini adapter. OpenAILLMAdapter is kept in-tree as a
    # fallback for non-Gemini deployments; select via env var LLM_BACKEND
    # ("gemini" | "openai"; default gemini since prod targets Gemini).
    llm_backend = os.environ.get("LLM_BACKEND", "gemini").lower()
    if llm_backend == "openai":
        _llm_provider = OpenAILLMAdapter(settings.llm)
    else:
        _llm_provider = GeminiLLMAdapter(settings.llm)
    logger.info(
        f"LLM provider: {settings.llm.model} (backend={llm_backend})"
    )

    # Neo4j
    _graph_store = Neo4jGraphAdapter(settings.neo4j)
    await _graph_store.connect()
    logger.info(f"Graph store: {settings.neo4j.uri}")

    # Qdrant
    _vector_store = QdrantVectorAdapter(
        settings=settings.qdrant,
        embedding_func=_embedding_adapter.generate_embedding,
    )
    await _vector_store.connect()
    logger.info(f"Vector store: {settings.qdrant.host}:{settings.qdrant.port}")

    # Redis trace (optional — used by knowledge-track; the v1 verdict cache
    # is gone and the v2 cache lives elsewhere).
    if settings.redis.enabled:
        _trace_store = RedisTraceAdapter(settings.redis)
        await _trace_store.connect()
        logger.info("Trace store connected")
    else:
        _trace_store = None
        logger.info("Redis disabled — trace store unavailable")

    # L1 decomposer
    _claim_decomposer = LLMClaimDecomposer(llm_provider=_llm_provider)

    # L3 NLI — Phase 2 LLM-based 3-way classifier (Stream B port, wired by
    # Stream D1). Always the native Gemini adapter regardless of LLM_BACKEND
    # because NLI depends on safetySettings=BLOCK_NONE + thinkingLevel=HIGH,
    # which the OpenAI-compat shim strips. We use model_copy to reuse the
    # decomposer's LLMSettings (same api_key, timeout, etc.) while swapping
    # the model id to settings.nli.llm_model — this lets L1 and L3 evolve
    # their model choices independently without duplicating all the other
    # LLM_* env vars.
    _nli_llm_settings = settings.llm.model_copy(
        update={"model": settings.nli.llm_model}
    )
    _nli_llm_provider = GeminiLLMAdapter(_nli_llm_settings)
    _nli_adapter = NliGeminiAdapter(
        llm=_nli_llm_provider,
        self_consistency_k=settings.nli.self_consistency_k,
        max_retries=3,
    )
    logger.info(
        "NLI adapter: %s (self_consistency_k=%d)",
        settings.nli.llm_model,
        settings.nli.self_consistency_k,
    )

    # MCP sources (dynamic; settings drives which ones)
    _mcp_sources = _build_mcp_sources(settings)
    # Per-source try/except: one bad endpoint must not abort lifespan. The
    # collector's `if source.is_available` gate skips any source whose
    # connect() raised, so failures here are tolerable and logged.
    for _src in _mcp_sources:
        try:
            await _src.connect()
            logger.info(
                f"MCP source connected: {_src.source_name} "
                f"(available={_src.is_available})"
            )
        except Exception as e:
            logger.warning(
                f"MCP source {getattr(_src, 'source_name', type(_src).__name__)} "
                f"connect failed — skipping: {e}"
            )
    _mcp_selector = SmartMCPSelector(_mcp_sources) if _mcp_sources else None

    # Adaptive collector (reused by v2 L1 retrieval layer)
    _evidence_collector = AdaptiveEvidenceCollector(
        graph_store=_graph_store,
        vector_store=_vector_store,
    )

    # Knowledge mesh builder (used by L1 retrieval). trace_store is optional
    # — None when Redis is disabled; build_mesh() falls back to live queries.
    _mesh_builder = KnowledgeMeshBuilder(
        trace_store=_trace_store,
        graph_store=_graph_store,
        vector_store=_vector_store,
    )

    # v2 Pipeline orchestrator — Phase 1 placeholder layers for L2/L3/L4/L6
    # are active; concrete adapters swap in during Phase 2/3 without a
    # signature change.
    _verdict_store = InMemoryVerdictStore()
    conformal_calibrator = SplitConformalCalibrator(InMemoryCalibrationStore())
    _pipeline = Pipeline(
        decomposer=_claim_decomposer,
        retrieval=_evidence_collector,
        conformal=conformal_calibrator,
        domain_router=None,  # Phase 3 Task 3.2
        nli=None,  # NLIService (cross-encoder) — reserved for future path
        nli_adapter=_nli_adapter,  # Phase 2 Task 2.1 (LLM-based NLI, live)
        pcg=None,  # Phase 2 Task 2.5
        domain_adapters=None,  # Phase 3 Task 3.1
    )

    logger.info("DI container initialized — v2 pipeline live with Phase 1 placeholders")


def _build_mcp_sources(settings: Any) -> list[MCPKnowledgeSource]:
    """Assemble the list of MCP knowledge sources enabled by config."""
    sources: list[MCPKnowledgeSource] = []
    try:
        mcp_cfg = settings.mcp
    except AttributeError:
        return sources

    # Skip MCP wiring in tests so unit tests never hit live external APIs
    # even if MCP_*_ENABLED defaults are left at their pydantic values.
    in_test_env = getattr(settings, "environment", "development") == "test"

    # MediaWiki (live Wikipedia Action API) — Phase 2 Wave 1 cheap-evidence
    # source. Default ON via MCPSettings.wikipedia_enabled.
    if getattr(mcp_cfg, "wikipedia_enabled", True) and not in_test_env:
        sources.append(MediaWikiAdapter())

    # OHI unified MCP server. Default URL http://ohi-mcp-server:8080 is
    # unresolvable in prod; require explicit MCP_OHI_ENABLED=true at the
    # process env so pydantic's default-True cannot accidentally enable
    # the dead adapter at Lambda cold-start. Import retained for local-
    # compose/dev use.
    _ohi_env = os.environ.get("MCP_OHI_ENABLED")
    if (
        _ohi_env is not None
        and _ohi_env.lower() == "true"
        and getattr(mcp_cfg, "ohi_enabled", False)
    ):
        sources.append(OHIMCPAdapter(mcp_cfg))

    return sources


async def _cleanup_adapters() -> None:
    """Close connections during lifespan shutdown."""
    if _graph_store is not None:
        try:
            await _graph_store.close()
        except Exception as e:
            logger.warning(f"Graph store close failed: {e}")
    if _vector_store is not None:
        try:
            await _vector_store.close()
        except Exception as e:
            logger.warning(f"Vector store close failed: {e}")
    if _trace_store is not None:
        try:
            await _trace_store.close()
        except Exception as e:
            logger.warning(f"Trace store close failed: {e}")
    for _src in _mcp_sources:
        try:
            await _src.disconnect()
        except Exception as e:
            logger.warning(
                f"MCP source {getattr(_src, 'source_name', type(_src).__name__)} "
                f"disconnect failed: {e}"
            )
    logger.info("DI container cleanup complete")


# -----------------------------------------------------------------------------
# FastAPI dependency functions
# -----------------------------------------------------------------------------


def get_llm_provider() -> LLMProvider:
    if _llm_provider is None:
        raise RuntimeError("LLM provider not initialized (lifespan not started?)")
    return _llm_provider


def get_nli_adapter() -> NliAdapter:
    if _nli_adapter is None:
        raise RuntimeError("NLI adapter not initialized (lifespan not started?)")
    return _nli_adapter


def get_graph_store() -> GraphKnowledgeStore:
    if _graph_store is None:
        raise RuntimeError("Graph store not initialized")
    return _graph_store


def get_vector_store() -> VectorKnowledgeStore:
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized")
    return _vector_store


def get_trace_store() -> RedisTraceAdapter | None:
    return _trace_store


def get_claim_decomposer() -> ClaimDecomposer:
    if _claim_decomposer is None:
        raise RuntimeError("Claim decomposer not initialized")
    return _claim_decomposer


def get_mcp_sources() -> list[MCPKnowledgeSource]:
    return list(_mcp_sources)


def get_evidence_collector() -> AdaptiveEvidenceCollector:
    if _evidence_collector is None:
        raise RuntimeError("Evidence collector not initialized")
    return _evidence_collector


def get_mcp_selector() -> SmartMCPSelector | None:
    return _mcp_selector


def get_mesh_builder() -> KnowledgeMeshBuilder:
    if _mesh_builder is None:
        raise RuntimeError("Mesh builder not initialized")
    return _mesh_builder


def get_pipeline() -> Pipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized (lifespan not started?)")
    return _pipeline


def get_verdict_store() -> VerdictStore:
    if _verdict_store is None:
        raise RuntimeError("Verdict store not initialized")
    return _verdict_store
