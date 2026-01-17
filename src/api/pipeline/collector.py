"""
Adaptive Evidence Collector
===========================

High-performance evidence collection with:
1. Tiered execution (local first, MCP second)
2. Quality-weighted accumulation
3. Early exit when sufficient evidence found
4. Background completion for cache warming
5. Latency tracking per source

Enhanced with Wikipedia ingestion metadata:
6. Ingestion quality_score integration (article importance)
7. PageRank-based importance bonus
8. Disambiguation/redirect page penalties
9. Wikidata and geographic coordinate bonuses
10. Incoming link popularity weighting

This is the core latency optimization component that minimizes
response time while maximizing evidence quality.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from models.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from interfaces.mcp import MCPKnowledgeSource
    from interfaces.stores import (
        GraphKnowledgeStore,
        VectorKnowledgeStore,
    )
    from models.entities import Claim
    from pipeline.selector import SmartMCPSelector

logger = logging.getLogger(__name__)


class CollectionTier(Enum):
    """Evidence collection tier."""

    LOCAL = "local"  # Neo4j + Qdrant
    MCP = "mcp"  # External MCP sources


@dataclass
class SourceLatencyStats:
    """Rolling latency statistics for a source."""

    source_name: str
    samples: list[float] = field(default_factory=list)
    max_samples: int = 100
    timeout_count: int = 0
    error_count: int = 0
    success_count: int = 0

    def record(self, latency_ms: float, success: bool = True) -> None:
        """Record a latency sample."""
        self.samples.append(latency_ms)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def record_timeout(self) -> None:
        """Record a timeout."""
        self.timeout_count += 1

    @property
    def p50(self) -> float:
        """Get 50th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = len(sorted_samples) // 2
        return sorted_samples[idx]

    @property
    def p95(self) -> float:
        """Get 95th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0-1.0)."""
        total = self.success_count + self.error_count + self.timeout_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class EvidenceQuality:
    """Quality assessment for a piece of evidence."""

    evidence: Evidence
    quality_score: float  # 0.0-1.0
    source_reliability: float  # 0.0-1.0 based on source type
    weighted_value: float  # Combined quality score
    classification_confidence: float | None = None

    @classmethod
    def assess(cls, evidence: Evidence) -> EvidenceQuality:
        """
        Assess evidence quality.
        
        Enhanced to leverage metadata from Wikipedia ingestion:
        - quality_score: Pre-computed article importance (0-1)
        - pagerank: Link-based importance score
        - is_disambiguation: Penalize disambiguation pages
        - is_redirect: Penalize redirect pages
        - has_wikidata: Bonus for Wikidata-linked articles
        """
        # Base quality from similarity score
        base_quality = evidence.similarity_score or 0.5

        # Source reliability weights
        source_weights = {
            # Local sources (highest trust)
            EvidenceSource.GRAPH_EXACT: 1.0,
            EvidenceSource.GRAPH_INFERRED: 0.9,
            EvidenceSource.VECTOR_SEMANTIC: 0.85,
            # Academic sources (high trust)
            EvidenceSource.ACADEMIC: 0.9,
            EvidenceSource.PUBMED: 0.95,
            EvidenceSource.CLINICAL_TRIALS: 0.9,
            # Knowledge graphs (good trust)
            EvidenceSource.WIKIPEDIA: 0.8,
            EvidenceSource.KNOWLEDGE_GRAPH: 0.8,
            # News (medium trust)
            EvidenceSource.NEWS: 0.65,
            # Other
            EvidenceSource.WORLD_BANK: 0.85,
            EvidenceSource.OSV: 0.9,
            EvidenceSource.EXTERNAL_API: 0.7,
        }

        source_reliability = source_weights.get(evidence.source, 0.5)

        # Match type bonus
        match_bonus = 0.0
        if evidence.match_type == "exact":
            match_bonus = 0.15
        elif evidence.match_type == "partial":
            match_bonus = 0.05
        elif evidence.match_type == "wikidata_lookup":
            match_bonus = 0.12  # Wikidata matches are authoritative
        elif evidence.match_type == "quality_filtered":
            match_bonus = 0.10  # Pre-filtered for quality
        elif evidence.match_type == "enhanced_quality":
            match_bonus = 0.10
        elif evidence.match_type == "geographic_proximity":
            match_bonus = 0.05  # Geo matches are contextual

        quality_score = min(1.0, base_quality + match_bonus)
        classification_confidence = evidence.classification_confidence
        if classification_confidence is not None:
            quality_score = min(1.0, (quality_score + classification_confidence) / 2.0)

        # =================================================================
        # NEW: Incorporate ingestion metadata from structured_data
        # =================================================================
        if evidence.structured_data:
            sd = evidence.structured_data
            
            # Use pre-computed quality_score from ingestion if available
            ingestion_quality = sd.get("quality_score")
            if ingestion_quality is not None and isinstance(ingestion_quality, (int, float)):
                # Blend ingestion quality with computed quality
                quality_score = (quality_score * 0.6) + (float(ingestion_quality) * 0.4)
            
            # Pagerank bonus (logarithmic scale)
            pagerank = sd.get("pagerank")
            if pagerank is not None and isinstance(pagerank, (int, float)) and pagerank > 0:
                # Small bonus for high-pagerank articles
                import math
                pr_bonus = min(0.1, math.log10(pagerank * 1000000 + 1) / 60)
                quality_score = min(1.0, quality_score + pr_bonus)
            
            # Penalty for disambiguation pages
            if sd.get("is_disambiguation") is True:
                quality_score *= 0.6  # 40% penalty
                source_reliability *= 0.7
            
            # Penalty for redirect pages (shouldn't happen but guard)
            if sd.get("is_redirect") is True:
                quality_score *= 0.5  # 50% penalty
                source_reliability *= 0.5
            
            # Bonus for Wikidata-linked articles
            if sd.get("wikidata_id") or sd.get("has_wikidata") is True:
                quality_score = min(1.0, quality_score + 0.05)
            
            # Bonus for articles with coordinates (well-defined entities)
            if sd.get("has_coordinates") is True:
                quality_score = min(1.0, quality_score + 0.03)
            
            # Incoming links bonus (popularity indicator)
            incoming_links = sd.get("incoming_links")
            if incoming_links is not None and isinstance(incoming_links, int) and incoming_links > 100:
                import math
                links_bonus = min(0.08, math.log10(incoming_links + 1) / 50)
                quality_score = min(1.0, quality_score + links_bonus)

        weighted_value = quality_score * source_reliability

        return cls(
            evidence=evidence,
            quality_score=quality_score,
            source_reliability=source_reliability,
            weighted_value=weighted_value,
            classification_confidence=classification_confidence,
        )


@dataclass
class AccumulatorState:
    """State of evidence accumulation."""

    evidence: list[Evidence] = field(default_factory=list)
    quality_scores: list[EvidenceQuality] = field(default_factory=list)
    supporting_evidence: list[Evidence] = field(default_factory=list)
    refuting_evidence: list[Evidence] = field(default_factory=list)
    classification_enabled: bool = False
    total_weighted_value: float = 0.0
    high_confidence_count: int = 0
    sources_queried: set[str] = field(default_factory=set)
    tier_complete: set[CollectionTier] = field(default_factory=set)

    def add(self, ev: Evidence) -> float:
        """
        Add evidence and return its weighted value.

        Returns the weighted value added.
        """
        quality = EvidenceQuality.assess(ev)
        self.evidence.append(ev)
        self.quality_scores.append(quality)

        if quality.classification_confidence is not None:
            if quality.classification_confidence >= 0.6:
                self.supporting_evidence.append(ev)
            elif quality.classification_confidence <= 0.4:
                self.refuting_evidence.append(ev)

        self.total_weighted_value += quality.weighted_value

        if quality.weighted_value >= 0.75:
            self.high_confidence_count += 1

        return quality.weighted_value

    def add_batch(self, evidence_list: list[Evidence]) -> float:
        """Add batch of evidence and return total weighted value added."""
        return sum(self.add(ev) for ev in evidence_list)

    def record_classification(self, supporting: list[Evidence], refuting: list[Evidence]) -> None:
        """Record supporting and refuting evidence classification results."""
        self.classification_enabled = True
        self.supporting_evidence.extend(supporting)
        self.refuting_evidence.extend(refuting)

    def is_sufficient(
        self,
        min_evidence: int = 3,
        min_weighted_value: float = 2.0,
        high_confidence_threshold: int = 2,
        *,
        require_count_only: bool = False,
    ) -> bool:
        """
        Check if accumulated evidence is sufficient.

        Uses quality-weighted assessment:
        - Need minimum count of supporting/refuting evidence
        - Or high-confidence evidence reduces requirements
        """
        # Check count threshold using non-neutral evidence when classification is enabled
        classified_count = len(self.supporting_evidence) + len(self.refuting_evidence)
        evidence_count = classified_count if self.classification_enabled else len(self.evidence)
        if evidence_count >= min_evidence:
            return True

        if require_count_only:
            return False

        # Early exit if we have enough high-confidence evidence
        if self.high_confidence_count >= high_confidence_threshold:
            return True

        # Check weighted value threshold
        return self.total_weighted_value >= min_weighted_value


@dataclass
class CollectionResult:
    """Result of evidence collection."""

    claim_id: str
    evidence: list[Evidence]
    quality_scores: list[EvidenceQuality] = field(default_factory=list)
    total_weighted_value: float = 0.0
    latency_ms: float = 0.0
    tier_latencies: dict[str, float] = field(default_factory=dict)
    sources_queried: list[str] = field(default_factory=list)
    early_exit: bool = False
    background_tasks_pending: int = 0
    supporting_evidence: list[Evidence] = field(default_factory=list)
    refuting_evidence: list[Evidence] = field(default_factory=list)


class AdaptiveEvidenceCollector:
    """
    Adaptive evidence collector with tiered execution.

    Collection flow:
    1. Start Tier 1 (local: Neo4j + Qdrant) in parallel
    2. Check sufficiency after Tier 1
    3. If insufficient, start Tier 2 (selected MCP sources)
    4. Use early-exit when sufficient evidence accumulated
    5. Allow background completion for cache warming
    """

    def __init__(
        self,
        graph_store: GraphKnowledgeStore | None = None,
        vector_store: VectorKnowledgeStore | None = None,
        mcp_selector: SmartMCPSelector | None = None,
        *,
        # Sufficiency thresholds
        min_evidence_count: int = 3,
        min_weighted_value: float = 2.0,
        high_confidence_threshold: int = 2,
        # Timeout settings
        local_timeout_ms: float = 50.0,
        mcp_timeout_ms: float = 500.0,
        total_timeout_ms: float = 2000.0,
        # Background completion
        enable_background_completion: bool = True,
    ) -> None:
        """Initialize the collector."""
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._mcp_selector = mcp_selector

        # Thresholds
        self._min_evidence = min_evidence_count
        self._min_weighted_value = min_weighted_value
        self._high_confidence = high_confidence_threshold

        # Timeouts
        self._local_timeout = local_timeout_ms / 1000.0
        self._mcp_timeout = mcp_timeout_ms / 1000.0
        self._total_timeout = total_timeout_ms / 1000.0

        # Background completion
        self._enable_background = enable_background_completion
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Latency tracking
        self._latency_stats: dict[str, SourceLatencyStats] = {}

        # Callbacks for persisting evidence
        self._persist_callbacks: list[Callable[[list[Evidence]], Coroutine[Any, Any, None]]] = []

    def add_persist_callback(
        self,
        callback: Callable[[list[Evidence]], Coroutine[Any, Any, None]],
    ) -> None:
        """Add a callback for persisting evidence (e.g., to Neo4j/Qdrant)."""
        self._persist_callbacks.append(callback)

    async def collect(
        self,
        claim: Claim,
        mcp_sources: list[MCPKnowledgeSource] | None = None,
        target_evidence_count: int | None = None,
        *,
        allow_early_exit: bool = True,
        classifier: Callable[[list[Evidence]], Coroutine[Any, Any, tuple[list[Evidence], list[Evidence]]]] | None = None,
    ) -> CollectionResult:
        """
        Collect evidence for a claim using adaptive tiered approach.

        Args:
            claim: The claim to find evidence for.
            mcp_sources: Optional MCP sources (uses selector if not provided).
            target_evidence_count: Optional target number of evidence items to find.
            classifier: Optional classifier returning (supporting, refuting) evidence.

        Returns:
            CollectionResult with evidence and metrics.
        """
        start_time = time.perf_counter()
        accumulator = AccumulatorState()
        tier_latencies: dict[str, float] = {}

        # === TIER 1: Local sources (Neo4j + Qdrant) ===
        tier1_start = time.perf_counter()
        tier1_evidence = await self._collect_local(claim, accumulator)
        tier1_latency = (time.perf_counter() - tier1_start) * 1000
        tier_latencies["local"] = tier1_latency

        accumulator.add_batch(tier1_evidence)
        if classifier and tier1_evidence:
            supporting, refuting = await classifier(tier1_evidence)
            accumulator.record_classification(supporting, refuting)
        accumulator.tier_complete.add(CollectionTier.LOCAL)

        logger.debug(
            f"Tier 1 complete: {len(tier1_evidence)} evidence, "
            f"weighted={accumulator.total_weighted_value:.2f}, "
            f"latency={tier1_latency:.1f}ms"
        )

        effective_min_evidence = (
            target_evidence_count
            if target_evidence_count is not None
            else self._min_evidence
        )

        # Check if local evidence is sufficient
        if allow_early_exit and accumulator.is_sufficient(
            effective_min_evidence,
            self._min_weighted_value,
            self._high_confidence,
            require_count_only=bool(target_evidence_count is not None),
        ):
            logger.debug("Early exit after Tier 1 (local sources sufficient)")
            return self._build_result(
                claim, accumulator, start_time, tier_latencies, early_exit=True
            )

        # === TIER 2: MCP sources (selected based on claim) ===
        tier2_start = time.perf_counter()
        tier2_evidence, pending_count = await self._collect_mcp(
            claim,
            mcp_sources,
            accumulator,
            min_evidence=effective_min_evidence,
            allow_early_exit=allow_early_exit,
            classifier=classifier,
        )
        tier2_latency = (time.perf_counter() - tier2_start) * 1000
        tier_latencies["mcp"] = tier2_latency

        accumulator.tier_complete.add(CollectionTier.MCP)

        logger.debug(
            f"Tier 2 complete: {len(tier2_evidence)} evidence, "
            f"total={len(accumulator.evidence)}, "
            f"weighted={accumulator.total_weighted_value:.2f}, "
            f"latency={tier2_latency:.1f}ms"
        )

        return self._build_result(
            claim,
            accumulator,
            start_time,
            tier_latencies,
            early_exit=False,
            pending_count=pending_count,
        )

    async def _collect_local(self, claim: Claim, accumulator: AccumulatorState) -> list[Evidence]:
        """Collect evidence from local sources (Neo4j + Qdrant)."""
        tasks: list[asyncio.Task[list[Evidence]]] = []
        task_names: list[str] = []

        if self._graph_store is not None:
            task = asyncio.create_task(
                self._timed_query("neo4j", self._graph_store.find_evidence_for_claim(claim))
            )
            tasks.append(task)
            task_names.append("neo4j")

        if self._vector_store is not None:
            task = asyncio.create_task(
                self._timed_query("qdrant", self._vector_store.find_evidence_for_claim(claim))
            )
            tasks.append(task)
            task_names.append("qdrant")

        if not tasks:
            return []

        # Wait with timeout
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._local_timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
        except Exception as e:
            logger.warning(f"Local collection failed: {e}")
            return []

        # Cancel timed-out tasks
        for task in pending:
            task.cancel()
            idx = tasks.index(task)
            self._record_timeout(task_names[idx])

        # Collect results
        evidence: list[Evidence] = []
        task_name_map = dict(zip(tasks, task_names, strict=False))
        for task in done:
            try:
                result = task.result()
                evidence.extend(result)
                source_name = task_name_map.get(task)
                if source_name:
                    accumulator.sources_queried.add(source_name)
            except Exception as e:
                logger.debug(f"Local source error: {e}")

        return evidence

    async def _collect_mcp(
        self,
        claim: Claim,
        mcp_sources: list[MCPKnowledgeSource] | None,
        accumulator: AccumulatorState,
        min_evidence: int | None = None,
        *,
        allow_early_exit: bool = True,
        classifier: Callable[[list[Evidence]], Coroutine[Any, Any, tuple[list[Evidence], list[Evidence]]]] | None = None,
    ) -> tuple[list[Evidence], int]:
        """
        Collect evidence from MCP sources with early exit.

        Returns (evidence_list, pending_background_tasks_count).
        """
        effective_min = min_evidence if min_evidence is not None else self._min_evidence
        logger.debug(f"Entering _collect_mcp, mcp_sources count: {len(mcp_sources) if mcp_sources else 'None'}")
        # Get MCP sources to query
        if mcp_sources:
            sources = mcp_sources
        elif self._mcp_selector:
            logger.debug("Using selector to find sources in collector")
            selection = await self._mcp_selector.select(claim)
            sources = self._mcp_selector.get_sources_for_selection(selection)
        else:
            logger.warning("No sources and no selector")
            return [], 0

        logger.debug(f"Effective sources to query: {[s.source_name for s in sources]}")

        if not sources:
            return [], 0
            
        # Create tasks for each source
        tasks: dict[asyncio.Task[list[Evidence]], str] = {}
        for source in sources:
            if source.is_available:
                logger.debug(f"Creating task for available source: {source.source_name}")
                task = asyncio.create_task(
                    self._timed_query(source.source_name, source.find_evidence(claim))
                )
                tasks[task] = source.source_name
            else:
                logger.warning(f"Source {source.source_name} is NOT available")

        if not tasks:
            return [], 0

        evidence: list[Evidence] = []
        pending_count = 0

        # Use wait with FIRST_COMPLETED for early exit
        remaining_tasks = set(tasks.keys())
        deadline = time.perf_counter() + self._mcp_timeout

        while remaining_tasks:
            # Check timeout
            time_left = deadline - time.perf_counter()
            if time_left <= 0:
                break

            # Wait for first completion
            try:
                done, remaining_tasks = await asyncio.wait(
                    remaining_tasks,
                    timeout=min(time_left, 0.1),  # Check every 100ms
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                break

            # Process completed tasks
            for task in done:
                source_name = tasks[task]
                try:
                    result = task.result()
                    evidence.extend(result)
                    accumulator.sources_queried.add(source_name)

                    # Add to accumulator for sufficiency check
                    for ev in result:
                        accumulator.add(ev)

                    if classifier:
                        supporting, refuting = await classifier(result)
                        accumulator.record_classification(supporting, refuting)

                    if result:
                        for callback in self._persist_callbacks:
                            try:
                                await callback(result)
                            except Exception as e:
                                logger.debug(f"Persist callback failed: {e}")

                except Exception as e:
                    logger.debug(f"MCP source {source_name} error: {e}")

            # Check if we have sufficient evidence
            if allow_early_exit and accumulator.is_sufficient(
                effective_min,
                self._min_weighted_value,
                self._high_confidence,
                require_count_only=bool(min_evidence is not None),
            ):
                logger.debug(
                    f"Early exit during MCP collection: {len(evidence)} evidence, "
                    f"{len(remaining_tasks)} tasks remaining"
                )

                # Handle remaining tasks
                if self._enable_background:
                    # Let them complete in background for caching
                    for task in remaining_tasks:
                        bg_task = asyncio.create_task(self._background_complete(task, tasks[task]))
                        self._background_tasks.add(bg_task)
                        bg_task.add_done_callback(self._background_tasks.discard)
                    pending_count = len(remaining_tasks)
                else:
                    # Cancel remaining tasks
                    for task in remaining_tasks:
                        task.cancel()

                break

        return evidence, pending_count

    async def _timed_query(
        self,
        source_name: str,
        coro: Coroutine[Any, Any, list[Evidence]],
    ) -> list[Evidence]:
        """Execute a query with timing."""
        start = time.perf_counter()
        try:
            result = await coro
            latency = (time.perf_counter() - start) * 1000
            self._record_latency(source_name, latency, success=True)
            return result
        except Exception:
            latency = (time.perf_counter() - start) * 1000
            self._record_latency(source_name, latency, success=False)
            raise

    async def _background_complete(
        self,
        task: asyncio.Task[list[Evidence]],
        source_name: str,
    ) -> None:
        """Complete a task in background and persist results."""
        try:
            result = await task
            if result:
                logger.debug(f"Background task {source_name} completed with {len(result)} evidence")
                # Trigger persist callbacks
                for callback in self._persist_callbacks:
                    try:
                        await callback(result)
                    except Exception as e:
                        logger.debug(f"Persist callback failed: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Background task {source_name} failed: {e}")

    def _record_latency(self, source: str, latency_ms: float, success: bool) -> None:
        """Record latency for a source."""
        if source not in self._latency_stats:
            self._latency_stats[source] = SourceLatencyStats(source_name=source)
        self._latency_stats[source].record(latency_ms, success)

    def _record_timeout(self, source: str) -> None:
        """Record timeout for a source."""
        if source not in self._latency_stats:
            self._latency_stats[source] = SourceLatencyStats(source_name=source)
        self._latency_stats[source].record_timeout()

    def _build_result(
        self,
        claim: Claim,
        accumulator: AccumulatorState,
        start_time: float,
        tier_latencies: dict[str, float],
        early_exit: bool,
        pending_count: int = 0,
    ) -> CollectionResult:
        """Build the collection result."""
        total_latency = (time.perf_counter() - start_time) * 1000

        return CollectionResult(
            claim_id=str(claim.id),
            evidence=accumulator.evidence,
            supporting_evidence=accumulator.supporting_evidence,
            refuting_evidence=accumulator.refuting_evidence,
            quality_scores=accumulator.quality_scores,
            total_weighted_value=accumulator.total_weighted_value,
            latency_ms=total_latency,
            tier_latencies=tier_latencies,
            sources_queried=list(accumulator.sources_queried),
            early_exit=early_exit,
            background_tasks_pending=pending_count,
        )

    def get_latency_stats(self) -> dict[str, dict[str, Any]]:
        """Get latency statistics for all sources."""
        return {
            name: {
                "p50_ms": stats.p50,
                "p95_ms": stats.p95,
                "success_rate": stats.success_rate,
                "timeout_count": stats.timeout_count,
                "sample_count": len(stats.samples),
            }
            for name, stats in self._latency_stats.items()
        }

    async def cleanup(self) -> None:
        """Cancel all background tasks."""
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
