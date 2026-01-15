"""
Hybrid Verification Oracle
==========================

Verifies claims using multiple strategies (graph + vector + MCP).
Includes ADAPTIVE strategy with intelligent tiered collection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from open_hallucination_index.domain.entities import Evidence
from open_hallucination_index.domain.results import (
    CitationTrace,
    VerificationStatus,
)
from open_hallucination_index.ports.verification_oracle import (
    VerificationOracle,
    VerificationStrategy,
)

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.domain.services.evidence_collector import (
        AdaptiveEvidenceCollector,
    )
    from open_hallucination_index.domain.services.mcp_selector import SmartMCPSelector
    from open_hallucination_index.ports.knowledge_store import (
        GraphKnowledgeStore,
        VectorKnowledgeStore,
    )
    from open_hallucination_index.ports.mcp_source import MCPKnowledgeSource

logger = logging.getLogger(__name__)

# Thresholds for verification decisions
SUPPORT_THRESHOLD = 0.75  # Similarity score to consider as supporting
REFUTE_THRESHOLD = 0.6  # Similarity + contradiction detection
MIN_EVIDENCE_COUNT = 1  # Minimum evidence pieces needed


class HybridVerificationOracle(VerificationOracle):
    """
    Verification oracle supporting multiple strategies.

    Strategies:
    - GRAPH_EXACT: Only use graph database for exact matching
    - VECTOR_SEMANTIC: Only use vector store for semantic matching
    - HYBRID: Use both in parallel, merge results
    - CASCADING: Try graph first, fall back to vector if no results
    - MCP_ENHANCED: Query MCP sources first, fall back to local stores
    - ADAPTIVE: Intelligent tiered collection with early-exit and latency optimization
    """

    def __init__(
        self,
        graph_store: GraphKnowledgeStore | None = None,
        vector_store: VectorKnowledgeStore | None = None,
        mcp_sources: list[MCPKnowledgeSource] | None = None,
        default_strategy: VerificationStrategy = VerificationStrategy.ADAPTIVE,
        persist_mcp_evidence: bool = True,
        persist_to_vector: bool = True,
        # New components for ADAPTIVE strategy
        evidence_collector: AdaptiveEvidenceCollector | None = None,
        mcp_selector: SmartMCPSelector | None = None,
    ) -> None:
        """
        Initialize the oracle.

        Args:
            graph_store: Graph knowledge store for exact matching.
            vector_store: Vector store for semantic matching.
            mcp_sources: List of MCP knowledge sources (Wikipedia, Context7, etc.).
            default_strategy: Default verification strategy.
            persist_mcp_evidence: Whether to persist MCP evidence to graph store.
            persist_to_vector: Whether to also persist to vector store.
            evidence_collector: AdaptiveEvidenceCollector for ADAPTIVE strategy.
            mcp_selector: SmartMCPSelector for intelligent source selection.
        """
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._mcp_sources = mcp_sources or []
        self._strategy = default_strategy
        self._persist_mcp_evidence = persist_mcp_evidence
        self._persist_to_vector = persist_to_vector
        self._evidence_collector = evidence_collector
        self._mcp_selector = mcp_selector

    @property
    def current_strategy(self) -> VerificationStrategy:
        """Return the current default verification strategy."""
        return self._strategy

    async def set_strategy(self, strategy: VerificationStrategy) -> None:
        """Change the default verification strategy."""
        self._strategy = strategy

    async def verify_claim(
        self,
        claim: Claim,
        strategy: VerificationStrategy | None = None,
        *,
        target_sources: int | None = None,
    ) -> tuple[VerificationStatus, CitationTrace]:
        """
        Verify a single claim against knowledge sources.

        Args:
            claim: The claim to verify.
            strategy: Verification strategy to use (or default).

        Returns:
            Tuple of (verification status, citation trace with evidence).
        """
        active_strategy = strategy or self._strategy

        # Gather evidence based on strategy
        supporting_evidence: list[Evidence] = []
        refuting_evidence: list[Evidence] = []
        all_evidence: list[Evidence] = []

        try:
            if active_strategy == VerificationStrategy.GRAPH_EXACT:
                all_evidence = await self._graph_evidence(claim)

            elif active_strategy == VerificationStrategy.VECTOR_SEMANTIC:
                all_evidence = await self._vector_evidence(claim)

            elif active_strategy == VerificationStrategy.HYBRID:
                # Query both in parallel
                graph_task = self._graph_evidence(claim)
                vector_task = self._vector_evidence(claim)
                graph_ev, vector_ev = await asyncio.gather(
                    graph_task, vector_task, return_exceptions=True
                )

                if not isinstance(graph_ev, BaseException):
                    all_evidence.extend(graph_ev)
                if not isinstance(vector_ev, BaseException):
                    all_evidence.extend(vector_ev)

            elif active_strategy == VerificationStrategy.CASCADING:
                # Try graph first
                all_evidence = await self._graph_evidence(claim)
                if not all_evidence:
                    # Fall back to vector
                    all_evidence = await self._vector_evidence(claim)

            elif active_strategy == VerificationStrategy.MCP_ENHANCED:
                # MCP Enhanced: Query MCP sources first, then fallback to local
                all_evidence = await self._mcp_enhanced_evidence(claim)

            elif active_strategy == VerificationStrategy.ADAPTIVE:
                # ADAPTIVE: Use AdaptiveEvidenceCollector for intelligent tiered collection
                all_evidence = await self._adaptive_evidence(
                    claim,
                    target_sources=target_sources,
                )

        except Exception as e:
            logger.error(f"Evidence gathering failed: {e}")
            return self._unverifiable_result(claim, str(e), active_strategy)

        # Classify evidence as supporting or refuting
        supporting_evidence, refuting_evidence = self._classify_evidence(claim, all_evidence)

        # Determine verification status
        status, confidence, reasoning = self._determine_status(
            claim, supporting_evidence, refuting_evidence
        )

        trace = CitationTrace(
            claim_id=claim.id,
            status=status,
            reasoning=reasoning,
            supporting_evidence=supporting_evidence,
            refuting_evidence=refuting_evidence,
            confidence=confidence,
            verification_strategy=active_strategy.value,
        )

        return status, trace

    async def verify_claims(
        self,
        claims: list[Claim],
        strategy: VerificationStrategy | None = None,
        *,
        target_sources: int | None = None,
    ) -> list[tuple[VerificationStatus, CitationTrace]]:
        """
        Verify multiple claims (parallelized).

        Args:
            claims: List of claims to verify.
            strategy: Verification strategy to use.

        Returns:
            List of (status, trace) tuples, one per claim.
        """
        if not claims:
            return []

        # Verify all claims in parallel
        tasks = [
            self.verify_claim(claim, strategy, target_sources=target_sources) for claim in claims
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results: list[tuple[VerificationStatus, CitationTrace]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Claim {i} verification failed: {result}")
                final_results.append(
                    self._unverifiable_result(claims[i], str(result), strategy or self._strategy)
                )
            else:
                final_results.append(result)

        return final_results

    async def _graph_evidence(self, claim: Claim) -> list[Evidence]:
        """Gather evidence from graph store."""
        if self._graph_store is None:
            return []

        try:
            return await self._graph_store.find_evidence_for_claim(claim)
        except Exception as e:
            logger.warning(f"Graph evidence lookup failed: {e}")
            return []

    async def _vector_evidence(self, claim: Claim) -> list[Evidence]:
        """Gather evidence from vector store."""
        if self._vector_store is None:
            return []

        try:
            return await self._vector_store.find_evidence_for_claim(claim)
        except Exception as e:
            logger.warning(f"Vector evidence lookup failed: {e}")
            return []

    async def _mcp_evidence(self, claim: Claim) -> list[Evidence]:
        """Gather evidence from all MCP sources."""
        if not self._mcp_sources:
            return []

        all_evidence: list[Evidence] = []
        tasks = []

        for source in self._mcp_sources:
            if source.is_available:
                tasks.append(source.find_evidence(claim))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"MCP source query failed: {result}")
            else:
                all_evidence.extend(result)
                # Persist MCP evidence to graph store if enabled
                if self._persist_mcp_evidence and result:
                    await self._persist_evidence_to_graph(result)

    async def _persist_evidence_to_graph(self, evidence_list: list[Evidence]) -> None:
        """Persist MCP evidence to graph store for future lookups."""
        if self._graph_store is None:
            return

        for ev in evidence_list:
            try:
                # Check if graph store has persist method
                if hasattr(self._graph_store, "persist_external_evidence"):
                    await self._graph_store.persist_external_evidence(ev)
            except Exception as e:
                logger.debug(f"Failed to persist MCP evidence: {e}")

    async def _mcp_enhanced_evidence(self, claim: Claim) -> list[Evidence]:
        """
        MCP-enhanced evidence gathering with fallback.

        1. Query ALL sources (MCP + local) in parallel for speed
        2. Persist MCP evidence to graph for future lookups
        3. Combine and deduplicate evidence
        """
        all_evidence: list[Evidence] = []
        tasks = []
        task_names = []

        # Queue MCP sources
        for source in self._mcp_sources:
            if source.is_available:
                tasks.append(source.find_evidence(claim))
                task_names.append(f"mcp:{source.source_name}")

        # Queue local stores (run in parallel with MCP)
        if self._graph_store is not None:
            tasks.append(self._graph_evidence(claim))
            task_names.append("graph")
        if self._vector_store is not None:
            tasks.append(self._vector_evidence(claim))
            task_names.append("vector")

        if not tasks:
            return []

        # Run ALL sources in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        mcp_evidence_to_persist = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.debug(f"Source {task_names[i]} failed: {result}")
            else:
                all_evidence.extend(result)
                # Collect MCP evidence for persistence
                if task_names[i].startswith("mcp:") and result:
                    mcp_evidence_to_persist.extend(result)

        # Persist MCP evidence asynchronously (don't wait)
        if self._persist_mcp_evidence and mcp_evidence_to_persist:
            asyncio.create_task(self._persist_evidence_to_graph(mcp_evidence_to_persist))

        if not all_evidence:
            logger.debug(
                "No evidence found from any source for claim: %s",
                claim.text[:100],
            )

    async def _adaptive_evidence(
        self,
        claim: Claim,
        *,
        target_sources: int | None = None,
    ) -> list[Evidence]:
        """
        Adaptive evidence gathering with intelligent tiered collection.

        Uses AdaptiveEvidenceCollector for:
        1. Tier 1: Local sources (Neo4j + Qdrant) with early-exit check
        2. Tier 2: Selected MCP sources based on claim domain
        3. Quality-weighted accumulation
        4. Background completion for cache warming

        Falls back to MCP_ENHANCED if collector not configured.
        """
        if self._evidence_collector is None:
            # Fallback to MCP_ENHANCED if no collector configured
            logger.debug("No evidence collector configured, falling back to MCP_ENHANCED")
            return await self._mcp_enhanced_evidence(claim)

        try:
            total_source_cap = 20
            local_sources_count = int(self._graph_store is not None) + int(
                self._vector_store is not None
            )
            max_mcp_allowed = max(total_source_cap - local_sources_count, 0)

            # Get MCP sources to query based on claim domain
            mcp_sources = None
            if self._mcp_selector is not None:
                selection = (
                    self._mcp_selector.select(
                        claim,
                        max_sources_override=target_sources,
                    )
                    if target_sources is not None
                    else self._mcp_selector.select(claim)
                )
                mcp_sources = self._mcp_selector.get_sources_for_selection(selection)
                if max_mcp_allowed and len(mcp_sources) > max_mcp_allowed:
                    mcp_sources = mcp_sources[:max_mcp_allowed]
                logger.debug(
                    f"Claim domain: {selection.domain.value}, "
                    f"selected {len(selection.selected_sources)} MCP sources"
                )
            else:
                # Use all available MCP sources
                mcp_sources = [s for s in self._mcp_sources if s.is_available]
                if target_sources is not None:
                    mcp_sources = mcp_sources[: min(target_sources, max_mcp_allowed)]
                elif max_mcp_allowed:
                    mcp_sources = mcp_sources[:max_mcp_allowed]

            # Collect evidence using adaptive collector
            result = await self._evidence_collector.collect(claim, mcp_sources)

            logger.debug(
                f"Adaptive collection: {len(result.evidence)} evidence, "
                f"weighted_value={result.total_weighted_value:.2f}, "
                f"latency={result.latency_ms:.1f}ms, "
                f"early_exit={result.early_exit}"
            )

            # Persist MCP evidence to both graph and vector stores
            mcp_evidence = [
                ev
                for ev in result.evidence
                if ev.source.value not in ("graph_exact", "graph_inferred", "vector_semantic")
            ]

            if mcp_evidence:
                if self._persist_mcp_evidence:
                    asyncio.create_task(self._persist_evidence_to_graph(mcp_evidence))
                if self._persist_to_vector and self._vector_store is not None:
                    asyncio.create_task(self._persist_evidence_to_vector(mcp_evidence))

            return result.evidence

        except Exception as e:
            logger.warning(
                "Adaptive evidence collection failed: %s, falling back to MCP_ENHANCED",
                e,
            )
            return await self._mcp_enhanced_evidence(claim)

    async def _persist_evidence_to_vector(self, evidence_list: list[Evidence]) -> None:
        """Persist evidence to vector store for semantic fallback."""
        if self._vector_store is None:
            return

        try:
            if hasattr(self._vector_store, "persist_external_evidence"):
                await self._vector_store.persist_external_evidence(evidence_list)
                logger.debug(f"Persisted {len(evidence_list)} evidence to vector store")
        except Exception as e:
            logger.debug(f"Failed to persist evidence to vector store: {e}")

    def _classify_evidence(
        self, claim: Claim, evidence: list[Evidence]
    ) -> tuple[list[Evidence], list[Evidence]]:
        """
        Classify evidence as supporting or refuting.

        Uses similarity scores and simple heuristics.
        """
        supporting: list[Evidence] = []
        refuting: list[Evidence] = []

        claim_text_lower = claim.text.lower()

        for ev in evidence:
            # Check for high similarity (likely supporting)
            if ev.similarity_score is not None and ev.similarity_score >= SUPPORT_THRESHOLD:
                # Check for negation/contradiction signals
                if self._has_contradiction_signals(claim_text_lower, ev.content.lower()):
                    refuting.append(ev)
                else:
                    supporting.append(ev)

            # Graph evidence without similarity scores - check content match
            elif ev.similarity_score is None:
                if self._content_supports_claim(claim, ev):
                    supporting.append(ev)
                elif self._content_refutes_claim(claim, ev):
                    refuting.append(ev)

            # Medium similarity - needs more analysis
            elif ev.similarity_score >= REFUTE_THRESHOLD:
                if self._has_contradiction_signals(claim_text_lower, ev.content.lower()):
                    refuting.append(ev)
                else:
                    # Weak supporting evidence
                    supporting.append(ev)

        return supporting, refuting

    def _has_contradiction_signals(self, claim: str, evidence: str) -> bool:
        """Check for contradiction signals between claim and evidence."""
        negation_words = [
            "not",
            "never",
            "no",
            "false",
            "incorrect",
            "wrong",
            "isn't",
            "wasn't",
            "aren't",
            "weren't",
        ]

        # Simple heuristic: if one has negation and other doesn't
        claim_has_negation = any(word in claim.split() for word in negation_words)
        evidence_has_negation = any(word in evidence.split() for word in negation_words)

        return claim_has_negation != evidence_has_negation

    def _content_supports_claim(self, claim: Claim, evidence: Evidence) -> bool:
        """Check if graph evidence supports the claim."""
        if not evidence.structured_data:
            return False

        data = evidence.structured_data

        # Check SPO match for graph evidence
        if claim.subject and claim.object:
            ev_subject = str(data.get("subject", "")).lower()
            ev_object = str(data.get("object", "")).lower()

            subject_match = (
                claim.subject.lower() in ev_subject or ev_subject in claim.subject.lower()
            )
            object_match = claim.object.lower() in ev_object or ev_object in claim.object.lower()

            if subject_match and object_match:
                return True

        return False

    def _content_refutes_claim(self, claim: Claim, evidence: Evidence) -> bool:
        """Check if graph evidence refutes the claim."""
        # For now, rely on contradiction signals
        return False

    def _determine_status(
        self,
        claim: Claim,
        supporting: list[Evidence],
        refuting: list[Evidence],
    ) -> tuple[VerificationStatus, float, str]:
        """
        Determine verification status from evidence.

        Uses evidence ratio to determine status:
        - Pure support (0 refuting) → SUPPORTED
        - Pure refute (0 supporting) → REFUTED
        - Overwhelming support (ratio ≥ 5:1) → SUPPORTED with reduced confidence
        - Strong support (ratio ≥ 3:1) → PARTIALLY_SUPPORTED with high confidence
        - Mixed evidence → PARTIALLY_SUPPORTED or UNCERTAIN based on ratio

        Returns:
            Tuple of (status, confidence, reasoning).
        """
        support_count = len(supporting)
        refute_count = len(refuting)
        total_evidence = support_count + refute_count

        if total_evidence == 0:
            return (
                VerificationStatus.UNVERIFIABLE,
                0.3,
                "No relevant evidence found in knowledge base.",
            )

        # Pure refutation - no supporting evidence
        if refute_count > 0 and support_count == 0:
            confidence = min(0.95, 0.6 + 0.05 * refute_count)
            return (
                VerificationStatus.REFUTED,
                confidence,
                f"Found {refute_count} piece(s) of contradicting evidence.",
            )

        # Pure support - no refuting evidence
        if support_count > 0 and refute_count == 0:
            # Scale confidence based on evidence count (more evidence = higher confidence)
            confidence = min(0.98, 0.7 + 0.03 * support_count)
            return (
                VerificationStatus.SUPPORTED,
                confidence,
                f"Found {support_count} piece(s) of supporting evidence.",
            )

        # Mixed evidence - calculate support ratio
        support_ratio = support_count / refute_count if refute_count > 0 else float("inf")
        refute_ratio = refute_count / support_count if support_count > 0 else float("inf")

        # Overwhelming support (e.g., 12:1 or better)
        if support_ratio >= 5.0:
            # Still SUPPORTED, but with slightly reduced confidence due to contradiction
            base_confidence = 0.75 + 0.02 * support_count
            # Penalty for each refuting piece, but capped
            penalty = min(0.15, 0.05 * refute_count)
            confidence = min(0.92, base_confidence - penalty)
            return (
                VerificationStatus.SUPPORTED,
                confidence,
                (
                    "Strongly supported: "
                    f"{support_count} supporting vs {refute_count} contradicting "
                    f"(ratio {support_ratio:.1f}:1)."
                ),
            )

        # Strong support (3:1 to 5:1)
        if support_ratio >= 3.0:
            confidence = 0.70 + 0.05 * (support_ratio - 3.0)
            confidence = min(0.85, confidence)
            return (
                VerificationStatus.PARTIALLY_SUPPORTED,
                confidence,
                (
                    "Well supported: "
                    f"{support_count} supporting vs {refute_count} contradicting "
                    f"(ratio {support_ratio:.1f}:1)."
                ),
            )

        # Moderate support (2:1 to 3:1)
        if support_ratio >= 2.0:
            confidence = 0.60 + 0.05 * (support_ratio - 2.0)
            return (
                VerificationStatus.PARTIALLY_SUPPORTED,
                confidence,
                (
                    "Moderately supported: "
                    f"{support_count} supporting vs {refute_count} contradicting."
                ),
            )

        # Slight support (1:1 to 2:1)
        if support_ratio >= 1.0:
            confidence = 0.50 + 0.10 * (support_ratio - 1.0)
            return (
                VerificationStatus.UNCERTAIN,
                confidence,
                (
                    "Mixed evidence with slight support: "
                    f"{support_count} supporting vs {refute_count} contradicting."
                ),
            )

        # More refuting than supporting
        if refute_ratio >= 3.0:
            confidence = 0.65 + 0.05 * min(refute_ratio, 5.0)
            return (
                VerificationStatus.REFUTED,
                min(0.85, confidence),
                f"Strongly refuted: {refute_count} contradicting vs {support_count} supporting.",
            )

        # Slightly more refuting
        return (
            VerificationStatus.UNCERTAIN,
            0.45,
            f"Conflicting evidence: {support_count} supporting, {refute_count} contradicting.",
        )

    def _unverifiable_result(
        self, claim: Claim, error: str, strategy: VerificationStrategy
    ) -> tuple[VerificationStatus, CitationTrace]:
        """Create an unverifiable result for error cases."""
        return (
            VerificationStatus.UNVERIFIABLE,
            CitationTrace(
                claim_id=claim.id,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"Verification failed: {error}",
                supporting_evidence=[],
                refuting_evidence=[],
                confidence=0.0,
                verification_strategy=strategy.value,
            ),
        )

    async def health_check(self) -> bool:
        """Check if the oracle and its dependencies are operational."""
        checks = []

        if self._graph_store is not None:
            checks.append(self._graph_store.health_check())

        if self._vector_store is not None:
            checks.append(self._vector_store.health_check())

        # MCP sources are optional, don't fail health check if unavailable
        for source in self._mcp_sources:
            if source.is_available:
                checks.append(source.health_check())

        if not checks:
            return False

        results = await asyncio.gather(*checks, return_exceptions=True)
        # At least one local store must be healthy
        local_healthy = any(r is True for r in results[:2] if not isinstance(r, BaseException))
        return local_healthy

    def get_latency_stats(self) -> dict[str, object]:
        """
        Get latency statistics from the adaptive evidence collector.

        Returns:
            Dictionary with per-source latency metrics.
        """
        if self._evidence_collector is not None:
            return self._evidence_collector.get_latency_stats()
        return {}

    async def cleanup(self) -> None:
        """Cleanup resources (background tasks, etc.)."""
        if self._evidence_collector is not None:
            await self._evidence_collector.cleanup()
