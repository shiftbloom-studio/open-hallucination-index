"""
Hybrid Verification Oracle
==========================

Verifies claims using multiple strategies (graph + vector + MCP).
Includes ADAPTIVE strategy with intelligent tiered collection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from interfaces.verification import (
    EvidenceTier,
    VerificationOracle,
    VerificationStrategy,
)
from models.entities import Evidence
from models.results import (
    CitationTrace,
    VerificationStatus,
)

if TYPE_CHECKING:
    from interfaces.llm import LLMProvider
    from interfaces.mcp import MCPKnowledgeSource
    from interfaces.stores import (
        GraphKnowledgeStore,
        VectorKnowledgeStore,
    )
    from models.entities import Claim
    from pipeline.collector import (
        AdaptiveEvidenceCollector,
    )
    from pipeline.selector import SmartMCPSelector

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
        # LLM for evidence classification
        llm_provider: LLMProvider | None = None,
        # Configuration settings
        verification_settings: Any | None = None,
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
            llm_provider: LLM provider for intelligent evidence classification.
            verification_settings: Configuration settings for verification behavior.
        """
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._mcp_sources = mcp_sources or []
        self._strategy = default_strategy
        self._persist_mcp_evidence = persist_mcp_evidence
        self._persist_to_vector = persist_to_vector
        self._evidence_collector = evidence_collector
        self._mcp_selector = mcp_selector
        self._llm_provider = llm_provider
        
        # Store verification settings (import here to avoid circular dependency)
        if verification_settings is None:
            from config.settings import VerificationSettings
            self._verification_settings = VerificationSettings()
        else:
            self._verification_settings = verification_settings

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
        tier: EvidenceTier = EvidenceTier.DEFAULT,
    ) -> tuple[VerificationStatus, CitationTrace]:
        """
        Verify a single claim against knowledge sources.

        Args:
            claim: The claim to verify.
            strategy: Verification strategy to use (or default).
            target_sources: Optional limit on number of sources to query.
            tier: Evidence collection tier (local, default, max).

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
                # Respect tier parameter - skip MCP for LOCAL tier
                if tier == EvidenceTier.LOCAL:
                    # LOCAL tier: only graph + vector
                    all_evidence = await self._hybrid_evidence(claim)
                else:
                    all_evidence = await self._mcp_enhanced_evidence(
                        claim,
                        target_sources=target_sources,
                        tier=tier,
                    )

            elif active_strategy == VerificationStrategy.ADAPTIVE:
                # ADAPTIVE: Use AdaptiveEvidenceCollector for intelligent tiered collection
                # Respect tier parameter for source selection
                all_evidence = await self._adaptive_evidence(
                    claim,
                    target_sources=target_sources,
                    tier=tier,
                )

        except Exception as e:
            logger.error(f"Evidence gathering failed: {e}")
            return self._unverifiable_result(claim, str(e), active_strategy)

        # Classify evidence as supporting or refuting
        supporting_evidence, refuting_evidence = await self._classify_evidence(claim, all_evidence)

        # If no evidence found anywhere, allow LLM plausibility fallback (low weight)
        if not all_evidence and self._llm_provider is not None:
            status, confidence, reasoning = await self._llm_plausibility_fallback(claim)
            trace = CitationTrace(
                claim_id=claim.id,
                status=status,
                reasoning=reasoning,
                supporting_evidence=[],
                refuting_evidence=[],
                confidence=confidence,
                verification_strategy=active_strategy.value,
            )
            return status, trace

        # Determine verification status
        status, confidence, reasoning = self._determine_status(
            claim, supporting_evidence, refuting_evidence
        )

        # Lightly blend in LLM plausibility prior as an initial bias
        if self._llm_provider is not None and all_evidence:
            prior_confidence, prior_reason = await self._llm_plausibility_prior(claim)
            confidence = 0.9 * confidence + 0.1 * prior_confidence
            confidence = max(0.0, min(1.0, confidence))
            if prior_reason:
                reasoning = f"{reasoning} LLM prior: {prior_reason}".strip()

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
        tier: EvidenceTier = EvidenceTier.DEFAULT,
    ) -> list[tuple[VerificationStatus, CitationTrace]]:
        """
        Verify multiple claims (parallelized).

        Args:
            claims: List of claims to verify.
            strategy: Verification strategy to use.
            target_sources: Optional limit on number of sources to query.
            tier: Evidence collection tier (local, default, max).

        Returns:
            List of (status, trace) tuples, one per claim.
        """
        if not claims:
            return []

        # Verify all claims in parallel
        tasks = [
            self.verify_claim(claim, strategy, target_sources=target_sources, tier=tier)
            for claim in claims
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
        """
        Gather evidence from graph store.
        
        Optimized to use relationship-aware queries from Neo4j adapter:
        - Detects entity types and uses specialized query methods
        - Leverages 27 relationship types from ingestion
        """
        if self._graph_store is None:
            return []

        try:
            # Try relationship-aware queries if Neo4j adapter supports them
            if hasattr(self._graph_store, "query_by_relationship_type"):
                # Extract entity hints from claim text
                entity = self._extract_entity_from_claim(claim)
                
                if entity:
                    # Use specialized queries based on claim type
                    if self._looks_like_person_claim(claim):
                        person_evidence = await self._graph_store.query_person_facts(entity)
                        if person_evidence:
                            return person_evidence
                    
                    elif self._looks_like_org_claim(claim):
                        org_evidence = await self._graph_store.query_organization_facts(entity)
                        if org_evidence:
                            return org_evidence
                    
                    elif self._looks_like_place_claim(claim):
                        place_evidence = await self._graph_store.query_geographic_facts(entity)
                        if place_evidence:
                            return place_evidence
            
            # Fallback to standard graph query
            return await self._graph_store.find_evidence_for_claim(claim)
        except Exception as e:
            logger.warning(f"Graph evidence lookup failed: {e}")
            return []
    
    def _extract_entity_from_claim(self, claim: Claim) -> str | None:
        """Extract the main entity from a claim (first capitalized phrase)."""
        if claim.subject:
            return claim.subject
        
        # Simple heuristic: find first capitalized word sequence
        words = claim.text.split()
        entity_words = []
        for word in words:
            if word and word[0].isupper():
                entity_words.append(word)
            elif entity_words:
                break
        
        return " ".join(entity_words) if entity_words else None
    
    def _looks_like_person_claim(self, claim: Claim) -> bool:
        """Check if claim is about a person."""
        person_keywords = ["born", "died", "married", "spouse", "child", "parent", 
                          "educated", "studied", "works", "worked", "founded", "won award"]
        text_lower = claim.text.lower()
        return any(kw in text_lower for kw in person_keywords)
    
    def _looks_like_org_claim(self, claim: Claim) -> bool:
        """Check if claim is about an organization."""
        org_keywords = ["company", "corporation", "organization", "founded", "headquarters",
                       "headquartered", "industry", "ceo", "employees"]
        text_lower = claim.text.lower()
        return any(kw in text_lower for kw in org_keywords)
    
    def _looks_like_place_claim(self, claim: Claim) -> bool:
        """Check if claim is about a place."""
        place_keywords = ["city", "country", "located", "capital", "region", "part of",
                         "population", "area", "geography"]
        text_lower = claim.text.lower()
        return any(kw in text_lower for kw in place_keywords)

    async def _vector_evidence(self, claim: Claim) -> list[Evidence]:
        """Gather evidence from vector store."""
        if self._vector_store is None:
            return []

        try:
            return await self._vector_store.find_evidence_for_claim(claim)
        except Exception as e:
            logger.warning(f"Vector evidence lookup failed: {e}")
            return []

    async def _hybrid_evidence(self, claim: Claim) -> list[Evidence]:
        """Gather evidence from both graph and vector stores in parallel (local only)."""
        graph_task = self._graph_evidence(claim)
        vector_task = self._vector_evidence(claim)
        
        graph_ev, vector_ev = await asyncio.gather(
            graph_task, vector_task, return_exceptions=True
        )
        
        all_evidence: list[Evidence] = []
        if not isinstance(graph_ev, BaseException):
            all_evidence.extend(graph_ev)
        if not isinstance(vector_ev, BaseException):
            all_evidence.extend(vector_ev)
        
        return all_evidence

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

        return all_evidence

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

    async def _mcp_enhanced_evidence(
        self,
        claim: Claim,
        *,
        target_sources: int | None = None,
        tier: EvidenceTier = EvidenceTier.DEFAULT,
    ) -> list[Evidence]:
        """
        MCP-enhanced evidence gathering with fallback.

        1. Query ALL sources (MCP + local) in parallel for speed
        2. Persist MCP evidence to graph for future lookups
        3. Combine and deduplicate evidence
        
        Tier behavior:
        - LOCAL: Skip MCP, only query graph + vector
        - DEFAULT: Normal behavior with early-exit check
        - MAX: Query all MCP sources (no source limits)
        """
        all_evidence: list[Evidence] = []
        tasks = []
        task_names = []

        # Queue local stores first (always run)
        if self._graph_store is not None:
            tasks.append(self._graph_evidence(claim))
            task_names.append("graph")
        if self._vector_store is not None:
            tasks.append(self._vector_evidence(claim))
            task_names.append("vector")

        # LOCAL tier: skip MCP sources entirely
        if tier != EvidenceTier.LOCAL:
            # Select MCP sources based on tier
            mcp_sources = [s for s in self._mcp_sources if s.is_available]
            
            if tier == EvidenceTier.MAX:
                # MAX tier: use ALL available MCP sources
                pass  # Keep all mcp_sources
            elif self._mcp_selector is not None:
                # DEFAULT tier: use intelligent selection with limits
                allow_all = target_sources is None
                selection = await self._mcp_selector.select(
                    claim,
                    max_sources_override=target_sources,
                    allow_all_relevant=allow_all,
                )
                
                mcp_sources = self._mcp_selector.get_sources_for_selection(selection)
                if target_sources is not None and len(mcp_sources) > target_sources:
                    mcp_sources = mcp_sources[:target_sources]
            elif target_sources is not None:
                mcp_sources = mcp_sources[:target_sources]

            # Queue MCP sources
            for source in mcp_sources:
                tasks.append(source.find_evidence(claim))
                task_names.append(f"mcp:{source.source_name}")

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

        return all_evidence

    async def _adaptive_evidence(
        self,
        claim: Claim,
        *,
        target_sources: int | None = None,
        tier: EvidenceTier = EvidenceTier.DEFAULT,
    ) -> list[Evidence]:
        """
        Adaptive evidence gathering with intelligent tiered collection.

        Uses AdaptiveEvidenceCollector for:
        1. Tier 1: Local sources (Neo4j + Qdrant) with early-exit check
        2. Tier 2: Selected MCP sources based on claim domain
        3. Quality-weighted accumulation
        4. Background completion for cache warming

        Tier behavior:
        - LOCAL: Only run Tier 1 (local sources), skip MCP entirely
        - DEFAULT: Normal adaptive behavior (local first, MCP if insufficient)
        - MAX: Query all MCP sources without early-exit

        Falls back to MCP_ENHANCED if collector not configured.
        """
        # LOCAL tier: skip adaptive collector, just use local sources
        if tier == EvidenceTier.LOCAL:
            logger.debug("LOCAL tier: using only local sources (Neo4j + Qdrant)")
            return await self._hybrid_evidence(claim)
        
        if self._evidence_collector is None:
            # Fallback to MCP_ENHANCED if no collector configured
            logger.debug("No evidence collector configured, falling back to MCP_ENHANCED")
            return await self._mcp_enhanced_evidence(claim, tier=tier)

        try:
            total_source_cap = 20 if tier != EvidenceTier.MAX else 100
            local_sources_count = int(self._graph_store is not None) + int(
                self._vector_store is not None
            )
            max_mcp_allowed = max(total_source_cap - local_sources_count, 0)

            # Get MCP sources to query based on claim domain and tier
            mcp_sources = None
            if tier == EvidenceTier.MAX:
                # MAX tier: use ALL available MCP sources
                mcp_sources = [s for s in self._mcp_sources if s.is_available]
                logger.debug(f"MAX tier: using all {len(mcp_sources)} available MCP sources")
            elif self._mcp_selector is not None:
                # DEFAULT tier with selector: use domain-based selection
                allow_all = target_sources is None
                selection = await self._mcp_selector.select(
                    claim,
                    max_sources_override=target_sources,
                    allow_all_relevant=allow_all,
                )
                
                mcp_sources = self._mcp_selector.get_sources_for_selection(selection)
                if max_mcp_allowed and len(mcp_sources) > max_mcp_allowed:
                    mcp_sources = mcp_sources[:max_mcp_allowed]
                logger.debug(
                    f"Claim domain: {selection.domain.value}, "
                    f"selected {len(selection.selected_sources)} MCP sources"
                )
            else:
                # DEFAULT tier without selector: use all available MCP sources (limited)
                mcp_sources = [s for s in self._mcp_sources if s.is_available]
                if target_sources is not None:
                    mcp_sources = mcp_sources[: min(target_sources, max_mcp_allowed)]
                elif max_mcp_allowed:
                    mcp_sources = mcp_sources[:max_mcp_allowed]

            # Collect evidence using adaptive collector
            target_count = None if tier == EvidenceTier.MAX else target_sources
            allow_early_exit = tier != EvidenceTier.MAX
            result = await self._evidence_collector.collect(
                claim,
                mcp_sources,
                target_evidence_count=target_count,
                allow_early_exit=allow_early_exit,
            )


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
            return await self._mcp_enhanced_evidence(claim, tier=tier)

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

    async def _classify_evidence(
        self, claim: Claim, evidence: list[Evidence]
    ) -> tuple[list[Evidence], list[Evidence]]:
        """
        Classify evidence as supporting or refuting.

        Supports three modes:
        1. Two-pass classification (reduce false NEUTRAL)
        2. Single-pass classification (original)
        3. Heuristic fallback (if no LLM available)
        """
        # If we have an LLM provider, use it for classification
        if self._llm_provider is not None and evidence:
            # STRICT MODE: If LLM is available, we MUST use it.
            # We do not fall back to heuristics on error, because the user
            # explicitly requires LLM verification for all evidence.
            # If LLM methods fail, we let the exception propagate so the
            # claim is marked as UNVERIFIABLE rather than potentially
            # incorrect due to weak heuristics.
            try:
                # Choose classification mode based on configuration
                if self._verification_settings.enable_confidence_scoring:
                    return await self._classify_evidence_with_confidence(claim, evidence)
                elif self._verification_settings.enable_two_pass_classification:
                    return await self._classify_evidence_two_pass(claim, evidence)
                else:
                    return await self._classify_evidence_with_llm(claim, evidence)
            except Exception as e:
                logger.error(f"Strict LLM classification failed: {e}")
                raise  # Propagate error to mark claim as UNVERIFIABLE

        # Fallback to heuristic-based classification (only if no LLM configured)
        return self._classify_evidence_heuristic(claim, evidence)

    def _classify_evidence_heuristic(
        self, claim: Claim, evidence: list[Evidence]
    ) -> tuple[list[Evidence], list[Evidence]]:
        """
        Classify evidence using simple heuristics (fallback).
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

    async def _classify_evidence_with_llm(
        self, claim: Claim, evidence: list[Evidence]
    ) -> tuple[list[Evidence], list[Evidence]]:
        """
        Classify evidence as supporting or refuting using LLM.

        This method uses the LLM to semantically understand whether each piece
        of evidence supports or refutes the claim. This is much more accurate
        than heuristic-based classification.
        """
        supporting: list[Evidence] = []
        refuting: list[Evidence] = []
        neutral: list[Evidence] = []

        if not self._llm_provider or not evidence:
            return supporting, refuting

        deduped = self._dedupe_evidence(evidence)

        # Process evidence in smaller batches to avoid token limits
        batch_size = self._verification_settings.classification_batch_size
        for i in range(0, len(deduped), batch_size):
            batch: list[Evidence] = deduped[i : i + batch_size]

            # Build evidence descriptions for the prompt
            evidence_descriptions = []
            for idx, ev in enumerate(batch):
                content_preview = ev.content[:250] if len(ev.content) > 250 else ev.content
                evidence_descriptions.append(
                    f"Evidence {idx + 1} (source: {ev.source.value}):\n{content_preview}"
                )

            evidence_text = "\n\n".join(evidence_descriptions)

            prompt = f"""You are a fact-checking assistant. Your goal is to determine whether each piece of evidence SUPPORTS, REFUTES, or is NEUTRAL toward the claim.

CLAIM: "{claim.text}"

EVIDENCE:
{evidence_text}

CLASSIFICATION GUIDELINES:

✅ SUPPORTS - Use when:
- The evidence directly confirms the claim
- The evidence provides strong contextual support (e.g., related facts that make the claim plausible)
- Minor details differ but core facts align (be PERMISSIVE with peripheral details)
- Example: Claim "Paris is the capital of France" + Evidence "Paris has been France's capital since 1682" → SUPPORTS

❌ REFUTES - Use when:
- The evidence directly contradicts a core fact in the claim
- Key details like locations, dates, names, or numbers conflict
- Example: Claim "Einstein was born in Berlin" + Evidence "Einstein was born in Ulm, Germany" → REFUTES

⚪ NEUTRAL - Use ONLY when:
- The evidence is completely unrelated to the claim
- The evidence neither supports nor refutes (truly ambiguous)
- Example: Claim "Paris is the capital of France" + Evidence "Paris has many museums" → NEUTRAL (unless claim is about culture)

IMPORTANT: 
- Be MORE PERMISSIVE with SUPPORTS - contextual relevance counts
- Be STRICT with REFUTES - only use when facts directly contradict
- Minimize NEUTRAL - if evidence has ANY relevance, classify as SUPPORTS or REFUTES

Respond with valid JSON only:
{{
    "classifications": [
        {{"evidence_index": 1, "classification": "SUPPORTS|REFUTES|NEUTRAL", "reasoning": "brief explanation"}}
    ]
}}
"""

            try:
                from interfaces.llm import LLMMessage

                messages = [
                    LLMMessage(role="system", content="You are a fact-checking assistant that classifies evidence."),
                    LLMMessage(role="user", content=prompt),
                ]
                response = await self._llm_provider.complete(
                    messages=messages,
                    max_tokens=512,
                    temperature=self._verification_settings.classification_temperature,
                )

                # Parse the LLM response
                classifications = self._parse_classification_response(response.content)
                if not classifications:
                    raise ValueError("LLM returned no classifications")

                # Track which evidence items were classified
                classified_indices: set[int] = set()
                
                for classification in classifications:
                    ev_idx = classification.get("evidence_index", 0) - 1
                    if 0 <= ev_idx < len(batch):
                        classified_indices.add(ev_idx)
                        ev = cast(Evidence, batch[ev_idx])
                        label = classification.get("classification", "NEUTRAL").upper()

                        if label == "SUPPORTS":
                            supporting.append(ev)
                            logger.debug(
                                f"LLM classified evidence as SUPPORTING: {ev.content[:100]}..."
                            )
                        elif label == "REFUTES":
                            refuting.append(ev)
                            logger.debug(
                                f"LLM classified evidence as REFUTING: {ev.content[:100]}..."
                            )
                        else:
                            neutral.append(ev)
                
                # Handle unclassified evidence (LLM didn't return classification)
                for ev_idx, ev in enumerate(batch):
                    if ev_idx not in classified_indices:
                        neutral.append(ev)
                        logger.debug(
                            f"Evidence unclassified by LLM, defaulting to NEUTRAL: {ev.content[:80]}..."
                        )

            except Exception as e:
                logger.warning(f"Failed to classify evidence batch with LLM: {e}")
                # Fall back to heuristic for this batch
                batch_supporting, batch_refuting = self._classify_evidence_heuristic(claim, batch)
                supporting.extend(batch_supporting)
                refuting.extend(batch_refuting)

        neutral_before = len(neutral)
        neutral = self._limit_neutral_evidence(neutral, max_items=12)
        total_classified = len(supporting) + len(refuting) + neutral_before
        logger.info(
            f"LLM classification complete: {len(supporting)} supporting, {len(refuting)} refuting, "
            f"{neutral_before} neutral (total: {total_classified}/{len(deduped)} from {len(evidence)} input)"
        )
        if neutral_before > len(neutral):
            logger.info(
                "Trimmed neutral evidence from %d to %d",
                neutral_before,
                len(neutral),
            )
        return supporting, refuting

    async def _classify_evidence_two_pass(
        self, claim: Claim, evidence: list[Evidence]
    ) -> tuple[list[Evidence], list[Evidence]]:
        """
        Two-pass classification to reduce false NEUTRAL classifications.
        
        Pass 1: Classify with relaxed criteria (encourage SUPPORTS/REFUTES)
        Pass 2: Verify SUPPORTS/REFUTES with strict criteria
        
        This reduces cases where relevant evidence is incorrectly marked NEUTRAL.
        """
        if not self._llm_provider or not evidence:
            return [], []
        
        deduped = self._dedupe_evidence(evidence)
        batch_size = self._verification_settings.classification_batch_size
        
        # PASS 1: Relaxed classification (minimize NEUTRAL)
        first_pass_supporting: list[Evidence] = []
        first_pass_refuting: list[Evidence] = []
        first_pass_neutral: list[Evidence] = []
        
        for i in range(0, len(deduped), batch_size):
            batch = deduped[i : i + batch_size]
            
            # Use relaxed temperature (higher = more varied classifications)
            relaxed_temp = min(0.3, self._verification_settings.classification_temperature + 0.2)
            supporting, refuting, neutral = await self._classify_batch(
                claim, batch, temperature=relaxed_temp, relaxed=True
            )
            
            first_pass_supporting.extend(supporting)
            first_pass_refuting.extend(refuting)
            first_pass_neutral.extend(neutral)
        
        # PASS 2: Verify SUPPORTS/REFUTES with strict criteria
        to_verify = first_pass_supporting + first_pass_refuting
        
        if not to_verify:
            logger.info("Two-pass: No evidence to verify (all NEUTRAL in pass 1)")
            return first_pass_supporting, first_pass_refuting
        
        verified_supporting: list[Evidence] = []
        verified_refuting: list[Evidence] = []
        
        for i in range(0, len(to_verify), batch_size):
            batch = to_verify[i : i + batch_size]
            
            # Use original temperature for strict verification
            supporting, refuting, neutral = await self._classify_batch(
                claim, batch, 
                temperature=self._verification_settings.classification_temperature,
                relaxed=False
            )
            
            verified_supporting.extend(supporting)
            verified_refuting.extend(refuting)
            # Items that become NEUTRAL in pass 2 are downgraded
            first_pass_neutral.extend(neutral)
        
        logger.info(
            f"Two-pass classification: Pass 1 ({len(first_pass_supporting)} S, "
            f"{len(first_pass_refuting)} R, {len(first_pass_neutral)} N) → "
            f"Pass 2 ({len(verified_supporting)} S, {len(verified_refuting)} R, "
            f"{len(first_pass_neutral)} N)"
        )
        
        return verified_supporting, verified_refuting

    async def _classify_batch(
        self,
        claim: Claim,
        batch: list[Evidence],
        temperature: float,
        relaxed: bool = True,
    ) -> tuple[list[Evidence], list[Evidence], list[Evidence]]:
        """
        Classify a batch of evidence with specified temperature and criteria.
        
        Args:
            claim: The claim being verified
            batch: Evidence items to classify
            temperature: LLM temperature
            relaxed: If True, use permissive prompt; if False, use strict prompt
        
        Returns:
            Tuple of (supporting, refuting, neutral) evidence lists
        """
        evidence_descriptions = []
        for idx, ev in enumerate(batch):
            content_preview = ev.content[:250] if len(ev.content) > 250 else ev.content
            evidence_descriptions.append(
                f"Evidence {idx + 1} (source: {ev.source.value}):\n{content_preview}"
            )
        
        evidence_text = "\n\n".join(evidence_descriptions)
        
        # Select prompt based on mode
        if relaxed:
            prompt = self._get_relaxed_classification_prompt(claim.text, evidence_text)
        else:
            prompt = self._get_strict_classification_prompt(claim.text, evidence_text)
        
        from interfaces.llm import LLMMessage
        
        messages = [
            LLMMessage(role="system", content="You are a fact-checking assistant that classifies evidence."),
            LLMMessage(role="user", content=prompt),
        ]
        
        response = await self._llm_provider.complete(
            messages=messages,
            max_tokens=512,
            temperature=temperature,
        )
        
        classifications = self._parse_classification_response(response.content)
        
        supporting: list[Evidence] = []
        refuting: list[Evidence] = []
        neutral: list[Evidence] = []
        classified_indices: set[int] = set()
        
        for classification in classifications:
            ev_idx = classification.get("evidence_index", 0) - 1
            if 0 <= ev_idx < len(batch):
                classified_indices.add(ev_idx)
                ev = batch[ev_idx]
                label = classification.get("classification", "NEUTRAL").upper()
                
                if label == "SUPPORTS":
                    supporting.append(ev)
                elif label == "REFUTES":
                    refuting.append(ev)
                else:
                    neutral.append(ev)
        
        # Default unclassified evidence to neutral
        for ev_idx, ev in enumerate(batch):
            if ev_idx not in classified_indices:
                neutral.append(ev)
        
        return supporting, refuting, neutral

    def _get_relaxed_classification_prompt(self, claim_text: str, evidence_text: str) -> str:
        """Get permissive classification prompt (minimize NEUTRAL)."""
        return f"""You are a fact-checking assistant. Classify evidence as SUPPORTS, REFUTES, or NEUTRAL.

CLAIM: "{claim_text}"

EVIDENCE:
{evidence_text}

GUIDELINES (PERMISSIVE MODE):
- SUPPORTS: Evidence confirms OR provides contextual support (be generous)
- REFUTES: Evidence contradicts core facts (strict criteria)
- NEUTRAL: Evidence is completely unrelated (minimize this category)

Respond with valid JSON only:
{{
    "classifications": [
        {{"evidence_index": 1, "classification": "SUPPORTS|REFUTES|NEUTRAL"}}
    ]
}}
"""

    def _get_strict_classification_prompt(self, claim_text: str, evidence_text: str) -> str:
        """Get strict verification prompt (confirm SUPPORTS/REFUTES)."""
        return f"""You are a fact-checking assistant. VERIFY whether evidence truly SUPPORTS or REFUTES the claim.

CLAIM: "{claim_text}"

EVIDENCE:
{evidence_text}

STRICT VERIFICATION:
- SUPPORTS: Only if evidence DIRECTLY confirms the claim
- REFUTES: Only if evidence DIRECTLY contradicts the claim
- NEUTRAL: If evidence is tangentially related or ambiguous

Respond with valid JSON only:
{{
    "classifications": [
        {{"evidence_index": 1, "classification": "SUPPORTS|REFUTES|NEUTRAL"}}
    ]
}}
"""

    async def _classify_evidence_with_confidence(
        self, claim: Claim, evidence: list[Evidence]
    ) -> tuple[list[Evidence], list[Evidence]]:
        """
        Classify evidence using 5-level confidence scale.
        
        Classifications:
        - STRONG_SUPPORT (0.9): Evidence directly confirms claim
        - WEAK_SUPPORT (0.7): Evidence provides contextual support
        - NEUTRAL (0.5): Evidence is unrelated or ambiguous
        - WEAK_REFUTE (0.3): Evidence suggests claim might be false
        - STRONG_REFUTE (0.1): Evidence directly contradicts claim
        
        This enables confidence-weighted scoring in trust computation.
        """
        if not self._llm_provider or not evidence:
            return [], []
        
        from models.results import EvidenceClassification
        
        deduped = self._dedupe_evidence(evidence)
        batch_size = self._verification_settings.classification_batch_size
        
        supporting_with_confidence: list[Evidence] = []
        refuting_with_confidence: list[Evidence] = []
        
        for i in range(0, len(deduped), batch_size):
            batch = deduped[i : i + batch_size]
            
            evidence_descriptions = []
            for idx, ev in enumerate(batch):
                content_preview = ev.content[:250] if len(ev.content) > 250 else ev.content
                evidence_descriptions.append(
                    f"Evidence {idx + 1} (source: {ev.source.value}):\n{content_preview}"
                )
            
            evidence_text = "\n\n".join(evidence_descriptions)
            
            prompt = f"""You are a fact-checking assistant. Classify evidence using a 5-level confidence scale.

CLAIM: "{claim.text}"

EVIDENCE:
{evidence_text}

CLASSIFICATION LEVELS:

🟢 STRONG_SUPPORT (confidence: 0.9)
   - Evidence directly confirms the claim with high certainty
   - All key facts align perfectly

🟡 WEAK_SUPPORT (confidence: 0.7)
   - Evidence provides contextual support
   - Core facts align but peripheral details may differ
   - Evidence makes the claim plausible

⚪ NEUTRAL (confidence: 0.5)
   - Evidence is unrelated or completely ambiguous
   - Cannot determine support or refutation

🟠 WEAK_REFUTE (confidence: 0.3)
   - Evidence suggests the claim might be false
   - Some contradictions but not definitive

🔴 STRONG_REFUTE (confidence: 0.1)
   - Evidence directly contradicts core facts
   - Clearly proves the claim is false

Respond with valid JSON only:
{{
    "classifications": [
        {{"evidence_index": 1, "classification": "STRONG_SUPPORT|WEAK_SUPPORT|NEUTRAL|WEAK_REFUTE|STRONG_REFUTE"}}
    ]
}}
"""
            
            try:
                from interfaces.llm import LLMMessage
                
                messages = [
                    LLMMessage(role="system", content="You classify evidence with confidence levels."),
                    LLMMessage(role="user", content=prompt),
                ]
                
                response = await self._llm_provider.complete(
                    messages=messages,
                    max_tokens=512,
                    temperature=self._verification_settings.classification_temperature,
                )
                
                classifications = self._parse_classification_response(response.content)
                classified_indices: set[int] = set()
                neutral_count = 0
                
                for classification in classifications:
                    ev_idx = classification.get("evidence_index", 0) - 1
                    if 0 <= ev_idx < len(batch):
                        classified_indices.add(ev_idx)
                        original_ev = batch[ev_idx]
                        label = classification.get("classification", "NEUTRAL").upper()
                        
                        try:
                            # Convert string to enum
                            classification_enum = EvidenceClassification[label]
                            confidence = classification_enum.to_confidence()
                            
                            # Create new Evidence with classification_confidence
                            ev_with_confidence = original_ev.model_copy(
                                update={"classification_confidence": confidence}
                            )
                            
                            # Categorize as supporting or refuting
                            if confidence >= 0.6:  # WEAK_SUPPORT or STRONG_SUPPORT
                                supporting_with_confidence.append(ev_with_confidence)
                                logger.debug(
                                    f"LLM: {label} ({confidence}) - {ev_with_confidence.content[:80]}..."
                                )
                            elif confidence <= 0.4:  # WEAK_REFUTE or STRONG_REFUTE
                                refuting_with_confidence.append(ev_with_confidence)
                                logger.debug(
                                    f"LLM: {label} ({confidence}) - {ev_with_confidence.content[:80]}..."
                                )
                            else:
                                # NEUTRAL (0.5) items are not included in either list
                                neutral_count += 1
                            
                        except KeyError:
                            logger.warning(f"Unknown classification label: {label}")
                            continue
                
                # Count unclassified items
                unclassified = len(batch) - len(classified_indices)
                if unclassified > 0:
                    logger.debug(f"Confidence classification: {unclassified} evidence items unclassified by LLM")
            
            except Exception as e:
                logger.warning(f"Failed to classify batch with confidence: {e}")
                # Fallback: use basic classification without confidence
                supporting_batch, refuting_batch, _ = await self._classify_batch(
                    claim, batch, self._verification_settings.classification_temperature, relaxed=True
                )
                supporting_with_confidence.extend(supporting_batch)
                refuting_with_confidence.extend(refuting_batch)
        
        logger.info(
            f"Confidence-weighted classification: {len(supporting_with_confidence)} supporting, "
            f"{len(refuting_with_confidence)} refuting (from {len(deduped)} total)"
        )
        
        return supporting_with_confidence, refuting_with_confidence

    @staticmethod
    def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
        """Remove near-duplicate evidence items by source + content + url."""
        seen: set[str] = set()
        deduped: list[Evidence] = []
        for ev in evidence:
            content_key = " ".join(ev.content.lower().split())[:300]
            key = f"{ev.source.value}|{ev.source_uri or ''}|{content_key}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)
        return deduped

    @staticmethod
    def _limit_neutral_evidence(
        evidence: list[Evidence],
        *,
        max_items: int = 12,
    ) -> list[Evidence]:
        """Limit neutral evidence only (preserve supporting/refuting evidence)."""
        if len(evidence) <= max_items:
            return evidence
        return sorted(
            evidence,
            key=lambda ev: ev.similarity_score if ev.similarity_score is not None else 0.0,
            reverse=True,
        )[:max_items]

    async def _llm_plausibility_fallback(
        self, claim: Claim
    ) -> tuple[VerificationStatus, float, str]:
        """
        Provide a low-weight plausibility estimate when no evidence exists.

        The LLM returns a plausibility score in [0, 1]. We map it to a
        conservative confidence range [0.375, 0.625] to avoid over-trusting.
        """
        if self._llm_provider is None:
            return (
                VerificationStatus.UNVERIFIABLE,
                0.3,
                "No relevant evidence found in knowledge sources.",
            )

        confidence, reason = await self._llm_plausibility_prior(claim)

        reasoning = (
            "No external evidence found. LLM plausibility prior used: "
            f"{confidence:.2f}. {reason}".strip()
        )

        return VerificationStatus.UNCERTAIN, confidence, reasoning

    async def _llm_plausibility_prior(self, claim: Claim) -> tuple[float, str]:
        """
        Get a conservative plausibility prior in [0.375, 0.625] from the LLM.
        """
        if self._llm_provider is None:
            return 0.5, ""

        from interfaces.llm import LLMMessage

        prompt = f"""You are a cautious fact-checking assistant. Without using external sources,
estimate how plausible the following claim is based on general world knowledge.

CLAIM: "{claim.text}"

Respond with valid JSON only in this exact shape:
{{
  "plausibility": 0.0,
  "reason": "short explanation"
}}
"""

        response = await self._llm_provider.complete(
            messages=[
                LLMMessage(role="system", content="You provide conservative plausibility estimates."),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.1,
            max_tokens=256,
        )

        plausibility, reason = self._parse_plausibility_response(response.content)

        # Map plausibility to a conservative confidence band
        confidence = 0.375 + (0.25 * plausibility)
        confidence = max(0.375, min(0.625, confidence))

        return confidence, reason

    def _parse_plausibility_response(self, response: str) -> tuple[float, str]:
        """Parse plausibility JSON response from the LLM."""
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", response, re.IGNORECASE)
            if json_match:
                candidate = json_match.group(1).strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    response = candidate

            response = response.strip()
            data = json.loads(response) if response else {}
            plausibility = float(data.get("plausibility", 0.5))
            reason = str(data.get("reason", ""))
        except Exception as e:
            logger.warning("Failed to parse plausibility response: %s", e)
            plausibility = 0.5
            reason = ""

        plausibility = max(0.0, min(1.0, plausibility))
        return plausibility, reason

    def _parse_classification_response(self, response: str) -> list[dict]:
        """Parse the LLM classification response."""
        try:
            # Handle cases where LLM might wrap JSON in markdown code blocks
            # Allow for unclosed code blocks (truncation) or standard blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", response, re.IGNORECASE)
            if json_match:
                # check if the captured group looks like JSON (starts with { or [)
                # otherwise we might have matched some preamble text
                candidate = json_match.group(1).strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    response = candidate

            response = response.strip()

            # Fast path: full JSON object or array
            if response.startswith("{") or response.startswith("["):
                data = json.loads(response)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    classifications = data.get("classifications", [])
                    return classifications if isinstance(classifications, list) else []

            # Try to find JSON object or array inside the response
            object_match = re.search(r"\{[\s\S]*\}", response)
            array_match = re.search(r"\[[\s\S]*\]", response)
            json_str = None
            if object_match:
                json_str = object_match.group(0)
            elif array_match:
                json_str = array_match.group(0)

            if json_str:
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    classifications = data.get("classifications", [])
                    return classifications if isinstance(classifications, list) else []

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM classification response: {e}")

        return []

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
        
        Supports confidence-weighted evidence when enabled.
        """
        # Check if confidence scoring is enabled and evidence has confidence values
        has_confidence = any(
            ev.classification_confidence is not None 
            for ev in (supporting + refuting)
        )
        
        if self._verification_settings.enable_confidence_scoring and has_confidence:
            return self._determine_status_weighted(claim, supporting, refuting)
        else:
            return self._determine_status_count_based(claim, supporting, refuting)

    def _determine_status_weighted(
        self,
        claim: Claim,
        supporting: list[Evidence],
        refuting: list[Evidence],
    ) -> tuple[VerificationStatus, float, str]:
        """
        Determine status using confidence-weighted evidence.
        
        Instead of counting evidence, we sum confidence scores:
        - Support score = sum of supporting evidence confidences
        - Refute score = sum of refuting evidence confidences
        - Compare weighted scores to determine status
        """
        # Calculate weighted scores
        support_score = sum(
            ev.classification_confidence or 0.0 
            for ev in supporting
        )
        refute_score = sum(
            ev.classification_confidence or 0.0
            for ev in refuting
        )
        
        total_score = support_score + refute_score
        
        if total_score == 0:
            return (
                VerificationStatus.UNVERIFIABLE,
                0.3,
                "No relevant evidence found in knowledge base.",
            )
        
        # Calculate support ratio
        support_ratio = support_score / total_score
        
        # Determine status based on weighted ratio
        if support_ratio >= 0.85:
            # Strong support
            confidence = min(0.95, 0.75 + (support_ratio - 0.85) * 2.0)
            return (
                VerificationStatus.SUPPORTED,
                confidence,
                f"Strong weighted support: score {support_score:.2f} vs {refute_score:.2f} "
                f"({len(supporting)} supporting, {len(refuting)} refuting)",
            )
        elif support_ratio >= 0.65:
            # Good support
            confidence = 0.65 + (support_ratio - 0.65) * 0.5
            return (
                VerificationStatus.PARTIALLY_SUPPORTED,
                confidence,
                f"Good weighted support: score {support_score:.2f} vs {refute_score:.2f}",
            )
        elif support_ratio >= 0.45:
            # Uncertain
            confidence = 0.40 + abs(support_ratio - 0.5) * 0.2
            return (
                VerificationStatus.UNCERTAIN,
                confidence,
                f"Mixed weighted evidence: score {support_score:.2f} vs {refute_score:.2f}",
            )
        elif support_ratio >= 0.25:
            # Weak refutation
            confidence = 0.55 + (0.45 - support_ratio) * 0.5
            return (
                VerificationStatus.UNCERTAIN,
                confidence,
                f"Slightly weighted toward refutation: score {support_score:.2f} vs {refute_score:.2f}",
            )
        else:
            # Strong refutation
            confidence = min(0.90, 0.65 + (0.25 - support_ratio) * 1.0)
            return (
                VerificationStatus.REFUTED,
                confidence,
                f"Strong weighted refutation: score {support_score:.2f} vs {refute_score:.2f} "
                f"({len(supporting)} supporting, {len(refuting)} refuting)",
            )

    def _determine_status_count_based(
        self,
        claim: Claim,
        supporting: list[Evidence],
        refuting: list[Evidence],
    ) -> tuple[VerificationStatus, float, str]:
        """
        Original count-based status determination (backward compatible).

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

    def get_latency_stats(self) -> dict[str, dict[str, Any]]:
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
