"""
Smart MCP Selector
==================

Intelligently selects which MCP tools to query based on claim analysis.
Reduces latency by skipping irrelevant sources.

Uses the ClaimRouter's domain classification to determine which
MCP tools are most likely to provide useful evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from open_hallucination_index.domain.services.claim_router import (
    ClaimDomain,
    RoutingDecision,
    SourceTier,
    get_claim_router,
)
from open_hallucination_index.ports.llm_provider import LLMProvider

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.ports.mcp_source import MCPKnowledgeSource

logger = logging.getLogger(__name__)


@dataclass
class MCPSelection:
    """Selection result for MCP sources."""

    claim_id: str
    domain: ClaimDomain
    selected_sources: list[str]  # Source names to query
    selected_tools: list[str]  # Specific MCP tools to call
    skipped_sources: list[str]  # Sources skipped as irrelevant
    routing_decision: RoutingDecision


class SmartMCPSelector:
    """
    Intelligent MCP source selector.

    Given a claim and available MCP sources, determines which
    sources are worth querying to minimize latency while
    maximizing evidence quality.

    Selection criteria:
    1. Domain relevance (medical claim → PubMed, not GDELT)
    2. Expected latency (prefer faster sources)
    3. Historical success rate (TODO: implement)
    4. Source availability
    """

    def __init__(
        self,
        mcp_sources: list[MCPKnowledgeSource] | None = None,
        max_sources_per_claim: int = 4,
        min_relevance_threshold: float = 0.5,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """
        Initialize the selector.

        Args:
            mcp_sources: Available MCP sources.
            max_sources_per_claim: Maximum sources to query per claim.
            min_relevance_threshold: Minimum relevance score to include source.
            llm_provider: Optional LLM provider for intelligent routing.
        """
        self._mcp_sources = mcp_sources or []
        self._router = get_claim_router()
        self._max_sources = max_sources_per_claim
        self._min_relevance = min_relevance_threshold

        if llm_provider:
            self._router.set_llm_provider(llm_provider)

        # Map source names to MCP source objects
        self._source_map: dict[str, MCPKnowledgeSource] = {}
        for source in self._mcp_sources:
            self._source_map[source.source_name] = source

    def set_mcp_sources(self, sources: list[MCPKnowledgeSource]) -> None:
        """Update available MCP sources."""
        self._mcp_sources = sources
        self._source_map = {s.source_name: s for s in sources}

    async def select(self, claim: Claim, *, max_sources_override: int | None = None) -> MCPSelection:
        """
        Select relevant MCP sources for a claim.

        Args:
            claim: The claim to find sources for.

        Returns:
            MCPSelection with selected and skipped sources.
        """
        # Get routing decision from ClaimRouter
        decision = await self._router.route(claim)
        logger.debug(f"Routing decision: {decision.domain}, confidence={decision.confidence}")
        
        # Filter to MCP sources only (exclude local)
        mcp_recommendations = [
            r
            for r in decision.recommendations
            if r.tier in (SourceTier.MCP_MEDIUM, SourceTier.MCP_SLOW)
        ]

        logger.debug(f"MCP recommendations: {[r.source_name for r in mcp_recommendations]}")

        selected_sources: list[str] = []
        selected_tools: list[str] = []
        skipped_sources: list[str] = []
        max_sources = max_sources_override or self._max_sources

        for rec in mcp_recommendations:
            # Check if source meets relevance threshold
            if rec.relevance_score < self._min_relevance:
                skipped_sources.append(rec.source_name)
                logger.debug(f"Skipping {rec.source_name}: low relevance {rec.relevance_score}")
                continue

            # Check if we've reached max sources
            if len(selected_sources) >= max_sources:
                skipped_sources.append(rec.source_name)
                logger.debug(f"Skipping {rec.source_name}: max sources reached")
                continue

            # Check if source is available
            is_avail = self._is_source_available(rec.source_name)
            logger.debug(f"Source {rec.source_name} available? {is_avail}")
            if not is_avail:
                skipped_sources.append(rec.source_name)
                continue

            selected_sources.append(rec.source_name)
            if rec.mcp_tool:
                selected_tools.append(rec.mcp_tool)

        logger.debug(
            f"Claim '{claim.text[:50]}...' [{decision.domain.value}]: "
            f"selected {len(selected_sources)} sources, skipped {len(skipped_sources)}"
        )

        return MCPSelection(
            claim_id=str(claim.id),
            domain=decision.domain,
            selected_sources=selected_sources,
            selected_tools=selected_tools,
            skipped_sources=skipped_sources,
            routing_decision=decision,
        )

    def select_batch(self, claims: list[Claim]) -> dict[str, MCPSelection]:
        """
        Select sources for multiple claims.

        Returns a mapping of claim_id → MCPSelection.
        """
        return {str(claim.id): self.select(claim) for claim in claims}

    def get_sources_for_selection(self, selection: MCPSelection) -> list[MCPKnowledgeSource]:
        """
        Get actual MCP source objects for a selection.

        Args:
            selection: The selection result.

        Returns:
            List of MCPKnowledgeSource objects to query.
        """
        sources: list[MCPKnowledgeSource] = []
        for source_name in selection.selected_sources:
            if source_name in self._source_map:
                sources.append(self._source_map[source_name])
            else:
                # Try partial matching (e.g., "ohi_wikipedia" → "ohi")
                for key, source in self._source_map.items():
                    if key in source_name or source_name in key:
                        sources.append(source)
                        break
        if not sources and self._mcp_sources:
            # Fallback: use any available MCP sources (e.g., unified OHI MCP adapter)
            logger.debug(
                "No MCP source name matched selection; falling back to available MCP sources."
            )
            sources = list(self._mcp_sources)
        return sources

    def _is_source_available(self, source_name: str) -> bool:
        """Check if a source is available and healthy."""
        # First check direct match
        if source_name in self._source_map:
            return self._source_map[source_name].is_available

        # Check if any source name contains this or vice versa
        for key, source in self._source_map.items():
            if key in source_name or source_name in key:
                return source.is_available

        # If not in our map, assume it might be available via OHI MCP
        return True


# Domain → preferred tools mapping for quick lookup
DOMAIN_TOOLS: dict[ClaimDomain, list[str]] = {
    ClaimDomain.MEDICAL: [
        "search_pubmed",
        "search_clinical_trials",
        "search_europepmc",
        "search_wikipedia",
    ],
    ClaimDomain.ACADEMIC: [
        "search_openalex",
        "search_crossref",
        "search_europepmc",
        "search_wikipedia",
    ],
    ClaimDomain.NEWS: [
        "search_wikipedia",
        "search_wikidata",
    ],
    ClaimDomain.TECHNICAL: [
        "resolve-library-id",
        "query-docs",
        "search_wikipedia",
        "search_wikidata",
    ],
    ClaimDomain.ECONOMIC: [
        "search_wikipedia",
        "search_wikidata",
        "search_openalex",
    ],
    ClaimDomain.SECURITY: [
        "resolve-library-id",
        "query-docs",
        "search_wikipedia",
    ],
    ClaimDomain.GENERAL: [
        "search_wikipedia",
        "search_wikidata",
        "search_dbpedia",
        "search_all",
    ],
}


def get_tools_for_domain(domain: ClaimDomain, max_tools: int = 4) -> list[str]:
    """Get preferred MCP tools for a domain."""
    tools = DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS[ClaimDomain.GENERAL])
    return tools[:max_tools]
