"""
OHI MCP Adapter
===============

Unified adapter for the OHI MCP server that aggregates 15+ knowledge sources.
Uses direct HTTP API for simple request-response pattern (no SSE overhead).

This adapter replaces the separate mcp_wikipedia.py and mcp_context7.py adapters.

Supported knowledge sources:
- Wikipedia cluster: Wikipedia, Wikidata (SPARQL), DBpedia (SPARQL)
- Context7: Library/API documentation for technical claims
- Academic: OpenAlex, Crossref, Europe PMC, OpenCitations
- Medical: PubMed/NCBI, ClinicalTrials.gov
- News: GDELT
- Economic: World Bank
- Security: OSV (Open Source Vulnerabilities)

Usage:
    adapter = OHIMCPAdapter(settings)
    await adapter.connect()

    # Search across all sources
    evidence = await adapter.find_evidence(claim)

    # Use specific source methods
    wiki_results = await adapter.search_wikipedia("Python programming")
    docs = await adapter.get_library_documentation("fastapi", "how to create routes")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from open_hallucination_index.adapters.outbound.mcp_http_base import HTTPMCPAdapter
from open_hallucination_index.domain.entities import Evidence, EvidenceSource
from open_hallucination_index.ports.mcp_source import MCPQueryError

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.infrastructure.config import MCPSettings

logger = logging.getLogger(__name__)


# Map tool names to evidence sources
TOOL_TO_SOURCE: dict[str, EvidenceSource] = {
    # Wikipedia / Wikidata / DBpedia
    "search_wikipedia": EvidenceSource.WIKIPEDIA,
    "get_wikipedia_summary": EvidenceSource.WIKIPEDIA,
    "get_summary": EvidenceSource.WIKIPEDIA,
    "search_wikidata": EvidenceSource.WIKIPEDIA,
    "query_wikidata_sparql": EvidenceSource.WIKIPEDIA,
    "search_dbpedia": EvidenceSource.WIKIPEDIA,
    # Context7 (library documentation)
    "resolve-library-id": EvidenceSource.MCP_CONTEXT7,
    "query-docs": EvidenceSource.MCP_CONTEXT7,
    # Academic sources
    "search_academic": EvidenceSource.ACADEMIC,
    "search_openalex": EvidenceSource.ACADEMIC,
    "search_crossref": EvidenceSource.ACADEMIC,
    "get_doi_metadata": EvidenceSource.ACADEMIC,
    "search_pubmed": EvidenceSource.PUBMED,
    "search_europepmc": EvidenceSource.ACADEMIC,
    "search_clinical_trials": EvidenceSource.CLINICAL_TRIALS,
    "get_citations": EvidenceSource.ACADEMIC,
    # Other sources
    "search_gdelt": EvidenceSource.NEWS,
    "get_world_bank_indicator": EvidenceSource.WORLD_BANK,
    "search_vulnerabilities": EvidenceSource.OSV,
    "get_vulnerability": EvidenceSource.OSV,
    "search_all": EvidenceSource.KNOWLEDGE_GRAPH,
}


class OHIMCPAdapter(HTTPMCPAdapter):
    """
    Adapter for the unified OHI MCP server.

    Uses direct HTTP API for simple request-response pattern.
    Aggregates 13+ knowledge sources into a single interface.
    """

    def __init__(self, settings: MCPSettings) -> None:
        """
        Initialize the OHI MCP adapter.

        Args:
            settings: MCP configuration with OHI MCP URL.
        """
        super().__init__(
            base_url=settings.ohi_url,
            timeout=30.0,
            connect_timeout=10.0,
            max_connections=1,
            max_keepalive=1,
        )
        self._settings = settings

    @property
    def source_name(self) -> str:
        """Human-readable name of this MCP source."""
        return "OHI Knowledge Sources"

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Query the MCP server for evidence related to a claim.

        Uses search_all for comprehensive results across all sources.

        Args:
            claim: The claim to find evidence for.

        Returns:
            List of Evidence objects from multiple sources.
        """
        return await self.search_for_evidence(claim, max_results=5, search_type="all")


class TargetedOHISource(OHIMCPAdapter):
    """
    Targeted adapter for a specific OHI MCP source.
    Allows routing to specific knowledge domains (e.g. PubMed, Wiki).
    """

    def __init__(self, settings: MCPSettings, search_type: str, source_name: str) -> None:
        super().__init__(settings)
        self._search_type = search_type
        self._source_name_override = source_name

    @property
    def source_name(self) -> str:
        return self._source_name_override

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        return await self.search_for_evidence(claim, max_results=5, search_type=self._search_type)

    async def search_for_evidence(
        self,
        claim: Claim,
        *,
        max_results: int = 5,
        search_type: str = "all",
    ) -> list[Evidence]:
        """
        Search for evidence to support or refute a claim.

        Uses the unified search_all tool by default, or specific tools
        based on search_type.

        Args:
            claim: The claim to find evidence for.
            max_results: Maximum number of results to return.
            search_type: Type of search - "all", "wikipedia", "academic",
                "medical", "news", "economic", "security".

        Returns:
            List of Evidence objects from multiple sources.
        """
        if not self._available:
            raise MCPQueryError("OHI MCP adapter not connected")

        # Map search type to appropriate tool
        tool_map = {
            "all": ("search_all", {"query": claim.text, "limit": max_results}),
            # Wikipedia cluster
            "wikipedia": ("search_wikipedia", {"query": claim.text, "limit": max_results}),
            "wikidata": ("search_wikidata", {"query": claim.text, "limit": max_results}),
            "dbpedia": ("search_dbpedia", {"query": claim.text, "limit": max_results}),
            # Academic cluster
            "academic": ("search_academic", {"query": claim.text, "limit": max_results}),
            "openalex": ("search_openalex", {"query": claim.text, "limit": max_results}),
            "crossref": ("search_crossref", {"query": claim.text, "limit": max_results}),
            "europepmc": ("search_europepmc", {"query": claim.text, "limit": max_results}),
            # Medical cluster
            "pubmed": ("search_pubmed", {"query": claim.text, "limit": max_results}),
            "clinical_trials": ("search_clinical_trials", {"query": claim.text, "limit": max_results}),
            # Other sources
            "gdelt": ("search_gdelt", {"query": claim.text, "limit": max_results}),
            "worldbank": ("get_world_bank_indicator", {"indicator": claim.text}),
            "osv": ("search_vulnerabilities", {"query": claim.text}),
            "context7": ("query-docs", {"query": claim.text, "libraryId": "/general/tech"}), # Placeholder lib ID
        }

        tool_name, arguments = tool_map.get(
            search_type,
            ("search_all", {"query": claim.text, "limit": max_results}),
        )

        try:
            results = await self.call_tool(tool_name, arguments)
            source = TOOL_TO_SOURCE.get(tool_name, EvidenceSource.KNOWLEDGE_GRAPH)

            evidence_list = [
                self._create_evidence(result, source, tool_name) for result in results[:max_results]
            ]

            logger.info(
                f"OHI MCP search for '{claim.text[:50]}...' "
                f"returned {len(evidence_list)} results via {tool_name}"
            )
            return evidence_list

        except MCPQueryError:
            raise
        except Exception as e:
            logger.error(f"OHI MCP search failed: {e}")
            raise MCPQueryError(f"Search failed: {e}") from e

    # =========================================================================
    # Convenience methods for specific source types
    # =========================================================================

    async def get_article_summary(self, title: str) -> Evidence | None:
        """Get Wikipedia article summary by title."""
        if not self._available:
            return None
        try:
            results = await self.call_tool("get_wikipedia_summary", {"title": title})
            if results:
                return self._create_evidence(
                    results[0], EvidenceSource.WIKIPEDIA, "get_wikipedia_summary"
                )
            return None
        except Exception as e:
            logger.warning(f"Failed to get Wikipedia summary for '{title}': {e}")
            return None

    async def get_doi_metadata(self, doi: str) -> Evidence | None:
        """Get metadata for a DOI."""
        if not self._available:
            return None
        try:
            results = await self.call_tool("get_doi_metadata", {"doi": doi})
            if results:
                ev = self._create_evidence(results[0], EvidenceSource.ACADEMIC, "get_doi_metadata")
                ev.source_uri = f"https://doi.org/{doi}"
                return ev
            return None
        except Exception as e:
            logger.warning(f"Failed to get DOI metadata for '{doi}': {e}")
            return None

    async def search_pubmed(self, query: str, *, max_results: int = 10) -> list[Evidence]:
        """Search PubMed for medical/scientific literature."""
        if not self._available:
            return []
        try:
            results = await self.call_tool("search_pubmed", {"query": query, "limit": max_results})
            return [
                self._create_evidence(r, EvidenceSource.PUBMED, "search_pubmed")
                for r in results[:max_results]
            ]
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
            return []

    async def search_clinical_trials(self, query: str, *, max_results: int = 10) -> list[Evidence]:
        """Search ClinicalTrials.gov for clinical studies."""
        if not self._available:
            return []
        try:
            results = await self.call_tool(
                "search_clinical_trials", {"query": query, "limit": max_results}
            )
            return [
                self._create_evidence(r, EvidenceSource.CLINICAL_TRIALS, "search_clinical_trials")
                for r in results[:max_results]
            ]
        except Exception as e:
            logger.warning(f"ClinicalTrials search failed: {e}")
            return []

    async def search_news(self, query: str, *, max_results: int = 10) -> list[Evidence]:
        """Search GDELT for global news."""
        if not self._available:
            return []
        try:
            results = await self.call_tool("search_gdelt", {"query": query, "limit": max_results})
            return [
                self._create_evidence(r, EvidenceSource.NEWS, "search_gdelt")
                for r in results[:max_results]
            ]
        except Exception as e:
            logger.warning(f"GDELT news search failed: {e}")
            return []

    async def get_economic_indicator(self, indicator: str, country: str = "all") -> list[Evidence]:
        """Get World Bank economic indicators."""
        if not self._available:
            return []
        try:
            results = await self.call_tool(
                "get_world_bank_indicator", {"indicator": indicator, "country": country}
            )
            return [
                self._create_evidence(r, EvidenceSource.WORLD_BANK, "get_world_bank_indicator")
                for r in results
            ]
        except Exception as e:
            logger.warning(f"World Bank indicator query failed: {e}")
            return []

    async def search_vulnerabilities(
        self, package: str, ecosystem: str | None = None
    ) -> list[Evidence]:
        """Search for security vulnerabilities in OSV database."""
        if not self._available:
            return []
        try:
            args: dict[str, Any] = {"query": package}
            if ecosystem:
                args["ecosystem"] = ecosystem
            results = await self.call_tool("search_vulnerabilities", args)
            return [
                self._create_evidence(r, EvidenceSource.OSV, "search_vulnerabilities")
                for r in results
            ]
        except Exception as e:
            logger.warning(f"OSV vulnerability search failed: {e}")
            return []

    # =========================================================================
    # Wikipedia / Wikidata / DBpedia methods
    # =========================================================================

    async def search_wikipedia(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        """Search Wikipedia for articles matching query."""
        if not self._available:
            return []
        try:
            results = await self.call_tool(
                "search_wikipedia", {"query": query, "limit": max_results}
            )
            return [
                self._create_evidence(r, EvidenceSource.WIKIPEDIA, "search_wikipedia")
                for r in results[:max_results]
            ]
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
            return []

    async def search_wikidata(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        """Search Wikidata for entities and structured facts."""
        if not self._available:
            return []
        try:
            results = await self.call_tool(
                "search_wikidata", {"query": query, "limit": max_results}
            )
            return [
                self._create_evidence(r, EvidenceSource.WIKIPEDIA, "search_wikidata")
                for r in results[:max_results]
            ]
        except Exception as e:
            logger.warning(f"Wikidata search failed: {e}")
            return []

    async def query_wikidata_sparql(self, sparql: str) -> list[Evidence]:
        """Execute a SPARQL query against Wikidata."""
        if not self._available:
            return []
        try:
            results = await self.call_tool("query_wikidata_sparql", {"sparql": sparql})
            return [
                self._create_evidence(r, EvidenceSource.WIKIPEDIA, "query_wikidata_sparql")
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Wikidata SPARQL query failed: {e}")
            return []

    async def search_dbpedia(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        """Search DBpedia for structured Wikipedia data."""
        if not self._available:
            return []
        try:
            results = await self.call_tool("search_dbpedia", {"query": query, "limit": max_results})
            return [
                self._create_evidence(r, EvidenceSource.WIKIPEDIA, "search_dbpedia")
                for r in results[:max_results]
            ]
        except Exception as e:
            logger.warning(f"DBpedia search failed: {e}")
            return []

    # =========================================================================
    # Context7 (Library Documentation) methods
    # =========================================================================

    async def resolve_library_id(self, library_name: str, query: str = "") -> str | None:
        """
        Resolve a library/package name to a Context7-compatible ID.

        Args:
            library_name: Name of the library (e.g., "react", "fastapi").
            query: Optional context query for better matching.

        Returns:
            Context7 library ID (e.g., "/facebook/react") or None.
        """
        if not self._available:
            return None
        try:
            results = await self.call_tool(
                "resolve-library-id",
                {"libraryName": library_name, "query": query or library_name},
            )
            if results and len(results) > 0:
                # Extract library ID from result
                result = results[0]
                if isinstance(result, dict) and "library_id" in result:
                    return result["library_id"]
                if isinstance(result, dict) and "text" in result:
                    # Parse from text response
                    import re

                    match = re.search(r"(/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)", result["text"])
                    if match:
                        return match.group(1)
                if isinstance(result, str):
                    import re

                    match = re.search(r"(/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)", result)
                    if match:
                        return match.group(1)
            return None
        except Exception as e:
            logger.warning(f"Context7 resolve-library-id failed for '{library_name}': {e}")
            return None

    async def query_library_docs(
        self, library_id: str, query: str, *, max_results: int = 3
    ) -> list[Evidence]:
        """
        Query Context7 documentation for a library.

        Args:
            library_id: Context7 library ID (e.g., "/facebook/react").
            query: Natural language query about the library.
            max_results: Maximum number of results.

        Returns:
            List of Evidence objects from library documentation.
        """
        if not self._available:
            return []
        try:
            results = await self.call_tool("query-docs", {"libraryId": library_id, "query": query})
            evidences = [
                self._create_evidence(r, EvidenceSource.MCP_CONTEXT7, "query-docs")
                for r in results[:max_results]
            ]
            # Enrich with library metadata
            for ev in evidences:
                if ev.structured_data is None:
                    ev.structured_data = {}
                ev.structured_data["library_id"] = library_id
                ev.source_uri = f"https://context7.com{library_id}"
            return evidences
        except Exception as e:
            logger.warning(f"Context7 query-docs failed for '{library_id}': {e}")
            return []

    async def get_library_documentation(
        self, library_name: str, query: str, *, max_results: int = 3
    ) -> list[Evidence]:
        """
        High-level method to get library documentation.

        Resolves the library ID and queries documentation in one call.

        Args:
            library_name: Name of the library (e.g., "react", "fastapi").
            query: Natural language query about the library.
            max_results: Maximum number of results.

        Returns:
            List of Evidence objects from library documentation.
        """
        library_id = await self.resolve_library_id(library_name, query)
        if not library_id:
            logger.debug(f"Could not resolve library ID for: {library_name}")
            return []
        return await self.query_library_docs(library_id, query, max_results=max_results)
