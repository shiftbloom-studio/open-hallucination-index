"""
OHI MCP Adapter
===============

Adapter for the unified OHI MCP server that aggregates 13+ knowledge sources.
Uses MCP SDK with SSE transport to communicate with ohi-mcp-server container.

Supported knowledge sources:
- Wikipedia (Wikidata, MediaWiki, Wikimedia REST, DBpedia)
- Academic (OpenAlex, Crossref, Europe PMC, OpenCitations)
- Medical (PubMed/NCBI, ClinicalTrials.gov)
- News (GDELT)
- Economic (World Bank)
- Security (OSV)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent

from open_hallucination_index.adapters.outbound.mcp_session_pool import (
    MCPSessionPool,
    MCPTransportType,
    PoolConfig,
)
from open_hallucination_index.domain.entities import Evidence, EvidenceSource
from open_hallucination_index.ports.mcp_source import (
    MCPConnectionError,
    MCPKnowledgeSource,
    MCPQueryError,
)

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.infrastructure.config import MCPSettings

logger = logging.getLogger(__name__)


# Map tool results to evidence sources
TOOL_TO_SOURCE: dict[str, EvidenceSource] = {
    "search_wikipedia": EvidenceSource.WIKIPEDIA,
    "get_wikipedia_summary": EvidenceSource.WIKIPEDIA,
    "search_wikidata": EvidenceSource.WIKIPEDIA,
    "query_wikidata_sparql": EvidenceSource.WIKIPEDIA,
    "search_dbpedia": EvidenceSource.WIKIPEDIA,
    "search_academic": EvidenceSource.ACADEMIC,
    "search_openalex": EvidenceSource.ACADEMIC,
    "search_crossref": EvidenceSource.ACADEMIC,
    "get_doi_metadata": EvidenceSource.ACADEMIC,
    "search_pubmed": EvidenceSource.PUBMED,
    "search_europepmc": EvidenceSource.ACADEMIC,
    "search_clinical_trials": EvidenceSource.CLINICAL_TRIALS,
    "get_citations": EvidenceSource.ACADEMIC,
    "search_gdelt": EvidenceSource.NEWS,
    "get_world_bank_indicator": EvidenceSource.WORLD_BANK,
    "search_vulnerabilities": EvidenceSource.OSV,
    "get_vulnerability": EvidenceSource.OSV,
    "search_all": EvidenceSource.KNOWLEDGE_GRAPH,
}


class OHIMCPAdapter(MCPKnowledgeSource):
    """
    Adapter for the unified OHI MCP server.

    Uses MCP SDK with SSE transport to communicate with the
    ohi-mcp-server container. Provides unified access to 13+ knowledge sources.
    """

    def __init__(self, settings: MCPSettings) -> None:
        """
        Initialize the OHI MCP adapter.

        Args:
            settings: MCP configuration with OHI MCP URL.
        """
        self._settings = settings
        self._base_url = settings.ohi_url.rstrip("/")
        # Ensure /sse endpoint for SSE transport
        if not self._base_url.endswith("/sse"):
            self._mcp_url = f"{self._base_url}/sse"
        else:
            self._mcp_url = self._base_url
        self._available = False
        self._tools: list[str] = []
        self._pool: MCPSessionPool | None = None
        self._use_pool = True

    @property
    def source_name(self) -> str:
        """Human-readable name of this MCP source."""
        return "OHI Knowledge Sources"

    @property
    def is_available(self) -> bool:
        """Whether the MCP server is currently reachable."""
        return self._available

    async def connect(self) -> None:
        """
        Test connection and list available tools.

        Initializes the session pool for persistent SSE connections.
        """
        try:
            if self._use_pool:
                self._pool = MCPSessionPool(
                    source_name="OHI-MCP",
                    mcp_url=self._mcp_url,
                    transport_type=MCPTransportType.SSE,
                    config=PoolConfig(
                        min_sessions=2,
                        max_sessions=5,
                        session_ttl_seconds=300.0,
                        idle_timeout_seconds=60.0,
                        health_check_interval_seconds=30.0,
                    ),
                )
                await self._pool.initialize()

                async with self._pool.acquire() as session:
                    tools = await session.list_tools()
                    self._tools = [t.name for t in tools.tools]
            else:
                async with self._session_fallback() as session:
                    tools = await session.list_tools()
                    self._tools = [t.name for t in tools.tools]

            self._available = True
            logger.info(f"Connected to OHI MCP at {self._mcp_url}")
            logger.info(f"Available tools ({len(self._tools)}): {self._tools}")
            logger.info(f"Session pooling: {'enabled' if self._pool else 'disabled'}")

        except Exception as e:
            self._available = False
            logger.error(f"OHI MCP connection failed: {e}")
            raise MCPConnectionError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect and shutdown the session pool."""
        if self._pool:
            await self._pool.shutdown()
            self._pool = None

        self._available = False
        self._tools = []
        logger.info("Disconnected from OHI MCP")

    async def health_check(self) -> bool:
        """Check if OHI MCP is responding."""
        try:
            if self._pool and self._pool.is_healthy:
                async with self._pool.acquire() as session:
                    await session.list_tools()
                    return True
            else:
                async with self._session_fallback() as session:
                    await session.list_tools()
                    return True
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Query the MCP server for evidence related to a claim.

        This is the main method called by VerificationOracle.
        Uses search_all for comprehensive results across all sources.

        Args:
            claim: The claim to find evidence for.

        Returns:
            List of Evidence objects from multiple sources.
        """
        return await self.search_for_evidence(claim, max_results=5, search_type="all")

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform a general search on this MCP source.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of result dictionaries.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                results = await self._call_tool(
                    session, "search_all", {"query": query, "limit": limit}
                )
                return results
        except Exception as e:
            logger.warning(f"OHI MCP search failed: {e}")
            return []

    @asynccontextmanager
    async def _session(self):
        """Get an MCP session (from pool or create new)."""
        if self._pool and self._pool.is_healthy:
            async with self._pool.acquire() as session:
                yield session
        else:
            async with self._session_fallback() as session:
                yield session

    @asynccontextmanager
    async def _session_fallback(self):
        """Create a new MCP session (non-pooled, for fallback)."""
        async with sse_client(self._mcp_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def get_pool_stats(self) -> dict[str, Any] | None:
        """Get session pool statistics."""
        if self._pool:
            return self._pool.get_stats()
        return None

    async def _call_tool(
        self, session: ClientSession, tool_name: str, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Call an MCP tool and parse the results.

        Args:
            session: Active MCP client session.
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            List of result dictionaries.
        """
        result = await session.call_tool(tool_name, arguments)

        results = []
        for content in result.content:
            if isinstance(content, TextContent):
                try:
                    import json
                    data = json.loads(content.text)
                    if isinstance(data, dict) and "results" in data:
                        results.extend(data["results"])
                    elif isinstance(data, list):
                        results.extend(data)
                    else:
                        results.append(data)
                except json.JSONDecodeError:
                    results.append({"text": content.text})

        return results

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

        try:
            async with self._session() as session:
                # Map search type to appropriate tool
                tool_map = {
                    "all": ("search_all", {"query": claim.text, "limit": max_results}),
                    "wikipedia": ("search_wikipedia", {"query": claim.text, "limit": max_results}),
                    "academic": ("search_academic", {"query": claim.text, "limit": max_results}),
                    "pubmed": ("search_pubmed", {"query": claim.text, "limit": max_results}),
                    "medical": ("search_clinical_trials", {"query": claim.text, "limit": max_results}),
                    "news": ("search_gdelt", {"query": claim.text, "limit": max_results}),
                    "economic": ("get_world_bank_indicator", {"indicator": claim.text}),
                    "security": ("search_vulnerabilities", {"package": claim.text}),
                }

                tool_name, arguments = tool_map.get(
                    search_type, 
                    ("search_all", {"query": claim.text, "limit": max_results})
                )

                results = await self._call_tool(session, tool_name, arguments)

                evidence_list = []
                for result in results[:max_results]:
                    source = TOOL_TO_SOURCE.get(tool_name, EvidenceSource.KNOWLEDGE_GRAPH)
                    
                    evidence = Evidence(
                        id=uuid4(),
                        source=source,
                        content=result.get("content", result.get("title", "")),
                        similarity_score=result.get("score", 0.8),
                        source_uri=result.get("url"),
                        retrieved_at=datetime.now(),
                        structured_data={
                            "tool": tool_name,
                            "original_source": result.get("source", "unknown"),
                            **result.get("metadata", {}),
                        },
                    )
                    evidence_list.append(evidence)

                logger.info(
                    f"OHI MCP search for '{claim.text[:50]}...' "
                    f"returned {len(evidence_list)} results via {tool_name}"
                )
                return evidence_list

        except Exception as e:
            logger.error(f"OHI MCP search failed: {e}")
            raise MCPQueryError(f"Search failed: {e}") from e

    async def get_article_summary(self, title: str) -> Evidence | None:
        """
        Get Wikipedia article summary by title.

        Args:
            title: Wikipedia article title.

        Returns:
            Evidence with article summary, or None if not found.
        """
        if not self._available:
            return None

        try:
            async with self._session() as session:
                results = await self._call_tool(
                    session, "get_wikipedia_summary", {"title": title}
                )

                if results:
                    result = results[0]
                    return Evidence(
                        id=uuid4(),
                        source=EvidenceSource.WIKIPEDIA,
                        content=result.get("content", ""),
                        similarity_score=1.0,
                        source_uri=result.get("url"),
                        retrieved_at=datetime.now(),
                        structured_data={"title": title, **result.get("metadata", {})},
                    )
                return None

        except Exception as e:
            logger.warning(f"Failed to get Wikipedia summary for '{title}': {e}")
            return None

    async def get_doi_metadata(self, doi: str) -> Evidence | None:
        """
        Get metadata for a DOI.

        Args:
            doi: Digital Object Identifier.

        Returns:
            Evidence with DOI metadata, or None if not found.
        """
        if not self._available:
            return None

        try:
            async with self._session() as session:
                results = await self._call_tool(
                    session, "get_doi_metadata", {"doi": doi}
                )

                if results:
                    result = results[0]
                    return Evidence(
                        id=uuid4(),
                        source=EvidenceSource.ACADEMIC,
                        content=result.get("content", ""),
                        similarity_score=1.0,
                        source_uri=f"https://doi.org/{doi}",
                        retrieved_at=datetime.now(),
                        structured_data={"doi": doi, **result.get("metadata", {})},
                    )
                return None

        except Exception as e:
            logger.warning(f"Failed to get DOI metadata for '{doi}': {e}")
            return None

    async def search_pubmed(
        self, query: str, *, max_results: int = 10
    ) -> list[Evidence]:
        """
        Search PubMed for medical/scientific literature.

        Args:
            query: Search query.
            max_results: Maximum results to return.

        Returns:
            List of Evidence from PubMed.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                results = await self._call_tool(
                    session, "search_pubmed", {"query": query, "limit": max_results}
                )

                return [
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.PUBMED,
                        content=r.get("content", r.get("title", "")),
                        similarity_score=r.get("score", 0.8),
                        source_uri=r.get("url"),
                        retrieved_at=datetime.now(),
                        structured_data=r.get("metadata", {}),
                    )
                    for r in results[:max_results]
                ]

        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
            return []

    async def search_clinical_trials(
        self, query: str, *, max_results: int = 10
    ) -> list[Evidence]:
        """
        Search ClinicalTrials.gov for clinical studies.

        Args:
            query: Search query.
            max_results: Maximum results to return.

        Returns:
            List of Evidence from ClinicalTrials.gov.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                results = await self._call_tool(
                    session,
                    "search_clinical_trials",
                    {"query": query, "limit": max_results},
                )

                return [
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.CLINICAL_TRIALS,
                        content=r.get("content", r.get("title", "")),
                        similarity_score=r.get("score", 0.8),
                        source_uri=r.get("url"),
                        retrieved_at=datetime.now(),
                        structured_data=r.get("metadata", {}),
                    )
                    for r in results[:max_results]
                ]

        except Exception as e:
            logger.warning(f"ClinicalTrials search failed: {e}")
            return []

    async def search_news(
        self, query: str, *, max_results: int = 10
    ) -> list[Evidence]:
        """
        Search GDELT for global news.

        Args:
            query: Search query.
            max_results: Maximum results to return.

        Returns:
            List of Evidence from GDELT news.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                results = await self._call_tool(
                    session, "search_gdelt", {"query": query, "limit": max_results}
                )

                return [
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.NEWS,
                        content=r.get("content", r.get("title", "")),
                        similarity_score=r.get("score", 0.7),
                        source_uri=r.get("url"),
                        retrieved_at=datetime.now(),
                        structured_data=r.get("metadata", {}),
                    )
                    for r in results[:max_results]
                ]

        except Exception as e:
            logger.warning(f"GDELT news search failed: {e}")
            return []

    async def get_economic_indicator(
        self, indicator: str, country: str = "all"
    ) -> list[Evidence]:
        """
        Get World Bank economic indicators.

        Args:
            indicator: World Bank indicator code (e.g., "NY.GDP.MKTP.CD").
            country: Country code or "all".

        Returns:
            List of Evidence with indicator data.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                results = await self._call_tool(
                    session,
                    "get_world_bank_indicator",
                    {"indicator": indicator, "country": country},
                )

                return [
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.WORLD_BANK,
                        content=r.get("content", r.get("title", "")),
                        similarity_score=1.0,
                        source_uri=r.get("url"),
                        retrieved_at=datetime.now(),
                        structured_data={"indicator": indicator, **r.get("metadata", {})},
                    )
                    for r in results
                ]

        except Exception as e:
            logger.warning(f"World Bank indicator query failed: {e}")
            return []

    async def search_vulnerabilities(
        self, package: str, ecosystem: str | None = None
    ) -> list[Evidence]:
        """
        Search for security vulnerabilities in OSV database.

        Args:
            package: Package name to search for.
            ecosystem: Optional package ecosystem (npm, pypi, etc.).

        Returns:
            List of Evidence with vulnerability information.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                args: dict[str, Any] = {"package": package}
                if ecosystem:
                    args["ecosystem"] = ecosystem

                results = await self._call_tool(
                    session, "search_vulnerabilities", args
                )

                return [
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.OSV,
                        content=r.get("content", r.get("title", "")),
                        similarity_score=1.0,
                        source_uri=r.get("url"),
                        retrieved_at=datetime.now(),
                        structured_data={"package": package, **r.get("metadata", {})},
                    )
                    for r in results
                ]

        except Exception as e:
            logger.warning(f"OSV vulnerability search failed: {e}")
            return []
