"""
Wikipedia MCP Adapter
=====================

Adapter for Wikipedia knowledge retrieval via MCP server.
Connects to the configured MCP endpoint using SSE transport.

Now supports session pooling for persistent SSE connections to improve performance.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
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
)

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.infrastructure.config import MCPSettings

logger = logging.getLogger(__name__)


class WikipediaMCPAdapter(MCPKnowledgeSource):
    """
    Adapter for Wikipedia MCP server.

    Uses MCP SDK with SSE transport to communicate with
    the configured MCP endpoint. Provides evidence from English Wikipedia.
    """

    def __init__(self, settings: MCPSettings) -> None:
        """
        Initialize the Wikipedia MCP adapter.

        Args:
            settings: MCP configuration with Wikipedia URL.
        """
        self._settings = settings
        self._base_url = settings.wikipedia_url.rstrip("/")
        # Ensure /sse endpoint for SSE transport
        if not self._base_url.endswith("/sse"):
            self._mcp_url = f"{self._base_url}/sse"
        else:
            self._mcp_url = self._base_url
        self._available = False
        self._tools: list[str] = []

        # Session pool for persistent connections
        self._pool: MCPSessionPool | None = None
        self._use_pool = True  # Enable session pooling by default

    @property
    def source_name(self) -> str:
        """Human-readable name of this MCP source."""
        return "Wikipedia"

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
            # Initialize session pool for persistent connections
            if self._use_pool:
                self._pool = MCPSessionPool(
                    source_name="Wikipedia",
                    mcp_url=self._mcp_url,
                    transport_type=MCPTransportType.SSE,
                    config=PoolConfig(
                        min_sessions=1,
                        max_sessions=3,
                        session_ttl_seconds=300.0,  # 5 minutes
                        idle_timeout_seconds=60.0,  # 1 minute
                        health_check_interval_seconds=30.0,
                    ),
                )
                await self._pool.initialize()

                # Get tools from pooled session
                async with self._pool.acquire() as session:
                    tools = await session.list_tools()
                    self._tools = [t.name for t in tools.tools]
            else:
                # Fallback to per-request sessions
                async with self._session_fallback() as session:
                    tools = await session.list_tools()
                    self._tools = [t.name for t in tools.tools]

            self._available = True
            logger.info(f"Connected to Wikipedia MCP at {self._mcp_url}")
            logger.info(f"Available tools: {self._tools}")
            logger.info(f"Session pooling: {'enabled' if self._pool else 'disabled'}")

        except Exception as e:
            self._available = False
            logger.error(f"Wikipedia MCP connection failed: {e}")
            raise MCPConnectionError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect and shutdown the session pool.
        """
        if self._pool:
            await self._pool.shutdown()
            self._pool = None

        self._available = False
        self._tools = []
        logger.info("Disconnected from Wikipedia MCP")

    async def health_check(self) -> bool:
        """
        Check if Wikipedia MCP is responding.

        Uses pooled session if available for efficiency.
        """
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

    @asynccontextmanager
    async def _session(self):
        """
        Get an MCP session (from pool or create new).

        Uses the session pool for persistent connections when available.
        Falls back to per-request sessions if pool is not initialized.
        """
        if self._pool and self._pool.is_healthy:
            async with self._pool.acquire() as session:
                yield session
        else:
            # Fallback to per-request session
            async with self._session_fallback() as session:
                yield session

    @asynccontextmanager
    async def _session_fallback(self):
        """Create a new MCP session (non-pooled, for fallback)."""
        async with (
            sse_client(self._mcp_url) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session

    def get_pool_stats(self) -> dict[str, Any] | None:
        """Get session pool statistics."""
        if self._pool:
            return self._pool.get_stats()
        return None

    async def _call_tool(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Call an MCP tool within an existing session.

        Args:
            session: Active MCP client session.
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            Tool result as text.
        """
        result = await session.call_tool(tool_name, arguments)

        # Extract text content from result
        texts = []
        for content in result.content:
            if isinstance(content, TextContent):
                texts.append(content.text)

        return "\n".join(texts)

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Query Wikipedia for evidence related to a claim.

        Args:
            claim: The claim to find evidence for.

        Returns:
            List of Evidence objects from Wikipedia.
        """
        if not self._available:
            logger.warning("Wikipedia MCP not available, skipping")
            return []

        evidences: list[Evidence] = []
        query = claim.text

        # If claim has structured form, use subject for more targeted search
        if claim.subject:
            query = claim.subject

        try:
            async with self._session() as session:
                # Search Wikipedia for relevant articles
                search_text = await self._call_tool(
                    session,
                    "search_wikipedia",
                    {
                        "query": query,
                        "limit": 3,
                    },
                )

                if not search_text:
                    logger.debug(f"No Wikipedia results for: {query}")
                    return []

                # Parse search results to find article titles
                titles = self._parse_search_results(search_text, query)

                # Get summaries for top results IN PARALLEL
                async def get_summary_safe(title: str) -> Evidence | None:
                    try:
                        summary_text = await self._call_tool(
                            session,
                            "get_summary",
                            {
                                "title": title,
                            },
                        )

                        if summary_text and len(summary_text) > 50:
                            return Evidence(
                                id=uuid4(),
                                source=EvidenceSource.MCP_WIKIPEDIA,
                                source_id=f"wikipedia:{title}",
                                content=summary_text[:2000],  # Limit length
                                structured_data={
                                    "title": title,
                                    "query": query,
                                },
                                similarity_score=0.85,  # Base confidence
                                match_type="mcp_search",
                                retrieved_at=datetime.now(UTC),
                                source_uri=(
                                    f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                                ),
                            )
                    except Exception as e:
                        logger.debug(f"Failed to get summary for {title}: {e}")
                    return None

                # Fetch all summaries in parallel
                import asyncio

                results = await asyncio.gather(*[get_summary_safe(t) for t in titles[:3]])
                evidences = [e for e in results if e is not None]

            logger.info(f"Found {len(evidences)} Wikipedia evidences for claim")
            return evidences

        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform a general Wikipedia search.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                result = await self._call_tool(
                    session,
                    "search_wikipedia",
                    {
                        "query": query,
                        "limit": limit,
                    },
                )
                # Return as list of dicts with text
                return [{"text": result}] if result else []
        except Exception:
            return []

    def _parse_search_results(self, text: str, fallback_query: str) -> list[str]:
        """Parse search result text to extract article titles.

        The Wikipedia MCP returns JSON with structure:
        {
            "query": "...",
            "results": [{"title": "...", "snippet": "...", ...}, ...],
            "status": "success",
            "count": N
        }
        """
        import json

        titles = []

        if not text:
            return [fallback_query]

        # Try parsing as JSON first (the expected format)
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "results" in data:
                for result in data["results"]:
                    if isinstance(result, dict) and "title" in result:
                        titles.append(result["title"])
            return titles if titles else [fallback_query]
        except json.JSONDecodeError:
            pass

        # Fallback: parse as plain text lines
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Common formats: "1. Title", "- Title", "Title: description"
            # Remove numbering
            if line[0].isdigit() and "." in line[:4]:
                line = line.split(".", 1)[1].strip()

            # Remove bullet points
            if line.startswith("- ") or line.startswith("* "):
                line = line[2:].strip()

            # Take first part before colon if present
            if ":" in line:
                line = line.split(":")[0].strip()

            # Clean up and add if looks like a title
            if line and len(line) > 2 and len(line) < 200:
                titles.append(line)

        return titles if titles else [fallback_query]
