"""
Context7 MCP Adapter
====================

Adapter for Context7 documentation retrieval via MCP server.
Connects to the configured MCP endpoint (streamable HTTP or SSE).
Provides up-to-date library/API documentation for technical claims.

Now supports session pooling for persistent HTTP connections to improve performance.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from open_hallucination_index.adapters.outbound.mcp_session_pool import (
    MCPSessionPool,
    MCPTransportType,
    PoolConfig,
)
from open_hallucination_index.domain.entities import Evidence, EvidenceSource
from open_hallucination_index.ports.mcp_source import (
    MCPKnowledgeSource,
)

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.infrastructure.config import MCPSettings

logger = logging.getLogger(__name__)

# Common programming libraries/frameworks to detect in claims
TECH_KEYWORDS = {
    "python",
    "javascript",
    "typescript",
    "react",
    "vue",
    "angular",
    "node",
    "fastapi",
    "django",
    "flask",
    "express",
    "next.js",
    "nextjs",
    "vite",
    "pytorch",
    "tensorflow",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "redis",
    "mongodb",
    "postgresql",
    "mysql",
    "graphql",
    "rest",
    "api",
    "sdk",
    "library",
    "framework",
    "package",
    "module",
    "npm",
    "pip",
    "cargo",
    "maven",
}


class Context7MCPAdapter(MCPKnowledgeSource):
    """
    Adapter for Context7 MCP server.

    Uses MCP SDK to communicate with the configured MCP endpoint.
    Provides evidence from library documentation.
    """

    def __init__(self, settings: MCPSettings) -> None:
        """
        Initialize the Context7 MCP adapter.

        Args:
            settings: MCP configuration with Context7 URL and API key.
        """
        self._settings = settings
        self._base_url = settings.context7_url.rstrip("/")
        # Choose transport based on URL suffix
        if self._base_url.endswith("/sse"):
            self._mcp_url = self._base_url
            self._transport_type = MCPTransportType.SSE
        elif self._base_url.endswith("/mcp"):
            self._mcp_url = self._base_url
            self._transport_type = MCPTransportType.STREAMABLE_HTTP
        else:
            # Default to SSE for the unified OHI MCP server
            self._mcp_url = f"{self._base_url}/sse"
            self._transport_type = MCPTransportType.SSE
        self._api_key = settings.context7_api_key
        self._available = False
        self._tools: list[str] = []

        # Session pool for persistent connections
        self._pool: MCPSessionPool | None = None
        self._use_pool = True  # Enable session pooling by default

        # Auth headers for Context7 (streamable HTTP only)
        self._headers: dict[str, str] = {}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"

    @property
    def source_name(self) -> str:
        """Human-readable name of this MCP source."""
        return "Context7"

    @property
    def is_available(self) -> bool:
        """Whether the MCP server is currently reachable."""
        return self._available

    async def connect(self) -> None:
        """
        Test connection and list available tools.

        Initializes the session pool for persistent HTTP connections.
        """
        try:
            # Initialize session pool for persistent connections
            if self._use_pool:
                pool_kwargs = {
                    "source_name": "Context7",
                    "mcp_url": self._mcp_url,
                    "transport_type": self._transport_type,
                    "config": PoolConfig(
                        min_sessions=1,
                        max_sessions=3,
                        session_ttl_seconds=300.0,  # 5 minutes
                        idle_timeout_seconds=60.0,  # 1 minute
                        health_check_interval_seconds=30.0,
                    ),
                }
                if self._transport_type == MCPTransportType.STREAMABLE_HTTP:
                    pool_kwargs["headers"] = self._headers

                self._pool = MCPSessionPool(**pool_kwargs)
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
            logger.info(f"Connected to Context7 MCP at {self._mcp_url}")
            logger.info(f"Available tools: {self._tools}")
            logger.info(f"Session pooling: {'enabled' if self._pool else 'disabled'}")

        except Exception as e:
            self._available = False
            logger.warning(f"Context7 MCP connection failed: {e}")
            # Don't raise - Context7 is optional

    async def disconnect(self) -> None:
        """
        Disconnect and shutdown the session pool.
        """
        if self._pool:
            await self._pool.shutdown()
            self._pool = None

        self._available = False
        self._tools = []
        logger.info("Disconnected from Context7 MCP")

    async def health_check(self) -> bool:
        """
        Check if Context7 MCP is responding.

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
        if self._transport_type == MCPTransportType.SSE:
            async with (
                sse_client(self._mcp_url) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
        else:
            async with (
                streamablehttp_client(self._mcp_url, headers=self._headers) as (read, write, _),
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
        self, session: ClientSession, tool_name: str, arguments: dict[str, Any]
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

    def _is_technical_claim(self, claim: Claim) -> bool:
        """Check if claim is about a technical topic (library, framework, etc.)."""
        text_lower = claim.text.lower()
        return any(keyword in text_lower for keyword in TECH_KEYWORDS)

    def _extract_library_name(self, claim: Claim) -> str | None:
        """Extract potential library/framework name from claim."""
        text = claim.text

        # Look for common patterns (explicit library names first, then generic patterns)
        patterns = [
            # First try to match explicit library names that are well-known
            r"\b(React|Vue|Angular|FastAPI|Django|Flask|Express|Next\.?js|PyTorch|TensorFlow|Pandas|NumPy|Scikit-learn|Keras|Redis|MongoDB|PostgreSQL|MySQL|GraphQL|REST|Docker|Kubernetes)\b",
            # Then try word-before-type patterns
            r"\b([A-Za-z][A-Za-z0-9_.-]+)\s+(?:library|framework|package|module)",
            r"(?:in|using|with)\s+([A-Za-z][A-Za-z0-9_.-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        # Check for subject if structured
        if claim.subject:
            subject_lower = claim.subject.lower()
            for keyword in TECH_KEYWORDS:
                if keyword in subject_lower:
                    return claim.subject

        return None

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Query Context7 for documentation evidence related to a claim.

        Args:
            claim: The claim to find evidence for.

        Returns:
            List of Evidence objects from Context7 documentation.
        """
        if not self._available:
            logger.debug("Context7 MCP not available, skipping")
            return []

        # Only query Context7 for technical claims
        if not self._is_technical_claim(claim):
            logger.debug("Claim is not technical, skipping Context7")
            return []

        evidences: list[Evidence] = []
        library_name = self._extract_library_name(claim)

        if not library_name:
            logger.debug("Could not extract library name from claim")
            return []

        try:
            async with self._session() as session:
                # First resolve the library to a Context7 ID
                # resolve-library-id requires both 'query' and 'libraryName' parameters
                resolve_text = await self._call_tool(
                    session,
                    "resolve-library-id",
                    {
                        "query": claim.text,  # Full claim text for context
                        "libraryName": library_name,
                    },
                )

                if not resolve_text:
                    logger.debug(f"No Context7 library found for: {library_name}")
                    return []

                library_id = self._extract_library_id(resolve_text)
                if not library_id:
                    return []

                # Get documentation for the library
                # query-docs requires 'libraryId' and 'query' parameters
                topic = self._extract_topic(claim)

                docs_text = await self._call_tool(
                    session,
                    "query-docs",
                    {
                        "libraryId": library_id,
                        "query": topic if topic else claim.text,
                    },
                )

                if docs_text and len(docs_text) > 50:
                    evidences.append(
                        Evidence(
                            id=uuid4(),
                            source=EvidenceSource.MCP_CONTEXT7,
                            source_id=f"context7:{library_id}",
                            content=docs_text[:3000],  # Limit length
                            structured_data={
                                "library_id": library_id,
                                "library_name": library_name,
                                "topic": topic,
                                "mcp_source": "context7",
                            },
                            similarity_score=0.9,  # High confidence for docs
                            match_type="mcp_docs",
                            retrieved_at=datetime.now(UTC),
                            source_uri=f"https://context7.com{library_id}",
                        )
                    )

            logger.info(f"Found {len(evidences)} Context7 evidences for claim")
            return evidences

        except Exception as e:
            logger.warning(f"Context7 query failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for libraries in Context7.

        Args:
            query: Search query (library name).
            limit: Maximum results (not used, Context7 returns best match).

        Returns:
            List of search results.
        """
        if not self._available:
            return []

        try:
            async with self._session() as session:
                result = await self._call_tool(
                    session,
                    "resolve-library-id",
                    {
                        "libraryName": query,
                    },
                )
                return [{"text": result}] if result else []
        except Exception:
            return []

    def _extract_library_id(self, text: str) -> str | None:
        """Extract library ID from resolve-library-id response."""
        if not text:
            return None

        # Look for "Context7-compatible library ID: /org/project" pattern
        # The response contains this field after each library title
        pattern = r"Context7-compatible library ID:\s*(/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Fallback: look for any library ID pattern that's NOT /org/project
        for m in re.finditer(r"(/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)", text):
            lib_id = m.group(1)
            if lib_id != "/org/project":
                return lib_id

        return None

    def _extract_topic(self, claim: Claim) -> str | None:
        """Extract specific topic from claim for focused documentation."""
        # Look for specific concepts mentioned in claim
        topic_patterns = [
            r"(hooks?|routing|authentication|api|components?|state|props)",
            r"(async|await|promises?|callbacks?)",
            r"(classes?|functions?|methods?|decorators?)",
        ]

        for pattern in topic_patterns:
            match = re.search(pattern, claim.text, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return None
