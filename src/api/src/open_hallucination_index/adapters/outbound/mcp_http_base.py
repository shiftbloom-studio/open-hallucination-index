"""
HTTP-based MCP Adapter Base
===========================

Base class for MCP adapters that use direct HTTP API instead of SSE.
Provides common functionality for connection management, tool calls, and error handling.

This is the recommended approach for request-response patterns.
SSE should only be used for streaming/real-time updates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx

from open_hallucination_index.domain.entities import Evidence, EvidenceSource
from open_hallucination_index.ports.mcp_source import (
    MCPConnectionError,
    MCPKnowledgeSource,
    MCPQueryError,
    get_mcp_call_cache,
)

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim

logger = logging.getLogger(__name__)


class HTTPMCPAdapter(MCPKnowledgeSource):
    """
    Base class for HTTP-based MCP adapters.

    Provides common functionality:
    - HTTP client management with connection pooling
    - Tool call execution via /api/call endpoint
    - Health checks via /health endpoint
    - Consistent error handling

    Subclasses must implement:
    - source_name property
    - _get_base_url() to return the server URL
    - find_evidence() for claim-specific evidence retrieval
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_connections: int = 10,
        max_keepalive: int = 5,
    ) -> None:
        """
        Initialize the HTTP MCP adapter.

        Args:
            base_url: Base URL of the MCP server (without /api suffix).
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
            max_connections: Maximum number of connections in pool.
            max_keepalive: Maximum keepalive connections.
        """
        self._base_url = base_url.rstrip("/")
        # Remove /sse suffix if present - we use HTTP API
        if self._base_url.endswith("/sse"):
            self._base_url = self._base_url[:-4]
        self._api_url = f"{self._base_url}/api"
        self._available = False
        self._tools: list[str] = []
        self._client: httpx.AsyncClient | None = None
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        )
        # Serialize tool calls to avoid upstream rate limits (429s)
        self._call_semaphore = asyncio.Semaphore(1)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of this MCP source."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether the MCP server is currently reachable."""
        return self._available

    @property
    def tools(self) -> list[str]:
        """List of available tool names."""
        return self._tools.copy()

    async def connect(self) -> None:
        """
        Connect to the MCP server and fetch available tools.

        Creates a persistent HTTP client with connection pooling.
        """
        try:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=self._limits,
            )

            # Fetch available tools
            response = await self._client.get(f"{self._api_url}/tools")
            response.raise_for_status()
            data = response.json()
            self._tools = [t["name"] for t in data.get("tools", [])]

            self._available = True
            logger.info(f"Connected to {self.source_name} at {self._base_url} (HTTP API)")
            logger.info(f"Available tools ({len(self._tools)}): {self._tools[:10]}...")

        except httpx.ConnectError as e:
            self._available = False
            logger.error(f"{self.source_name} connection failed: {e}")
            raise MCPConnectionError(f"Failed to connect to {self.source_name}: {e}") from e
        except Exception as e:
            self._available = False
            logger.error(f"{self.source_name} connection failed: {e}")
            raise MCPConnectionError(f"Failed to connect to {self.source_name}: {e}") from e

    async def disconnect(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

        self._available = False
        self._tools = []
        logger.info(f"Disconnected from {self.source_name}")

    async def health_check(self) -> bool:
        """Check if the MCP server is healthy and responding."""
        if not self._client:
            return False
        try:
            response = await self._client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Call a tool via the HTTP API.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments as a dictionary.

        Returns:
            List of result dictionaries.

        Raises:
            MCPQueryError: If the tool call fails.
        """
        if not self._client:
            raise MCPQueryError(f"{self.source_name} not connected")

        cache = get_mcp_call_cache()
        cache_key: str | None = None
        if cache is not None:
            try:
                args_payload = json.dumps(arguments, sort_keys=True, default=str)
                cache_key = hashlib.sha256(f"{tool_name}:{args_payload}".encode("utf-8")).hexdigest()
                cached = cache.get(cache_key)
                if cached is not None:
                    logger.debug(
                        "MCP cache hit for %s (%s)",
                        tool_name,
                        self.source_name,
                    )
                    return cached
            except Exception:
                cache_key = None

        try:
            async with self._call_semaphore:
                response = await self._client.post(
                    f"{self._api_url}/call",
                    json={"tool": tool_name, "arguments": arguments},
                )
                response.raise_for_status()
                data = response.json()

            # Normalize response to list format
            if isinstance(data, dict):
                if "results" in data:
                    results = data["results"]
                elif "error" in data:
                    raise MCPQueryError(data["error"])
                else:
                    results = [data]
            elif isinstance(data, list):
                results = data
            else:
                results = [{"content": str(data)}]

            if cache is not None and cache_key is not None:
                cache[cache_key] = results
            return results

        except httpx.TimeoutException:
            raise MCPQueryError(f"Timeout calling {tool_name} on {self.source_name}") from None
        except httpx.HTTPStatusError as e:
            raise MCPQueryError(f"HTTP error calling {tool_name}: {e}") from e

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform a general search using the search_all tool.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of search result dictionaries.
        """
        if not self._available:
            return []

        try:
            return await self.call_tool("search_all", {"query": query, "limit": limit})
        except MCPQueryError as e:
            logger.warning(f"{self.source_name} search failed: {e}")
            return []

    def _create_evidence(
        self,
        result: dict[str, Any],
        source: EvidenceSource,
        tool_name: str,
    ) -> Evidence:
        """
        Create an Evidence object from a tool result.

        Args:
            result: Raw result dictionary from tool call.
            source: Evidence source type.
            tool_name: Name of the tool that produced this result.

        Returns:
            Evidence object.
        """
        raw_score = result.get("score", result.get("similarity_score", 0.8))
        try:
            if isinstance(raw_score, str) and raw_score.endswith("%"):
                score = float(raw_score.rstrip("%")) / 100.0
            else:
                score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.8

        # Normalize scores that may be on 0-100 scale or otherwise out of range
        if score > 1.0:
            if score <= 100.0:
                score = score / 100.0
            else:
                score = 1.0
        elif score < 0.0:
            score = 0.0

        return Evidence(
            id=uuid4(),
            source=source,
            content=result.get("content", result.get("title", "")),
            similarity_score=score,
            source_uri=result.get("url"),
            retrieved_at=datetime.now(),
            structured_data={
                "tool": tool_name,
                "original_source": result.get("source", "unknown"),
                **result.get("metadata", {}),
            },
        )

    @abstractmethod
    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Query the MCP server for evidence related to a claim.

        Subclasses must implement this method to define how claims
        are converted to tool calls and results to Evidence objects.

        Args:
            claim: The claim to find evidence for.

        Returns:
            List of Evidence objects from this MCP source.
        """
        ...
