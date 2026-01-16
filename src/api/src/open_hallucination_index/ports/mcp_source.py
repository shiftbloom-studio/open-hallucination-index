"""
MCP Knowledge Source Port
=========================

Abstract interface for MCP-based external knowledge sources.
Defines the contract for Wikipedia, Context7, and other MCP servers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim, Evidence


class MCPKnowledgeSource(ABC):
    """
    Port for MCP-based knowledge retrieval.

    Implementations connect to MCP servers (Wikipedia, Context7, etc.)
    via HTTP/SSE transport and query for evidence related to claims.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of this MCP source."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the MCP server is currently reachable."""
        ...

    @abstractmethod
    async def connect(self) -> None:
        """
        Initialize connection to the MCP server.

        Raises:
            MCPConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the MCP server is healthy and responding.

        Returns:
            True if server is operational, False otherwise.
        """
        ...

    @abstractmethod
    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Query the MCP server for evidence related to a claim.

        Args:
            claim: The claim to find evidence for.

        Returns:
            List of Evidence objects from this MCP source.
            Empty list if no evidence found.
        """
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform a general search on this MCP source.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of search result dictionaries.
        """
        ...


class MCPConnectionError(Exception):
    """Raised when connection to MCP server fails."""

    pass


class MCPQueryError(Exception):
    """Raised when an MCP query fails."""

    pass


# -----------------------------------------------------------------------------
# Request-scoped MCP tool cache (per request/text/claim)
# -----------------------------------------------------------------------------

_mcp_call_cache: ContextVar[dict[str, list[dict[str, Any]]] | None] = ContextVar(
    "mcp_call_cache",
    default=None,
)


def set_mcp_call_cache() -> Token[dict[str, list[dict[str, Any]]] | None]:
    """Enable request-scoped MCP call cache for the current context."""
    return _mcp_call_cache.set({})


def reset_mcp_call_cache(token: Token[dict[str, list[dict[str, Any]]] | None]) -> None:
    """Reset the request-scoped MCP call cache to the previous context."""
    _mcp_call_cache.reset(token)


def get_mcp_call_cache() -> dict[str, list[dict[str, Any]]] | None:
    """Get the current request-scoped MCP call cache (if enabled)."""
    return _mcp_call_cache.get()
