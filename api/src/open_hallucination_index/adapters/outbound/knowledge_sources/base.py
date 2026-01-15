"""
Base HTTP Knowledge Source
==========================

Abstract base class for HTTP-based external knowledge sources.
Unlike MCP adapters, these connect directly to REST/SPARQL APIs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx

from open_hallucination_index.domain.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim

logger = logging.getLogger(__name__)


class HTTPKnowledgeSourceError(Exception):
    """Exception raised when HTTP knowledge source operations fail."""

    pass


class HTTPKnowledgeSource(ABC):
    """
    Abstract base class for HTTP-based knowledge sources.

    Provides common HTTP client functionality and connection pooling.
    Subclasses implement specific API protocols (REST, SPARQL, etc.).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_connections: int = 10,
        user_agent: str = "OpenHallucinationIndex/1.0 (https://github.com/open-hallucination-index)",
    ) -> None:
        """
        Initialize the HTTP knowledge source.

        Args:
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_connections: Maximum concurrent connections.
            user_agent: User-Agent header for requests.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._user_agent = user_agent
        self._client: httpx.AsyncClient | None = None
        self._available = False

        # Connection limits
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections // 2,
        )

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of this knowledge source."""
        ...

    @property
    @abstractmethod
    def evidence_source(self) -> EvidenceSource:
        """The EvidenceSource enum value for this adapter."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether the source is currently available."""
        return self._available

    @property
    def base_url(self) -> str:
        """Return the base URL."""
        return self._base_url

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        if self._client is not None:
            return

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            limits=self._limits,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

        # Test connection
        try:
            await self.health_check()
            self._available = True
            logger.info(f"{self.source_name} connected: {self._base_url}")
        except Exception as e:
            logger.warning(f"{self.source_name} connection test failed: {e}")
            self._available = False

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._available = False
        logger.info(f"{self.source_name} disconnected")

    async def health_check(self) -> bool:
        """Check if the API is responding."""
        if not self._client:
            return False
        try:
            # Subclasses can override with specific health check endpoints
            response = await self._client.get("/", timeout=5.0)
            return response.status_code < 500
        except Exception:
            return False

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make a GET request."""
        if not self._client:
            raise HTTPKnowledgeSourceError("Client not connected")

        response = await self._client.get(path, params=params, headers=headers)
        response.raise_for_status()
        return response

    async def _post(
        self,
        path: str,
        data: dict[str, Any] | str | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make a POST request."""
        if not self._client:
            raise HTTPKnowledgeSourceError("Client not connected")

        response = await self._client.post(path, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response

    @abstractmethod
    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Find evidence for a claim from this knowledge source.

        Args:
            claim: The claim to find evidence for.

        Returns:
            List of Evidence objects.
        """
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Search this knowledge source.

        Args:
            query: Search query string.
            limit: Maximum results.

        Returns:
            List of result dictionaries.
        """
        ...

    def _create_evidence(
        self,
        content: str,
        source_id: str,
        source_uri: str | None = None,
        similarity_score: float = 0.8,
        structured_data: dict[str, Any] | None = None,
    ) -> Evidence:
        """Helper to create Evidence objects with standard fields."""
        return Evidence(
            id=uuid4(),
            source=self.evidence_source,
            source_id=source_id,
            content=content[:4000],  # Limit content length
            structured_data=structured_data,
            similarity_score=similarity_score,
            match_type="api_search",
            retrieved_at=datetime.now(UTC),
            source_uri=source_uri,
        )


class SPARQLKnowledgeSource(HTTPKnowledgeSource):
    """
    Base class for SPARQL endpoint knowledge sources.

    Provides SPARQL query execution functionality.
    """

    async def _execute_sparql(
        self,
        query: str,
        endpoint: str = "",
    ) -> dict[str, Any]:
        """
        Execute a SPARQL query.

        Args:
            query: SPARQL query string.
            endpoint: Endpoint path (default uses base URL).

        Returns:
            SPARQL results as JSON.
        """
        if not self._client:
            raise HTTPKnowledgeSourceError("Client not connected")

        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = await self._client.post(
            endpoint or "/sparql",
            data={"query": query},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        """Check SPARQL endpoint health with simple query."""
        if not self._client:
            return False
        try:
            # Simple ASK query to test endpoint
            await self._execute_sparql("ASK { ?s ?p ?o } LIMIT 1")
            return True
        except Exception:
            return False
