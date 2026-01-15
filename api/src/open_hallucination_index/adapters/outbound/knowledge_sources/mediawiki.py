"""
MediaWiki Action API Adapter
============================

Queries Wikipedia and other MediaWiki sites via Action API.
https://www.mediawiki.org/wiki/API:Action_API
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
    HTTPKnowledgeSource,
)
from open_hallucination_index.domain.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim

logger = logging.getLogger(__name__)


class MediaWikiAdapter(HTTPKnowledgeSource):
    """
    Adapter for MediaWiki Action API.

    Provides access to Wikipedia articles, search, and metadata
    via the standard MediaWiki API.
    """

    def __init__(
        self,
        base_url: str = "https://en.wikipedia.org/w/api.php",
        timeout: float = 30.0,
    ) -> None:
        # MediaWiki uses api.php as the endpoint
        # We keep the full path as base_url for simplicity
        super().__init__(
            base_url=base_url.rsplit("/", 1)[0] if "/api.php" in base_url else base_url,
            timeout=timeout,
        )
        self._api_path = "/api.php" if "/api.php" not in base_url else "/" + base_url.split("/")[-1]

    @property
    def source_name(self) -> str:
        return "MediaWiki"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.MEDIAWIKI

    async def health_check(self) -> bool:
        """Check MediaWiki API health."""
        if not self._client:
            return False
        try:
            response = await self._client.get(
                self._api_path,
                params={"action": "query", "meta": "siteinfo", "format": "json"},
            )
            return response.status_code == 200
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find Wikipedia evidence for a claim via Action API."""
        if not self._available:
            return []

        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]

        try:
            # Search for articles
            search_results = await self._search(search_term, limit=3)

            for result in search_results:
                title = result.get("title", "")
                if not title:
                    continue

                # Get article extract (summary)
                extract = await self._get_extract(title)
                if not extract:
                    continue

                page_id = result.get("pageid", "")
                evidences.append(
                    self._create_evidence(
                        content=f"{title}\n\n{extract}",
                        source_id=f"wikipedia:{page_id}",
                        source_uri=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        similarity_score=0.85,
                        structured_data={
                            "title": title,
                            "pageid": page_id,
                            "snippet": result.get("snippet", ""),
                        },
                    )
                )

            logger.debug(f"Found {len(evidences)} MediaWiki evidences for claim")
            return evidences

        except Exception as e:
            logger.warning(f"MediaWiki search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Wikipedia articles."""
        if not self._available:
            return []
        return await self._search(query, limit)

    async def _search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Execute search query via Action API."""
        try:
            response = await self._client.get(
                self._api_path,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "srprop": "snippet|titlesnippet",
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("query", {}).get("search", [])
        except Exception as e:
            logger.warning(f"MediaWiki search error: {e}")
            return []

    async def _get_extract(self, title: str, sentences: int = 5) -> str:
        """Get article extract/summary."""
        try:
            response = await self._client.get(
                self._api_path,
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "exsentences": sentences,
                    "exlimit": 1,
                    "explaintext": True,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
            return ""
        except Exception:
            return ""

    async def get_page_info(self, title: str) -> dict[str, Any]:
        """Get detailed page information."""
        try:
            response = await self._client.get(
                self._api_path,
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "info|categories|links",
                    "pllimit": 20,
                    "cllimit": 10,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page
            return {}
        except Exception:
            return {}
