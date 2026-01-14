"""
Wikimedia REST API Adapter
==========================

Queries Wikimedia REST API (rest_v1) for structured content.
https://wikimedia.org/api/rest_v1/
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
    HTTPKnowledgeSource,
)
from open_hallucination_index.domain.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim

logger = logging.getLogger(__name__)


class WikimediaRESTAdapter(HTTPKnowledgeSource):
    """
    Adapter for Wikimedia REST API.
    
    Provides access to Wikipedia summaries, page content,
    and on-this-day events via the structured REST API.
    """

    def __init__(
        self,
        base_url: str = "https://en.wikipedia.org/api/rest_v1",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)

    @property
    def source_name(self) -> str:
        return "Wikimedia REST"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.WIKIMEDIA_REST

    async def health_check(self) -> bool:
        """Check Wikimedia REST API health."""
        if not self._client:
            return False
        try:
            # Check feed endpoint
            response = await self._client.get("/page/random/summary")
            return response.status_code == 200
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find Wikipedia evidence via REST API."""
        if not self._available:
            return []
        
        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]
        
        try:
            # Get page summary directly
            summary = await self._get_summary(search_term)
            
            if summary and summary.get("extract"):
                title = summary.get("title", search_term)
                evidences.append(self._create_evidence(
                    content=f"{title}\n\n{summary.get('extract', '')}",
                    source_id=f"wikipedia:{summary.get('pageid', '')}",
                    source_uri=summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    similarity_score=0.88,
                    structured_data={
                        "title": title,
                        "description": summary.get("description", ""),
                        "pageid": summary.get("pageid"),
                        "type": summary.get("type"),
                    },
                ))
            
            logger.debug(f"Found {len(evidences)} Wikimedia REST evidences")
            return evidences
            
        except Exception as e:
            logger.warning(f"Wikimedia REST search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search is not directly supported; use summary lookup."""
        if not self._available:
            return []
        
        summary = await self._get_summary(query)
        if summary:
            return [summary]
        return []

    async def _get_summary(self, title: str) -> dict[str, Any] | None:
        """Get page summary via REST API."""
        try:
            encoded_title = quote(title.replace(" ", "_"), safe="")
            response = await self._client.get(f"/page/summary/{encoded_title}")
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.debug(f"Summary fetch failed for {title}: {e}")
            return None

    async def get_on_this_day(
        self,
        month: int,
        day: int,
        event_type: str = "events",
    ) -> list[dict[str, Any]]:
        """
        Get historical events for a date.
        
        Args:
            month: Month (1-12)
            day: Day of month
            event_type: One of 'events', 'births', 'deaths', 'holidays'
        """
        try:
            response = await self._client.get(
                f"/feed/onthisday/{event_type}/{month:02d}/{day:02d}"
            )
            response.raise_for_status()
            data = response.json()
            return data.get(event_type, [])
        except Exception:
            return []

    async def get_page_html(self, title: str) -> str | None:
        """Get parsed HTML content for a page."""
        try:
            encoded_title = quote(title.replace(" ", "_"), safe="")
            response = await self._client.get(
                f"/page/html/{encoded_title}",
                headers={"Accept": "text/html"},
            )
            if response.status_code == 200:
                return response.text
            return None
        except Exception:
            return None
