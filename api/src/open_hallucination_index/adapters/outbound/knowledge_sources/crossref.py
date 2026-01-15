"""
Crossref API Adapter
====================

Queries Crossref for DOI metadata and scholarly works.
https://www.crossref.org/documentation/retrieve-metadata/rest-api/
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


class CrossrefAdapter(HTTPKnowledgeSource):
    """
    Adapter for Crossref REST API.

    Crossref is a DOI registration agency providing metadata
    for scholarly publications.
    """

    def __init__(
        self,
        base_url: str = "https://api.crossref.org",
        email: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self._email = email  # For polite pool

    @property
    def source_name(self) -> str:
        return "Crossref"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.CROSSREF

    async def connect(self) -> None:
        """Initialize with custom headers."""
        await super().connect()
        if self._client and self._email:
            self._client.headers["mailto"] = self._email

    async def health_check(self) -> bool:
        """Check Crossref API health."""
        if not self._client:
            return False
        try:
            response = await self._client.get("/works", params={"rows": 1})
            return response.status_code == 200
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find scholarly evidence for a claim."""
        if not self._available:
            return []

        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]

        try:
            works = await self._search_works(search_term, limit=5)

            for work in works:
                title = self._get_title(work)
                if not title:
                    continue

                # Build content
                authors = self._format_authors(work.get("author", []))
                content = f"{title}\n\nAuthors: {authors}"

                abstract = work.get("abstract", "")
                if abstract:
                    # Strip JATS XML tags if present
                    import re

                    abstract = re.sub(r"<[^>]+>", "", abstract)
                    content += f"\n\nAbstract: {abstract[:800]}"

                published = self._get_publication_date(work)
                if published:
                    content += f"\n\nPublished: {published}"

                container = work.get("container-title", [])
                if container:
                    content += f"\nJournal: {container[0]}"

                doi = work.get("DOI", "")

                evidences.append(
                    self._create_evidence(
                        content=content,
                        source_id=f"crossref:{doi}",
                        source_uri=f"https://doi.org/{doi}" if doi else "",
                        similarity_score=0.85,
                        structured_data={
                            "title": title,
                            "doi": doi,
                            "type": work.get("type", ""),
                            "publisher": work.get("publisher", ""),
                            "is_referenced_by_count": work.get("is-referenced-by-count", 0),
                        },
                    )
                )

            logger.debug(f"Found {len(evidences)} Crossref evidences")
            return evidences

        except Exception as e:
            logger.warning(f"Crossref search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Crossref works."""
        if not self._available:
            return []
        return await self._search_works(query, limit)

    async def _search_works(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for works via Crossref."""
        try:
            response = await self._client.get(
                "/works",
                params={
                    "query": query,
                    "rows": limit,
                    "sort": "relevance",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("items", [])
        except Exception as e:
            logger.warning(f"Crossref search error: {e}")
            return []

    async def get_work_by_doi(self, doi: str) -> dict[str, Any] | None:
        """Get work metadata by DOI."""
        try:
            encoded_doi = quote(doi, safe="/")
            response = await self._client.get(f"/works/{encoded_doi}")
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {})
            return None
        except Exception:
            return None

    def _get_title(self, work: dict[str, Any]) -> str:
        """Extract title from work."""
        titles = work.get("title", [])
        return titles[0] if titles else ""

    def _format_authors(self, authors: list[dict]) -> str:
        """Format author list."""
        names = []
        for author in authors[:5]:
            given = author.get("given", "")
            family = author.get("family", "")
            if family:
                name = f"{given} {family}".strip() if given else family
                names.append(name)

        if len(authors) > 5:
            names.append(f"and {len(authors) - 5} more")

        return ", ".join(names) if names else "Unknown"

    def _get_publication_date(self, work: dict[str, Any]) -> str:
        """Extract publication date."""
        published = work.get("published", {}) or work.get("created", {})
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            parts = date_parts[0]
            if len(parts) >= 3:
                return f"{parts[0]}-{parts[1]:02d}-{parts[2]:02d}"
            elif len(parts) >= 2:
                return f"{parts[0]}-{parts[1]:02d}"
            elif len(parts) >= 1:
                return str(parts[0])
        return ""
