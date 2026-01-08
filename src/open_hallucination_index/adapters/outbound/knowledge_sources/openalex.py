"""
OpenAlex API Adapter
====================

Queries OpenAlex for academic works, authors, and institutions.
https://docs.openalex.org/
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


class OpenAlexAdapter(HTTPKnowledgeSource):
    """
    Adapter for OpenAlex API.
    
    OpenAlex is an open catalog of the world's scholarly papers,
    researchers, journals, and institutions.
    """

    def __init__(
        self,
        base_url: str = "https://api.openalex.org",
        email: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        # OpenAlex requests email for polite pool
        self._email = email

    @property
    def source_name(self) -> str:
        return "OpenAlex"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.OPENALEX

    async def health_check(self) -> bool:
        """Check OpenAlex API health."""
        if not self._client:
            return False
        try:
            response = await self._client.get("/works", params={"per_page": 1})
            return response.status_code == 200
        except Exception:
            return False

    def _get_params(self, **kwargs: Any) -> dict[str, Any]:
        """Add email to params if configured."""
        params = dict(kwargs)
        if self._email:
            params["mailto"] = self._email
        return params

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find academic evidence for a claim."""
        if not self._available:
            return []
        
        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]
        
        try:
            # Search for works matching the claim
            works = await self._search_works(search_term, limit=5)
            
            for work in works:
                title = work.get("title", "")
                if not title:
                    continue
                
                # Build content from work metadata
                abstract = work.get("abstract_inverted_index")
                abstract_text = self._reconstruct_abstract(abstract) if abstract else ""
                
                authors = self._format_authors(work.get("authorships", []))
                
                content = f"{title}\n\nAuthors: {authors}"
                if abstract_text:
                    content += f"\n\nAbstract: {abstract_text[:800]}"
                
                publication_date = work.get("publication_date", "")
                if publication_date:
                    content += f"\n\nPublished: {publication_date}"
                
                venue = work.get("primary_location", {}).get("source", {}).get("display_name", "")
                if venue:
                    content += f"\nVenue: {venue}"
                
                doi = work.get("doi", "")
                openalex_id = work.get("id", "").replace("https://openalex.org/", "")
                
                evidences.append(self._create_evidence(
                    content=content,
                    source_id=f"openalex:{openalex_id}",
                    source_uri=doi or work.get("id", ""),
                    similarity_score=0.85,
                    structured_data={
                        "title": title,
                        "doi": doi,
                        "openalex_id": openalex_id,
                        "publication_date": publication_date,
                        "cited_by_count": work.get("cited_by_count", 0),
                        "type": work.get("type", ""),
                    },
                ))
            
            logger.debug(f"Found {len(evidences)} OpenAlex evidences")
            return evidences
            
        except Exception as e:
            logger.warning(f"OpenAlex search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search OpenAlex works."""
        if not self._available:
            return []
        return await self._search_works(query, limit)

    async def _search_works(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search for academic works."""
        try:
            response = await self._client.get(
                "/works",
                params=self._get_params(
                    search=query,
                    per_page=limit,
                    sort="relevance_score:desc",
                ),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.warning(f"OpenAlex works search error: {e}")
            return []

    async def search_authors(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search for authors."""
        try:
            response = await self._client.get(
                "/authors",
                params=self._get_params(search=query, per_page=limit),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception:
            return []

    async def get_work_by_doi(self, doi: str) -> dict[str, Any] | None:
        """Get work by DOI."""
        try:
            encoded_doi = quote(doi, safe="")
            response = await self._client.get(
                f"/works/https://doi.org/{encoded_doi}",
                params=self._get_params(),
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def _reconstruct_abstract(
        self, inverted_index: dict[str, list[int]]
    ) -> str:
        """Reconstruct abstract from inverted index format."""
        if not inverted_index:
            return ""
        
        # Build word position map
        words = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        
        # Sort by position and join
        words.sort(key=lambda x: x[0])
        return " ".join(w[1] for w in words)

    def _format_authors(self, authorships: list[dict]) -> str:
        """Format author list."""
        authors = []
        for authorship in authorships[:5]:  # Limit to first 5
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)
        
        if len(authorships) > 5:
            authors.append(f"and {len(authorships) - 5} more")
        
        return ", ".join(authors) if authors else "Unknown"
