"""
OpenCitations Index API Adapter
===============================

Queries OpenCitations for citation data.
https://api.opencitations.net/index/v1
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


class OpenCitationsAdapter(HTTPKnowledgeSource):
    """
    Adapter for OpenCitations Index API.

    OpenCitations provides open scholarly citation data.
    Useful for verifying claims about citation counts
    and relationships between academic works.
    """

    def __init__(
        self,
        base_url: str = "https://opencitations.net/index/api/v1",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)

    @property
    def source_name(self) -> str:
        return "OpenCitations"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.OPENCITATIONS

    async def health_check(self) -> bool:
        """Check OpenCitations API health."""
        if not self._client:
            return False
        try:
            # Try to get metadata endpoint
            response = await self._client.get("/metadata/doi:10.1038/nature12373")
            return response.status_code in (200, 404)  # 404 is OK, means API works
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Find citation evidence for a claim.

        Works best when claim contains DOI references.
        """
        if not self._available:
            return []

        evidences: list[Evidence] = []

        # Extract DOIs from claim text
        dois = self._extract_dois(claim.text)

        if not dois:
            # OpenCitations needs DOIs to query
            logger.debug("No DOIs found in claim, skipping OpenCitations")
            return []

        try:
            for doi in dois[:3]:  # Limit to first 3 DOIs
                # Get citation metadata
                metadata = await self._get_metadata(doi)
                if not metadata:
                    continue

                meta = metadata[0] if metadata else {}

                # Get citation count
                citations = await self._get_citations(doi)
                references = await self._get_references(doi)

                title = meta.get("title", f"Work: {doi}")
                author = meta.get("author", "Unknown")

                content = f"{title}\n\nAuthors: {author}"
                content += f"\n\nCited by: {len(citations)} works"
                content += f"\nReferences: {len(references)} works"

                pub_date = meta.get("pub_date", "")
                if pub_date:
                    content += f"\n\nPublished: {pub_date}"

                venue = meta.get("source_title", "")
                if venue:
                    content += f"\nVenue: {venue}"

                evidences.append(
                    self._create_evidence(
                        content=content,
                        source_id=f"opencitations:{doi}",
                        source_uri=f"https://doi.org/{doi}",
                        similarity_score=0.9,
                        structured_data={
                            "doi": doi,
                            "title": title,
                            "citation_count": len(citations),
                            "reference_count": len(references),
                            "oa_link": meta.get("oa_link", ""),
                        },
                    )
                )

            logger.debug(f"Found {len(evidences)} OpenCitations evidences")
            return evidences

        except Exception as e:
            logger.warning(f"OpenCitations search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        OpenCitations doesn't support text search.
        Returns metadata if query contains DOIs.
        """
        if not self._available:
            return []

        dois = self._extract_dois(query)
        results = []

        for doi in dois[:limit]:
            metadata = await self._get_metadata(doi)
            if metadata:
                results.extend(metadata)

        return results

    async def _get_metadata(self, doi: str) -> list[dict[str, Any]]:
        """Get metadata for a DOI."""
        try:
            encoded = quote(f"doi:{doi}", safe=":")
            response = await self._client.get(f"/metadata/{encoded}")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    async def _get_citations(self, doi: str) -> list[dict[str, Any]]:
        """Get works that cite this DOI."""
        try:
            encoded = quote(f"doi:{doi}", safe=":")
            response = await self._client.get(f"/citations/{encoded}")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    async def _get_references(self, doi: str) -> list[dict[str, Any]]:
        """Get works referenced by this DOI."""
        try:
            encoded = quote(f"doi:{doi}", safe=":")
            response = await self._client.get(f"/references/{encoded}")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def _extract_dois(self, text: str) -> list[str]:
        """Extract DOIs from text."""
        import re

        # DOI pattern: 10.xxxx/xxxxx
        pattern = r"10\.\d{4,}/[^\s\]\)\>]+"
        matches = re.findall(pattern, text)
        # Clean trailing punctuation
        return [m.rstrip(".,;:") for m in matches]
