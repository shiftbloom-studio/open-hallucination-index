"""
Europe PMC API Adapter
======================

Queries Europe PMC for life sciences literature.
https://europepmc.org/RestfulWebService
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


class EuropePMCAdapter(HTTPKnowledgeSource):
    """
    Adapter for Europe PMC REST API.
    
    Europe PMC provides access to life sciences literature,
    including full-text articles and abstracts.
    """

    def __init__(
        self,
        base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)

    @property
    def source_name(self) -> str:
        return "Europe PMC"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.EUROPE_PMC

    async def health_check(self) -> bool:
        """Check Europe PMC API health."""
        if not self._client:
            return False
        try:
            response = await self._client.get(
                "/search",
                params={"query": "test", "resultType": "lite", "pageSize": 1, "format": "json"},
            )
            return response.status_code == 200
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find biomedical evidence for a claim."""
        if not self._available:
            return []
        
        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]
        
        try:
            articles = await self._search_articles(search_term, limit=5)
            
            for article in articles:
                title = article.get("title", "")
                if not title:
                    continue
                
                # Build content
                authors = article.get("authorString", "Unknown")
                content = f"{title}\n\nAuthors: {authors}"
                
                abstract = article.get("abstractText", "")
                if abstract:
                    content += f"\n\nAbstract: {abstract[:800]}"
                
                pub_year = article.get("pubYear", "")
                if pub_year:
                    content += f"\n\nPublished: {pub_year}"
                
                journal = article.get("journalTitle", "")
                if journal:
                    content += f"\nJournal: {journal}"
                
                pmid = article.get("pmid", "")
                pmcid = article.get("pmcid", "")
                doi = article.get("doi", "")
                
                source_uri = ""
                if pmcid:
                    source_uri = f"https://europepmc.org/article/PMC/{pmcid}"
                elif pmid:
                    source_uri = f"https://europepmc.org/article/MED/{pmid}"
                elif doi:
                    source_uri = f"https://doi.org/{doi}"
                
                evidences.append(self._create_evidence(
                    content=content,
                    source_id=f"europepmc:{pmid or pmcid or doi}",
                    source_uri=source_uri,
                    similarity_score=0.85,
                    structured_data={
                        "title": title,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "doi": doi,
                        "isOpenAccess": article.get("isOpenAccess", "N"),
                        "citedByCount": article.get("citedByCount", 0),
                    },
                ))
            
            logger.debug(f"Found {len(evidences)} Europe PMC evidences")
            return evidences
            
        except Exception as e:
            logger.warning(f"Europe PMC search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Europe PMC articles."""
        if not self._available:
            return []
        return await self._search_articles(query, limit)

    async def _search_articles(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search for articles."""
        try:
            response = await self._client.get(
                "/search",
                params={
                    "query": query,
                    "resultType": "core",
                    "pageSize": limit,
                    "format": "json",
                    "sort": "RELEVANCE",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("resultList", {}).get("result", [])
        except Exception as e:
            logger.warning(f"Europe PMC search error: {e}")
            return []

    async def get_article_by_pmid(self, pmid: str) -> dict[str, Any] | None:
        """Get article by PubMed ID."""
        try:
            response = await self._client.get(
                f"/search",
                params={
                    "query": f"EXT_ID:{pmid}",
                    "resultType": "core",
                    "format": "json",
                },
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("resultList", {}).get("result", [])
                return results[0] if results else None
            return None
        except Exception:
            return None

    async def get_citations(
        self, source: str, identifier: str, limit: int = 25
    ) -> list[dict[str, Any]]:
        """
        Get articles citing a given article.
        
        Args:
            source: Source type ('MED' for PubMed, 'PMC' for PMC)
            identifier: Article identifier (PMID or PMCID)
            limit: Maximum citations to return
        """
        try:
            response = await self._client.get(
                f"/{source}/{identifier}/citations",
                params={"page": 1, "pageSize": limit, "format": "json"},
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("citationList", {}).get("citation", [])
            return []
        except Exception:
            return []
