"""
NCBI E-utilities Adapter
========================

Queries NCBI databases (PubMed, etc.) via E-utilities.
https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
    HTTPKnowledgeSource,
)
from open_hallucination_index.domain.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim

logger = logging.getLogger(__name__)


class NCBIAdapter(HTTPKnowledgeSource):
    """
    Adapter for NCBI E-utilities.
    
    Provides access to PubMed and other NCBI databases
    for biomedical literature and data.
    """

    def __init__(
        self,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        api_key: str | None = None,
        email: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self._api_key = api_key  # Optional, increases rate limit
        self._email = email  # Required for higher rate limits

    @property
    def source_name(self) -> str:
        return "NCBI"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.NCBI

    def _get_params(self, **kwargs: Any) -> dict[str, Any]:
        """Add common parameters to requests."""
        params = dict(kwargs)
        if self._api_key:
            params["api_key"] = self._api_key
        if self._email:
            params["email"] = self._email
        params["retmode"] = kwargs.get("retmode", "json")
        return params

    async def health_check(self) -> bool:
        """Check NCBI E-utilities health."""
        if not self._client:
            return False
        try:
            response = await self._client.get(
                "/einfo.fcgi",
                params=self._get_params(),
            )
            return response.status_code == 200
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find PubMed evidence for a claim."""
        if not self._available:
            return []
        
        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]
        
        try:
            # Search PubMed
            pmids = await self._search_pubmed(search_term, limit=5)
            
            if not pmids:
                return []
            
            # Fetch article details
            articles = await self._fetch_articles(pmids)
            
            for article in articles:
                title = article.get("title", "")
                if not title:
                    continue
                
                authors = article.get("authors", "Unknown")
                content = f"{title}\n\nAuthors: {authors}"
                
                abstract = article.get("abstract", "")
                if abstract:
                    content += f"\n\nAbstract: {abstract[:800]}"
                
                pub_date = article.get("pubdate", "")
                if pub_date:
                    content += f"\n\nPublished: {pub_date}"
                
                journal = article.get("source", "")
                if journal:
                    content += f"\nJournal: {journal}"
                
                pmid = article.get("uid", "")
                
                evidences.append(self._create_evidence(
                    content=content,
                    source_id=f"pubmed:{pmid}",
                    source_uri=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    similarity_score=0.85,
                    structured_data={
                        "pmid": pmid,
                        "title": title,
                        "journal": journal,
                        "pubdate": pub_date,
                    },
                ))
            
            logger.debug(f"Found {len(evidences)} NCBI evidences")
            return evidences
            
        except Exception as e:
            logger.warning(f"NCBI search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search PubMed."""
        if not self._available:
            return []
        
        pmids = await self._search_pubmed(query, limit)
        if pmids:
            return await self._fetch_articles(pmids)
        return []

    async def _search_pubmed(
        self, query: str, limit: int = 5
    ) -> list[str]:
        """Search PubMed and return PMIDs."""
        try:
            response = await self._client.get(
                "/esearch.fcgi",
                params=self._get_params(
                    db="pubmed",
                    term=query,
                    retmax=limit,
                    sort="relevance",
                ),
            )
            response.raise_for_status()
            data = response.json()
            
            result = data.get("esearchresult", {})
            return result.get("idlist", [])
        except Exception as e:
            logger.warning(f"PubMed search error: {e}")
            return []

    async def _fetch_articles(
        self, pmids: list[str]
    ) -> list[dict[str, Any]]:
        """Fetch article details for given PMIDs."""
        if not pmids:
            return []
        
        try:
            response = await self._client.get(
                "/esummary.fcgi",
                params=self._get_params(
                    db="pubmed",
                    id=",".join(pmids),
                ),
            )
            response.raise_for_status()
            data = response.json()
            
            result = data.get("result", {})
            articles = []
            
            for pmid in pmids:
                if pmid in result:
                    articles.append(result[pmid])
            
            return articles
        except Exception as e:
            logger.warning(f"Article fetch error: {e}")
            return []

    async def get_abstract(self, pmid: str) -> str:
        """Fetch full abstract for a PMID."""
        try:
            response = await self._client.get(
                "/efetch.fcgi",
                params=self._get_params(
                    db="pubmed",
                    id=pmid,
                    retmode="xml",
                    rettype="abstract",
                ),
            )
            response.raise_for_status()
            
            # Parse XML response
            root = ElementTree.fromstring(response.text)
            abstract_elem = root.find(".//AbstractText")
            if abstract_elem is not None and abstract_elem.text:
                return abstract_elem.text
            return ""
        except Exception:
            return ""

    async def search_gene(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search NCBI Gene database."""
        try:
            # Search
            search_resp = await self._client.get(
                "/esearch.fcgi",
                params=self._get_params(
                    db="gene",
                    term=query,
                    retmax=limit,
                ),
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            
            gene_ids = search_data.get("esearchresult", {}).get("idlist", [])
            if not gene_ids:
                return []
            
            # Fetch summaries
            summary_resp = await self._client.get(
                "/esummary.fcgi",
                params=self._get_params(
                    db="gene",
                    id=",".join(gene_ids),
                ),
            )
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
            
            result = summary_data.get("result", {})
            genes = []
            for gid in gene_ids:
                if gid in result:
                    genes.append(result[gid])
            
            return genes
        except Exception:
            return []
