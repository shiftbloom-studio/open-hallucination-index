"""
Wikidata SPARQL Adapter
=======================

Queries Wikidata knowledge graph via SPARQL endpoint.
https://query.wikidata.org/sparql
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
    SPARQLKnowledgeSource,
)
from open_hallucination_index.domain.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim

logger = logging.getLogger(__name__)


class WikidataAdapter(SPARQLKnowledgeSource):
    """
    Adapter for Wikidata SPARQL queries.
    
    Provides structured knowledge from Wikidata's vast
    linked data knowledge graph.
    """

    def __init__(
        self,
        base_url: str = "https://query.wikidata.org",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            user_agent="OpenHallucinationIndex/1.0 (https://github.com/open-hallucination-index; mailto:info@example.com)",
        )

    @property
    def source_name(self) -> str:
        return "Wikidata"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.WIKIDATA

    async def health_check(self) -> bool:
        """Check Wikidata endpoint with simple query."""
        if not self._client:
            return False
        try:
            query = "SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 1"
            await self._execute_sparql(query, "/sparql")
            return True
        except Exception:
            return False

    async def _execute_sparql(
        self,
        query: str,
        endpoint: str = "/sparql",
    ) -> dict[str, Any]:
        """Execute SPARQL query against Wikidata."""
        if not self._client:
            from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
                HTTPKnowledgeSourceError,
            )
            raise HTTPKnowledgeSourceError("Client not connected")
        
        response = await self._client.get(
            endpoint,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()
        return response.json()

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """
        Find Wikidata evidence for a claim.
        
        Uses text search and entity lookup to find relevant facts.
        """
        if not self._available:
            return []
        
        evidences: list[Evidence] = []
        
        # Extract search terms
        search_term = claim.subject or claim.text[:100]
        search_term = self._sanitize_query(search_term)
        
        try:
            # Search for entities matching the claim subject
            results = await self._search_entities(search_term, limit=3)
            
            for result in results:
                entity_id = result.get("id", "")
                label = result.get("label", "")
                description = result.get("description", "")
                
                if not entity_id:
                    continue
                
                # Get detailed properties for the entity
                props = await self._get_entity_properties(entity_id)
                
                content = f"{label}: {description}"
                if props:
                    content += f"\n\nProperties:\n{self._format_properties(props)}"
                
                evidences.append(self._create_evidence(
                    content=content,
                    source_id=f"wikidata:{entity_id}",
                    source_uri=f"https://www.wikidata.org/wiki/{entity_id}",
                    similarity_score=0.85,
                    structured_data={
                        "entity_id": entity_id,
                        "label": label,
                        "description": description,
                        "properties": props[:10] if props else [],
                    },
                ))
            
            logger.debug(f"Found {len(evidences)} Wikidata evidences for claim")
            return evidences
            
        except Exception as e:
            logger.warning(f"Wikidata search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Wikidata entities."""
        if not self._available:
            return []
        
        try:
            return await self._search_entities(query, limit)
        except Exception as e:
            logger.warning(f"Wikidata search failed: {e}")
            return []

    async def _search_entities(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search for Wikidata entities by text."""
        # Use Wikidata's search API via MediaWiki Action API
        # This is more efficient than SPARQL for text search
        search_url = "https://www.wikidata.org/w/api.php"
        
        # Use a separate request to wikidata.org API
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                search_url,
                params={
                    "action": "wbsearchentities",
                    "search": query,
                    "language": "en",
                    "limit": limit,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("search", []):
            results.append({
                "id": item.get("id", ""),
                "label": item.get("label", ""),
                "description": item.get("description", ""),
                "url": item.get("concepturi", ""),
            })
        
        return results

    async def _get_entity_properties(
        self, entity_id: str, limit: int = 10
    ) -> list[dict[str, str]]:
        """Get properties for a Wikidata entity via SPARQL."""
        query = f"""
        SELECT ?propLabel ?valueLabel WHERE {{
            wd:{entity_id} ?prop ?value .
            ?property wikibase:directClaim ?prop .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """
        
        try:
            result = await self._execute_sparql(query, "/sparql")
            props = []
            for binding in result.get("results", {}).get("bindings", []):
                prop_label = binding.get("propLabel", {}).get("value", "")
                value_label = binding.get("valueLabel", {}).get("value", "")
                if prop_label and value_label:
                    props.append({"property": prop_label, "value": value_label})
            return props
        except Exception:
            return []

    def _format_properties(self, props: list[dict[str, str]]) -> str:
        """Format properties as readable text."""
        lines = []
        for p in props:
            lines.append(f"- {p['property']}: {p['value']}")
        return "\n".join(lines)

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query string for SPARQL/API."""
        # Remove special characters that could break queries
        return re.sub(r'[^\w\s\-\.]', ' ', query).strip()
