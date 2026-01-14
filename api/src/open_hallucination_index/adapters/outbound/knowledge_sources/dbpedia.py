"""
DBpedia SPARQL Adapter
======================

Queries DBpedia knowledge graph via SPARQL endpoint.
http://dbpedia.org/sparql
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


class DBpediaAdapter(SPARQLKnowledgeSource):
    """
    Adapter for DBpedia SPARQL queries.
    
    DBpedia extracts structured content from Wikipedia,
    providing a semantic layer over Wikipedia articles.
    """

    def __init__(
        self,
        base_url: str = "http://dbpedia.org",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def source_name(self) -> str:
        return "DBpedia"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.DBPEDIA

    async def _execute_sparql(
        self,
        query: str,
        endpoint: str = "/sparql",
    ) -> dict[str, Any]:
        """Execute SPARQL query against DBpedia."""
        if not self._client:
            from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
                HTTPKnowledgeSourceError,
            )
            raise HTTPKnowledgeSourceError("Client not connected")
        
        response = await self._client.get(
            endpoint,
            params={
                "query": query,
                "format": "application/sparql-results+json",
            },
        )
        response.raise_for_status()
        return response.json()

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find DBpedia evidence for a claim."""
        if not self._available:
            return []
        
        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]
        search_term = self._sanitize_for_sparql(search_term)
        
        try:
            # Search for resources with matching labels
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbr: <http://dbpedia.org/resource/>
            
            SELECT DISTINCT ?resource ?label ?abstract WHERE {{
                ?resource rdfs:label ?label .
                ?resource dbo:abstract ?abstract .
                FILTER(LANG(?label) = 'en')
                FILTER(LANG(?abstract) = 'en')
                FILTER(CONTAINS(LCASE(?label), LCASE("{search_term}")))
            }}
            LIMIT 3
            """
            
            result = await self._execute_sparql(query, "/sparql")
            
            for binding in result.get("results", {}).get("bindings", []):
                resource = binding.get("resource", {}).get("value", "")
                label = binding.get("label", {}).get("value", "")
                abstract = binding.get("abstract", {}).get("value", "")
                
                if not resource or not abstract:
                    continue
                
                # Get additional properties
                props = await self._get_resource_properties(resource)
                
                content = f"{label}\n\n{abstract[:1500]}"
                if props:
                    content += f"\n\nRelated facts:\n{self._format_properties(props)}"
                
                resource_id = resource.split("/")[-1]
                evidences.append(self._create_evidence(
                    content=content,
                    source_id=f"dbpedia:{resource_id}",
                    source_uri=resource,
                    similarity_score=0.82,
                    structured_data={
                        "resource": resource,
                        "label": label,
                        "properties": props[:10] if props else [],
                    },
                ))
            
            logger.debug(f"Found {len(evidences)} DBpedia evidences for claim")
            return evidences
            
        except Exception as e:
            logger.warning(f"DBpedia search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search DBpedia resources."""
        if not self._available:
            return []
        
        query_safe = self._sanitize_for_sparql(query)
        
        try:
            sparql = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?resource ?label WHERE {{
                ?resource rdfs:label ?label .
                FILTER(LANG(?label) = 'en')
                FILTER(CONTAINS(LCASE(?label), LCASE("{query_safe}")))
            }}
            LIMIT {limit}
            """
            
            result = await self._execute_sparql(sparql, "/sparql")
            
            results = []
            for binding in result.get("results", {}).get("bindings", []):
                results.append({
                    "resource": binding.get("resource", {}).get("value", ""),
                    "label": binding.get("label", {}).get("value", ""),
                })
            return results
            
        except Exception as e:
            logger.warning(f"DBpedia search failed: {e}")
            return []

    async def _get_resource_properties(
        self, resource_uri: str, limit: int = 10
    ) -> list[dict[str, str]]:
        """Get key properties for a DBpedia resource."""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT ?propLabel ?value WHERE {{
            <{resource_uri}> ?prop ?value .
            ?prop rdfs:label ?propLabel .
            FILTER(LANG(?propLabel) = 'en')
            FILTER(isLiteral(?value) || isIRI(?value))
        }}
        LIMIT {limit}
        """
        
        try:
            result = await self._execute_sparql(query, "/sparql")
            props = []
            for binding in result.get("results", {}).get("bindings", []):
                prop_label = binding.get("propLabel", {}).get("value", "")
                value = binding.get("value", {}).get("value", "")
                if prop_label and value:
                    props.append({"property": prop_label, "value": str(value)[:200]})
            return props
        except Exception:
            return []

    def _format_properties(self, props: list[dict[str, str]]) -> str:
        """Format properties as readable text."""
        lines = []
        for p in props:
            lines.append(f"- {p['property']}: {p['value']}")
        return "\n".join(lines)

    def _sanitize_for_sparql(self, text: str) -> str:
        """Escape text for safe use in SPARQL queries."""
        # Escape quotes and backslashes
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        # Remove other problematic characters
        text = re.sub(r'[^\w\s\-\.\,]', '', text)
        return text.strip()[:200]
