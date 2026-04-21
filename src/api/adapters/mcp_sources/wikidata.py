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

import httpx

from adapters.mcp_sources.base import (
    SPARQLKnowledgeSource,
)
from models.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from models.entities import Claim

logger = logging.getLogger(__name__)


class WikidataAdapter(SPARQLKnowledgeSource):
    """
    Adapter for Wikidata SPARQL queries.

    Provides structured knowledge from Wikidata's vast
    linked data knowledge graph.
    """

    _ENTITY_STOPWORDS = {
        "A",
        "An",
        "Der",
        "Die",
        "Das",
        "Den",
        "Dem",
        "Des",
        "Ein",
        "Eine",
        "Einem",
        "Einen",
        "The",
    }

    _SUBJECT_VERB_HINTS = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "hat",
        "haben",
        "ist",
        "war",
        "wurde",
        "invented",
        "erfunden",
        "founded",
        "created",
        "developed",
        "discovered",
        "won",
        "became",
        "built",
    }

    _LEADING_DETERMINERS = {
        "the",
        "a",
        "an",
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "einem",
        "einen",
    }

    # Base similarity score used for the top result; lower-ranked results
    # will receive slightly smaller scores derived from this value.
    DEFAULT_SIMILARITY_SCORE: float = 0.85

    def __init__(
        self,
        base_url: str = "https://query.wikidata.org",
        timeout: float = 30.0,
    ) -> None:
        user_agent = (
            "OpenHallucinationIndex/1.0 "
            "(https://github.com/open-hallucination-index; "
            "mailto:contact@open-hallucination-index.org)"
        )
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            user_agent=user_agent,
        )
        self._wikidata_api_url = "https://www.wikidata.org/w/api.php"

    @property
    def source_name(self) -> str:
        return "Wikidata"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.WIKIDATA

    def _compute_similarity_score(self, rank: int, total_results: int) -> float:
        """
        Compute a simple similarity score based on the result rank.

        The highest-ranked result (rank == 0) gets DEFAULT_SIMILARITY_SCORE,
        and subsequent results get linearly decreasing scores, but never
        below 0.0.
        """
        if total_results <= 1:
            return self.DEFAULT_SIMILARITY_SCORE

        # Linearly decay the score with rank, clamped to [0.0, 1.0].
        decay_step = self.DEFAULT_SIMILARITY_SCORE / max(total_results - 1, 1)
        score = self.DEFAULT_SIMILARITY_SCORE - (decay_step * rank)
        return max(0.0, min(1.0, score))

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
            from adapters.mcp_sources.base import (
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

        try:
            # Build robust search terms from subject + text entities. This keeps
            # Wikidata retrieval working even when decomposition leaves subject
            # empty for non-English claims.
            candidate_results: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            for search_term in self._build_search_terms(claim):
                per_term_added = 0
                for result in await self._search_entities(search_term, limit=4):
                    entity_id = result.get("id", "")
                    if not entity_id or entity_id in seen_ids:
                        continue
                    seen_ids.add(entity_id)
                    result["matched_query"] = search_term
                    candidate_results.append(result)
                    per_term_added += 1
                    if per_term_added >= 2:
                        break
                    if len(candidate_results) >= 6:
                        break
                if len(candidate_results) >= 6:
                    break

            total_results = len(candidate_results)
            for rank, result in enumerate(candidate_results[:3]):
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

                evidences.append(
                    self._create_evidence(
                        content=content,
                        source_id=f"wikidata:{entity_id}",
                        source_uri=f"https://www.wikidata.org/wiki/{entity_id}",
                        similarity_score=self._compute_similarity_score(rank, total_results),
                        structured_data={
                            "entity_id": entity_id,
                            "label": label,
                            "description": description,
                            "matched_query": result.get("matched_query"),
                            "properties": props[:10] if props else [],
                        },
                    )
                )

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

    async def _search_entities(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for Wikidata entities by text."""
        # Use Wikidata's search API via MediaWiki Action API
        # This is more efficient than SPARQL for text search
        search_url = self._wikidata_api_url

        # Use a separate request to wikidata.org API
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
        ) as client:
            results: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for language in self._candidate_languages(query):
                response = await client.get(
                    search_url,
                    params={
                        "action": "wbsearchentities",
                        "search": query,
                        "language": language,
                        "limit": limit,
                        "format": "json",
                    },
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("search", []):
                    entity_id = item.get("id", "")
                    if not entity_id or entity_id in seen_ids:
                        continue
                    seen_ids.add(entity_id)
                    results.append(
                        {
                            "id": entity_id,
                            "label": item.get("label", ""),
                            "description": item.get("description", ""),
                            "url": item.get("concepturi", ""),
                        }
                    )
                    if len(results) >= limit:
                        return results

        return results[:limit]

    async def _get_entity_properties(self, entity_id: str, limit: int = 10) -> list[dict[str, str]]:
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

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize query string for SPARQL/API."""
        # Remove special characters that could break queries
        return re.sub(r"[^\w\s\-\.]", " ", query).strip()

    def _build_search_terms(self, claim: Claim) -> list[str]:
        candidates = [
            self._sanitize_query(claim.subject or ""),
            *self._extract_subject_fallback(claim.text),
            self._sanitize_query(claim.object or ""),
            self._sanitize_query(claim.normalized_form or ""),
            *self._extract_entity_candidates(claim.text),
            self._sanitize_query(claim.text[:120]),
        ]
        seen: set[str] = set()
        terms: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                terms.append(candidate)
            if len(terms) >= 4:
                break
        return terms or [self._sanitize_query(claim.text[:100])]

    @classmethod
    def _extract_subject_fallback(cls, text: str | None) -> list[str]:
        if not text:
            return []

        words = re.findall(r"[a-zA-Z\u00c0-\u024f][a-zA-Z0-9\u00c0-\u024f\-]*", text)
        if len(words) < 2:
            return []

        lower_words = [w.lower() for w in words]
        split_idx = next(
            (idx for idx, word in enumerate(lower_words) if word in cls._SUBJECT_VERB_HINTS),
            -1,
        )
        if split_idx <= 0:
            return []

        subject_tokens = words[:split_idx]
        while subject_tokens and subject_tokens[0].lower() in cls._LEADING_DETERMINERS:
            subject_tokens = subject_tokens[1:]

        subject_tokens = subject_tokens[:4]
        if not subject_tokens:
            return []

        candidate = cls._sanitize_query(" ".join(subject_tokens))
        return [candidate] if candidate else []

    @classmethod
    def _extract_entity_candidates(cls, text: str | None) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        current: list[str] = []
        for raw_word in re.split(r"\s+", text):
            word = re.sub(r"^[^\w\u00c0-\u024f]+|[^\w\u00c0-\u024f\-]+$", "", raw_word)
            if not word:
                continue
            if word[0].isupper() and len(word) > 1 and word not in cls._ENTITY_STOPWORDS:
                current.append(word)
                continue
            if current:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))

        ranked = sorted(chunks, key=lambda phrase: (len(phrase.split()), len(phrase)), reverse=True)
        seen: set[str] = set()
        out: list[str] = []
        for phrase in ranked:
            sanitized = re.sub(r"\s+", " ", phrase).strip()
            if sanitized and sanitized not in seen:
                seen.add(sanitized)
                out.append(sanitized)
            if len(out) >= 4:
                break
        return out

    @staticmethod
    def _candidate_languages(query: str) -> list[str]:
        lowered = query.lower()
        has_umlaut = bool(re.search(r"[äöüß]", lowered))
        has_german_function_words = bool(
            re.search(r"\b(der|die|das|und|ist|hat|ein|eine|des|im|am|zum|zur|nicht)\b", lowered)
        )
        if has_umlaut or has_german_function_words:
            return ["de", "en"]
        return ["en"]
