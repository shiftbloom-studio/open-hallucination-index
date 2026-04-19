"""
MediaWiki Action API Adapter
============================

Queries Wikipedia and other MediaWiki sites via Action API.
https://www.mediawiki.org/wiki/API:Action_API
"""

from __future__ import annotations

import html
import logging
import re
from typing import TYPE_CHECKING, Any

from adapters.mcp_sources.base import (
    HTTPKnowledgeSource,
)
from models.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from models.entities import Claim

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

        try:
            search_results: list[tuple[float, dict[str, Any]]] = []
            seen_page_ids: set[str] = set()

            for query_index, search_term in enumerate(self._build_query_candidates(claim)):
                for result_index, result in enumerate(await self._search(search_term, limit=5)):
                    page_id = str(result.get("pageid", ""))
                    dedup_key = page_id or result.get("title", "")
                    if not dedup_key or dedup_key in seen_page_ids:
                        continue
                    seen_page_ids.add(dedup_key)
                    result["matched_query"] = search_term
                    score = self._score_result(
                        claim=claim,
                        result=result,
                        query_index=query_index,
                        result_index=result_index,
                    )
                    search_results.append((score, result))

            ranked_results = [
                result
                for _, result in sorted(
                    search_results,
                    key=lambda item: item[0],
                    reverse=True,
                )[:3]
            ]

            for rank, result in enumerate(ranked_results):
                title = result.get("title", "")
                if not title:
                    continue

                snippet = self._clean_snippet(result.get("snippet", ""))
                extract = await self._get_extract(title) if not self._is_useful_text(snippet) else ""
                content = self._build_content(title, snippet, extract)
                if not content:
                    continue

                page_id = result.get("pageid", "")
                evidences.append(
                    self._create_evidence(
                        content=content,
                        source_id=f"wikipedia:{page_id}",
                        source_uri=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        similarity_score=max(0.65, 0.9 - (rank * 0.08)),
                        structured_data={
                            "title": title,
                            "pageid": page_id,
                            "snippet": snippet or None,
                            "lead_extract": extract or None,
                            "matched_query": result.get("matched_query"),
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

    @staticmethod
    def _sanitize_query(query: str | None) -> str:
        if not query:
            return ""
        query = re.sub(r"\s+", " ", query).strip()
        return query[:200]

    def _build_query_candidates(self, claim: Claim) -> list[str]:
        candidates = [
            self._sanitize_query(claim.normalized_form),
            self._sanitize_query(claim.text),
            self._sanitize_query(" ".join(part for part in [claim.subject, claim.predicate, claim.object] if part)),
            self._sanitize_query(claim.subject),
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered or [self._sanitize_query(claim.text[:100])]

    @staticmethod
    def _clean_snippet(snippet: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", snippet)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _is_useful_text(text: str) -> bool:
        return len(text.strip()) >= 40

    def _build_content(self, title: str, snippet: str, extract: str) -> str:
        if self._is_useful_text(snippet):
            return f"{title}: {snippet}"
        if extract:
            return f"{title}: {extract}"
        return title

    def _score_result(
        self,
        *,
        claim: Claim,
        result: dict[str, Any],
        query_index: int,
        result_index: int,
    ) -> float:
        """Rank search results so entity-exact pages beat early noisy hits."""
        title = self._normalize_text(str(result.get("title", "")))
        snippet = self._normalize_text(self._clean_snippet(str(result.get("snippet", ""))))
        query = self._normalize_text(str(result.get("matched_query", "")))
        subject = self._normalize_text(claim.subject)
        obj = self._normalize_text(claim.object)
        predicate_terms = self._extract_keywords(claim.predicate)

        score = 0.0

        if subject:
            subject_terms = subject.split()
            matched_terms = sum(1 for term in subject_terms if term in title)
            score += matched_terms * 5.0
            if title == subject:
                score += 40.0
            elif subject in title:
                score += 15.0

        if query:
            if query == subject:
                score += 10.0
            elif query == title:
                score += 6.0

        if obj:
            if obj in snippet:
                score += 12.0
            elif obj in title:
                score += 6.0

        for term in predicate_terms:
            if term and term in snippet:
                score += 2.5

        if snippet:
            score += min(8.0, len(snippet) / 80.0)

        score -= query_index * 0.75
        score -= result_index * 1.5
        return score

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _extract_keywords(text: str | None) -> list[str]:
        if not text:
            return []
        words = re.findall(r"[a-z0-9]+", text.lower())
        stopwords = {"was", "were", "is", "are", "be", "been", "being", "in", "of", "the"}
        return [word for word in words if word not in stopwords]
