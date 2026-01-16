"""
GraphRAG Evaluator
==================

Graph-only baseline using Neo4j keyword matching.
No vector search, no MCP, no LLM decomposition.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from neo4j import GraphDatabase

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.evaluators.base import (
    AtomicFact,
    BaseEvaluator,
    EvaluatorResult,
    EvidenceItem,
    FActScoreResult,
    VerificationVerdict,
)


class GraphRAGEvaluator(BaseEvaluator):
    """Graph-only evaluator using Neo4j keyword/fulltext matching."""

    name = "GraphRAG"

    def __init__(self, config: ComparisonBenchmarkConfig) -> None:
        self.config = config
        self.graph_config = config.graph_rag
        self._driver = None

    async def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.graph_config.neo4j_uri,
                auth=(self.graph_config.neo4j_username, self.graph_config.neo4j_password),
                connection_timeout=5.0,
                max_connection_pool_size=5,
            )
        return self._driver

    def _extract_terms(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        stop = {
            "the",
            "is",
            "in",
            "of",
            "and",
            "to",
            "a",
            "an",
            "on",
            "for",
            "with",
            "by",
            "from",
            "as",
            "was",
            "were",
            "are",
            "be",
        }
        terms = [t for t in tokens if len(t) > 3 and t not in stop]
        # Keep most informative unique terms
        unique_terms = []
        for term in terms:
            if term not in unique_terms:
                unique_terms.append(term)
            if len(unique_terms) >= 6:
                break
        return unique_terms

    def _build_fulltext_query(self, terms: list[str]) -> str:
        return " ".join(terms)

    def _node_text(self, node: Any) -> str:
        parts: list[str] = []
        for key in ("first_paragraph", "content", "text", "title", "name"):
            value = node.get(key)
            if value:
                parts.append(str(value))
        return " ".join(parts)

    def _node_label(self, node: Any) -> str:
        labels = list(getattr(node, "labels", []) or [])
        if labels:
            return labels[0]
        return "node"

    def _query_graph(self, terms: list[str], limit: int) -> list[dict[str, Any]]:
        if not terms:
            return []

        driver = self._driver
        if driver is None:
            return []

        if self.graph_config.use_fulltext:
            query = self._build_fulltext_query(terms)
            cypher = """
            CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score
            WITH node, score
            ORDER BY score DESC
            LIMIT $limit
            OPTIONAL MATCH (node)-[r]-(neighbor)
            RETURN node, score, collect(DISTINCT neighbor)[0..$neighbor_limit] AS neighbors
            """

            with driver.session() as session:
                try:
                    result = session.run(
                        cypher,
                        index=self.graph_config.fulltext_index,
                        query=query,
                        limit=limit,
                        neighbor_limit=self.graph_config.neighbor_limit,
                    )
                    return [
                        {
                            "node": record["node"],
                            "score": record["score"],
                            "neighbors": record["neighbors"],
                            "source": "fulltext",
                        }
                        for record in result
                    ]
                except Exception:
                    pass

        cypher = """
        MATCH (n)
        WHERE ANY(term IN $terms WHERE (
            (n.name IS NOT NULL AND toLower(n.name) CONTAINS term) OR
            (n.title IS NOT NULL AND toLower(n.title) CONTAINS term) OR
            (n.text IS NOT NULL AND toLower(n.text) CONTAINS term) OR
            (n.content IS NOT NULL AND toLower(n.content) CONTAINS term) OR
            (n.first_paragraph IS NOT NULL AND toLower(n.first_paragraph) CONTAINS term)
        ))
        WITH n, [term IN $terms WHERE (
            (n.name IS NOT NULL AND toLower(n.name) CONTAINS term) OR
            (n.title IS NOT NULL AND toLower(n.title) CONTAINS term) OR
            (n.text IS NOT NULL AND toLower(n.text) CONTAINS term) OR
            (n.content IS NOT NULL AND toLower(n.content) CONTAINS term) OR
            (n.first_paragraph IS NOT NULL AND toLower(n.first_paragraph) CONTAINS term)
        )] AS matched_terms
        WITH n, matched_terms
        ORDER BY size(matched_terms) DESC
        LIMIT $limit
        OPTIONAL MATCH (n)-[r]-(neighbor)
        RETURN n, matched_terms, collect(DISTINCT neighbor)[0..$neighbor_limit] AS neighbors
        """

        with driver.session() as session:
            result = session.run(
                cypher,
                terms=terms,
                limit=limit,
                neighbor_limit=self.graph_config.neighbor_limit,
            )
            return [
                {
                    "node": record["n"],
                    "matched_terms": record["matched_terms"],
                    "neighbors": record["neighbors"],
                    "source": "keyword",
                }
                for record in result
            ]

    async def health_check(self) -> bool:
        try:
            driver = await self._get_driver()

            def _ping() -> bool:
                driver.verify_connectivity()
                return True

            return await asyncio.wait_for(asyncio.to_thread(_ping), timeout=5)
        except (asyncio.TimeoutError, Exception):
            return False

    async def verify(self, claim: str) -> EvaluatorResult:
        start_time = time.perf_counter()

        try:
            await self._get_driver()
            terms = self._extract_terms(claim)
            if not terms:
                return EvaluatorResult(
                    claim=claim,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    trust_score=0.0,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    evaluator=self.name,
                )

            results = await asyncio.to_thread(
                self._query_graph, terms, self.graph_config.top_k
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if not results:
                return EvaluatorResult(
                    claim=claim,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    trust_score=0.0,
                    latency_ms=latency_ms,
                    evaluator=self.name,
                )

            scored_results: list[dict[str, Any]] = []
            for r in results:
                node = r["node"]
                node_text = self._node_text(node).lower()
                matched_terms = {
                    t for t in terms if t and t in node_text
                }
                coverage = len(matched_terms) / max(1, len(terms))
                scored_results.append(
                    {
                        **r,
                        "coverage": coverage,
                        "matched_terms": list(matched_terms),
                        "node_text": node_text,
                    }
                )

            max_score = max(r["coverage"] for r in scored_results)

            if max_score >= self.graph_config.min_support_ratio:
                verdict = VerificationVerdict.SUPPORTED
            elif max_score >= self.graph_config.min_partial_ratio:
                verdict = VerificationVerdict.PARTIAL
            else:
                verdict = VerificationVerdict.UNVERIFIABLE

            evidence: list[EvidenceItem] = []
            for r in scored_results:
                node = r["node"]
                content = self._node_text(node)
                neighbors = r.get("neighbors") or []
                neighbor_names = []
                for neighbor in neighbors:
                    neighbor_names.append(
                        neighbor.get("name")
                        or neighbor.get("title")
                        or neighbor.get("id")
                        or self._node_label(neighbor)
                    )

                evidence.append(
                    EvidenceItem(
                        text=str(content)[:300],
                        source=f"neo4j:{self._node_label(node)}",
                        similarity_score=min(1.0, r["coverage"]),
                        metadata={
                            "matched_terms": r["matched_terms"],
                            "neighbors": [n for n in neighbor_names if n],
                            "retrieval": r.get("source"),
                            "fulltext_score": r.get("score"),
                        },
                    )
                )

            return EvaluatorResult(
                claim=claim,
                verdict=verdict,
                trust_score=max_score,
                latency_ms=latency_ms,
                evidence=evidence,
                evaluator=self.name,
                metadata={
                    "terms": terms,
                    "top_k": self.graph_config.top_k,
                    "retrieval": (
                        "fulltext" if self.graph_config.use_fulltext else "keyword"
                    ),
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return EvaluatorResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                trust_score=0.0,
                latency_ms=latency_ms,
                evaluator=self.name,
                error=str(e),
            )

    async def decompose_and_verify(self, text: str) -> FActScoreResult:
        start_time = time.perf_counter()

        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]

        atomic_facts: list[AtomicFact] = []

        for index, sentence in enumerate(sentences):
            result = await self.verify(sentence)
            atomic_facts.append(
                AtomicFact(
                    text=sentence,
                    source_text=text,
                    index=index,
                    verified=result.verdict in {
                        VerificationVerdict.SUPPORTED,
                        VerificationVerdict.PARTIAL,
                    },
                    evidence=result.evidence,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return FActScoreResult(
            original_text=text,
            atomic_facts=atomic_facts,
            latency_ms=latency_ms,
            evaluator=self.name,
        )

    async def close(self) -> None:
        if self._driver is not None:
            await asyncio.to_thread(self._driver.close)
            self._driver = None