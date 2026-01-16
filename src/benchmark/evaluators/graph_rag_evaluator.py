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
    """Graph-only evaluator using Neo4j keyword matching."""

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

    def _query_graph(self, terms: list[str], limit: int) -> list[dict[str, Any]]:
        if not terms:
            return []

        cypher = """
        UNWIND $terms AS term
        MATCH (n)
        WHERE (
            (exists(n.name) AND toLower(n.name) CONTAINS term) OR
            (exists(n.title) AND toLower(n.title) CONTAINS term) OR
            (exists(n.text) AND toLower(n.text) CONTAINS term) OR
            (exists(n.content) AND toLower(n.content) CONTAINS term)
        )
        WITH n, collect(DISTINCT term) AS matched_terms
        RETURN n, size(matched_terms) AS match_count, matched_terms
        ORDER BY match_count DESC
        LIMIT $limit
        """

        driver = self._driver
        if driver is None:
            return []

        with driver.session() as session:
            result = session.run(cypher, terms=terms, limit=limit)
            return [
                {
                    "node": record["n"],
                    "match_count": record["match_count"],
                    "matched_terms": record["matched_terms"],
                }
                for record in result
            ]

    async def health_check(self) -> bool:
        try:
            driver = await self._get_driver()

            def _ping() -> bool:
                with driver.session() as session:
                    session.run("RETURN 1").single()
                return True

            return await asyncio.to_thread(_ping)
        except Exception:
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

            max_match = max(r["match_count"] for r in results)
            max_score = max_match / max(1, len(terms))

            if max_score >= self.graph_config.min_support_ratio:
                verdict = VerificationVerdict.SUPPORTED
            elif max_score >= self.graph_config.min_partial_ratio:
                verdict = VerificationVerdict.PARTIAL
            else:
                verdict = VerificationVerdict.UNVERIFIABLE

            evidence: list[EvidenceItem] = []
            for r in results:
                node = r["node"]
                content = (
                    node.get("content")
                    or node.get("text")
                    or node.get("title")
                    or node.get("name")
                    or ""
                )
                evidence.append(
                    EvidenceItem(
                        text=str(content)[:300],
                        source="neo4j",
                        similarity_score=min(1.0, r["match_count"] / max(1, len(terms))),
                        metadata={"matched_terms": r["matched_terms"]},
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

        for sentence in sentences:
            result = await self.verify(sentence)
            atomic_facts.append(
                AtomicFact(
                    text=sentence,
                    is_supported=result.verdict in {
                        VerificationVerdict.SUPPORTED,
                        VerificationVerdict.PARTIAL,
                    },
                    confidence=result.trust_score,
                    evidence=result.evidence,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        supported = sum(1 for f in atomic_facts if f.is_supported)
        precision = supported / len(atomic_facts) if atomic_facts else 0.0

        return FActScoreResult(
            input_text=text,
            atomic_facts=atomic_facts,
            precision=precision,
            latency_ms=latency_ms,
            evaluator=self.name,
        )

    async def close(self) -> None:
        if self._driver is not None:
            await asyncio.to_thread(self._driver.close)
            self._driver = None