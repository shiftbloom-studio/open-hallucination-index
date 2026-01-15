"""
Fair VectorRAG Evaluator
========================

VectorRAG baseline using PUBLIC Wikipedia API - not our local database.

This provides a FAIR comparison because:
1. Uses same data source any VectorRAG user would have access to
2. No advantage from our specialized knowledge preprocessing
3. Standard RAG workflow: search Wikipedia → embed → compare

This simulates what a typical VectorRAG implementation would look like
without access to our curated knowledge graph.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.evaluators.base import (
    AtomicFact,
    BaseEvaluator,
    EvaluatorResult,
    EvidenceItem,
    FActScoreResult,
    VerificationVerdict,
)


class FairVectorRAGEvaluator(BaseEvaluator):
    """
    Fair VectorRAG baseline using public Wikipedia API.
    
    This is how a typical VectorRAG user would implement verification:
    1. Search Wikipedia for relevant articles
    2. Embed claim and article text
    3. Compare embeddings for similarity
    
    No access to our specialized database - pure public API.
    """
    
    name = "VectorRAG"
    
    WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, config: ComparisonBenchmarkConfig) -> None:
        self.config = config
        self.rag_config = config.vector_rag
        
        self._http_client: httpx.AsyncClient | None = None
        self._embedding_model = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                headers={
                    "User-Agent": "OHI-Benchmark/1.0 (https://openhallucination.xyz; research)"
                }
            )
        return self._http_client
    
    async def _get_embedding_model(self):
        """Get or load sentence transformer model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self.rag_config.embedding_model
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedding_model
    
    async def _embed(self, text: str) -> list[float]:
        """Embed text using sentence transformer."""
        model = await self._get_embedding_model()
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    async def _search_wikipedia(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search Wikipedia using the public API.
        
        Returns list of article summaries with titles and extracts.
        """
        client = await self._get_client()
        
        try:
            # Step 1: Search for relevant articles
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
            }
            
            response = await client.get(self.WIKIPEDIA_API, params=search_params)
            response.raise_for_status()
            data = response.json()
            
            if "query" not in data or "search" not in data["query"]:
                return []
            
            # Get page IDs
            page_ids = [str(item["pageid"]) for item in data["query"]["search"]]
            
            if not page_ids:
                return []
            
            # Step 2: Get article extracts
            extract_params = {
                "action": "query",
                "pageids": "|".join(page_ids),
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json",
            }
            
            response = await client.get(self.WIKIPEDIA_API, params=extract_params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "query" in data and "pages" in data["query"]:
                for page_id, page_data in data["query"]["pages"].items():
                    if "extract" in page_data:
                        results.append({
                            "title": page_data.get("title", ""),
                            "text": page_data.get("extract", "")[:1000],  # Limit text
                            "pageid": page_id,
                        })
            
            return results
            
        except Exception:
            return []
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    
    async def health_check(self) -> bool:
        """Check if Wikipedia API is accessible."""
        try:
            client = await self._get_client()
            response = await client.get(
                self.WIKIPEDIA_API,
                params={"action": "query", "meta": "siteinfo", "format": "json"}
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def verify(self, claim: str) -> EvaluatorResult:
        """
        Verify a claim using Wikipedia search + embedding similarity.
        
        Standard RAG approach with public API:
        1. Search Wikipedia for relevant articles
        2. Embed claim and article text
        3. Compare embeddings
        4. Return verdict based on max similarity
        """
        start_time = time.perf_counter()
        
        try:
            # Search Wikipedia
            articles = await self._search_wikipedia(claim, limit=5)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if not articles:
                return EvaluatorResult(
                    claim=claim,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    trust_score=0.0,
                    latency_ms=latency_ms,
                    evaluator=self.name,
                    metadata={"source": "wikipedia_api", "articles_found": 0},
                )
            
            # Embed claim
            claim_embedding = await self._embed(claim)
            
            # Embed and score each article
            scored_articles = []
            for article in articles:
                article_embedding = await self._embed(article["text"])
                similarity = self._cosine_similarity(claim_embedding, article_embedding)
                scored_articles.append({
                    **article,
                    "score": similarity,
                })
            
            # Sort by similarity
            scored_articles.sort(key=lambda x: x["score"], reverse=True)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Get max similarity score
            max_score = scored_articles[0]["score"] if scored_articles else 0.0
            
            # Simple threshold-based verdict
            if max_score >= self.rag_config.similarity_threshold:
                verdict = VerificationVerdict.SUPPORTED
                trust_score = max_score
            elif max_score >= self.rag_config.similarity_threshold * 0.7:
                verdict = VerificationVerdict.PARTIAL
                trust_score = max_score
            else:
                verdict = VerificationVerdict.UNVERIFIABLE
                trust_score = max_score
            
            # Convert to evidence items
            evidence = [
                EvidenceItem(
                    text=a["text"][:300] if a["text"] else "",
                    source=f"Wikipedia: {a['title']}",
                    similarity_score=a["score"],
                    metadata={"pageid": a.get("pageid", "")},
                )
                for a in scored_articles[:3]
            ]
            
            return EvaluatorResult(
                claim=claim,
                verdict=verdict,
                trust_score=trust_score,
                latency_ms=latency_ms,
                evidence=evidence,
                evaluator=self.name,
                metadata={
                    "source": "wikipedia_api",
                    "articles_found": len(articles),
                    "threshold": self.rag_config.similarity_threshold,
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
        """
        Simple sentence-based decomposition and verification.
        
        VectorRAG uses basic sentence splitting (no LLM decomposition)
        then verifies each sentence independently.
        """
        start_time = time.perf_counter()
        
        try:
            # Simple sentence splitting
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
            
            atomic_facts: list[AtomicFact] = []
            
            for i, sentence in enumerate(sentences):
                # Verify each sentence
                result = await self.verify(sentence)
                
                atomic_facts.append(AtomicFact(
                    text=sentence,
                    source_text=text,
                    index=i,
                    verified=result.is_supported,
                    evidence=result.evidence,
                ))
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return FActScoreResult(
                original_text=text,
                atomic_facts=atomic_facts,
                latency_ms=latency_ms,
                evaluator=self.name,
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return FActScoreResult(
                original_text=text,
                atomic_facts=[],
                latency_ms=latency_ms,
                evaluator=self.name,
                error=str(e),
            )
    
    async def close(self) -> None:
        """Close connections."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._embedding_model = None
