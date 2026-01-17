"""
VectorRAG Evaluator
===================

Simple vector similarity-based claim verification.
Standard RAG baseline - fast but less accurate.

Uses only Qdrant vector search without:
- Graph relationships
- MCP knowledge sources
- Claim decomposition
- Evidence aggregation
"""

from __future__ import annotations

import time

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.evaluators.base import (
    AtomicFact,
    BaseEvaluator,
    EvaluatorResult,
    EvidenceItem,
    FActScoreResult,
    VerificationVerdict,
)


class VectorRAGEvaluator(BaseEvaluator):
    """
    Simple vector similarity-based claim verification.
    
    Standard RAG implementation:
    1. Embed the claim
    2. Search for similar passages in Qdrant
    3. If max similarity > threshold → supported
    
    This is the simplest baseline - fast but less accurate
    than hybrid approaches like OHI.
    """
    
    name = "VectorRAG"
    
    def __init__(self, config: ComparisonBenchmarkConfig) -> None:
        self.config = config
        self.rag_config = config.vector_rag
        
        self._qdrant_client = None
        self._embedding_model = None
    
    async def _get_qdrant_client(self):
        """Get or create Qdrant client."""
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
                self._qdrant_client = QdrantClient(
                    host=self.rag_config.qdrant_host,
                    port=self.rag_config.qdrant_port,
                    timeout=30,
                )
            except ImportError:
                raise ImportError(
                    "qdrant-client not installed. "
                    "Install with: pip install qdrant-client"
                )
        return self._qdrant_client
    
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
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def _search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search Qdrant for similar passages.
        
        Returns list of hits with text, score, and metadata.
        """
        client = await self._get_qdrant_client()
        
        try:
            using_vector = "dense"

            results = client.query_points(
                collection_name=self.rag_config.collection_name,
                query=query_embedding,
                using=using_vector,
                limit=top_k,
                with_payload=True,
            )
            
            return [
                {
                    "text": hit.payload.get("text", hit.payload.get("content", "")),
                    "score": hit.score,
                    "title": hit.payload.get("title", ""),
                    "url": hit.payload.get("url", ""),
                }
                for hit in results.points
            ]
        except Exception:
            return []
    
    async def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            client = await self._get_qdrant_client()
            collections = client.get_collections()
            return any(
                c.name == self.rag_config.collection_name
                for c in collections.collections
            )
        except Exception:
            return False
    
    async def verify(self, claim: str) -> EvaluatorResult:
        """
        Verify a claim using simple vector similarity.
        
        Standard RAG approach:
        1. Embed claim
        2. Find similar passages
        3. Check if max similarity > threshold
        """
        start_time = time.perf_counter()
        
        try:
            # Embed the claim
            embedding = await self._embed(claim)
            
            # Search for similar content
            results = await self._search_similar(
                embedding,
                top_k=self.rag_config.top_k,
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
            
            # Get max similarity score
            max_score = max(r["score"] for r in results)
            
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
                    text=r["text"][:300] if r["text"] else "",
                    source=r.get("title", "Wikipedia"),
                    similarity_score=r["score"],
                    metadata={"url": r.get("url", "")},
                )
                for r in results[:3]
            ]
            
            return EvaluatorResult(
                claim=claim,
                verdict=verdict,
                trust_score=trust_score,
                latency_ms=latency_ms,
                evidence=evidence,
                evaluator=self.name,
                metadata={
                    "top_k": self.rag_config.top_k,
                    "threshold": self.rag_config.similarity_threshold,
                    "results_count": len(results),
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
        if self._qdrant_client:
            self._qdrant_client.close()
            self._qdrant_client = None
        self._embedding_model = None
