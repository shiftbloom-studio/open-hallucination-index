"""
OHI Evaluator
=============

Evaluator using the Open Hallucination Index API.
This is our primary system - expected to outperform baselines.
"""

from __future__ import annotations

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


class OHIEvaluator(BaseEvaluator):
    """
    Evaluator using the OHI API for claim verification.
    
    Uses the full OHI pipeline:
    - Hybrid graph + vector search
    - MCP knowledge sources
    - Claim decomposition
    - Evidence aggregation
    """
    
    name = "OHI"
    
    def __init__(self, config: ComparisonBenchmarkConfig) -> None:
        self.config = config
        self.base_url = config.ohi_api_base_url
        self.verify_url = config.ohi_verify_url
        self.strategy = config.ohi_strategy
        self.timeout = config.timeout_seconds
        
        # Persistent HTTP client
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=20),
            )
        return self._client
    
    async def health_check(self) -> bool:
        """Check if OHI API is healthy."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def verify(self, claim: str) -> EvaluatorResult:
        """
        Verify a claim using OHI API.
        
        Args:
            claim: The claim text to verify.
            
        Returns:
            EvaluatorResult with OHI verification results.
        """
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            
            payload = {
                "text": claim,
                "strategy": self.strategy,
                "return_evidence": True,
                "target_sources": 10,
                "use_cache": False,  # Disable cache for accurate benchmarking
            }
            
            if self.config.ohi_api_key:
                headers = {"X-API-Key": self.config.ohi_api_key}
            else:
                headers = {}
            
            response = await client.post(
                self.verify_url,
                json=payload,
                headers=headers,
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if response.status_code != 200:
                return EvaluatorResult(
                    claim=claim,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    trust_score=0.0,
                    latency_ms=latency_ms,
                    evaluator=self.name,
                    error=f"API error: {response.status_code}",
                )
            
            data = response.json()
            
            # Parse OHI response
            trust_score = data.get("trust_score", 0.0)
            claims_data = data.get("claims", [])
            
            # Determine verdict from trust score
            if trust_score >= 0.7:
                verdict = VerificationVerdict.SUPPORTED
            elif trust_score >= 0.4:
                verdict = VerificationVerdict.PARTIAL
            elif trust_score >= 0.2:
                verdict = VerificationVerdict.UNVERIFIABLE
            else:
                verdict = VerificationVerdict.REFUTED
            
            # Extract evidence
            evidence: list[EvidenceItem] = []
            for claim_data in claims_data:
                for ev in claim_data.get("evidence", []):
                    evidence.append(EvidenceItem(
                        text=ev.get("text", "")[:500],
                        source=ev.get("source", "unknown"),
                        similarity_score=ev.get("similarity_score", 0.0),
                        metadata=ev.get("metadata", {}),
                    ))
            
            return EvaluatorResult(
                claim=claim,
                verdict=verdict,
                trust_score=trust_score,
                latency_ms=latency_ms,
                evidence=evidence[:5],  # Limit evidence count
                evaluator=self.name,
                metadata={
                    "claims_count": len(claims_data),
                    "strategy": self.strategy,
                    "response_id": data.get("id"),
                },
            )
            
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return EvaluatorResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                trust_score=0.0,
                latency_ms=latency_ms,
                evaluator=self.name,
                error="Request timeout",
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
        Decompose text and verify each atomic fact using OHI.
        
        OHI's claim decomposer extracts atomic facts,
        then each is verified individually.
        """
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            
            # First, get claim decomposition
            payload = {
                "text": text,
                "strategy": self.strategy,
                "return_evidence": True,
                "target_sources": 10,
            }
            
            headers = {}
            if self.config.ohi_api_key:
                headers["Authorization"] = f"Bearer {self.config.ohi_api_key}"
            
            response = await client.post(
                self.verify_url,
                json=payload,
                headers=headers,
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if response.status_code != 200:
                return FActScoreResult(
                    original_text=text,
                    atomic_facts=[],
                    latency_ms=latency_ms,
                    evaluator=self.name,
                    error=f"API error: {response.status_code}",
                )
            
            data = response.json()
            claims_data = data.get("claims", [])
            
            # Convert OHI claims to atomic facts
            atomic_facts: list[AtomicFact] = []
            for i, claim_data in enumerate(claims_data):
                claim_text = claim_data.get("text", "")
                claim_verdict = claim_data.get("verdict", "unverifiable")
                
                # Map OHI verdict to verified boolean
                verified = claim_verdict.lower() == "supported"
                
                # Extract evidence
                evidence = [
                    EvidenceItem(
                        text=ev.get("text", "")[:300],
                        source=ev.get("source", ""),
                        similarity_score=ev.get("similarity_score", 0.0),
                    )
                    for ev in claim_data.get("evidence", [])[:3]
                ]
                
                atomic_facts.append(AtomicFact(
                    text=claim_text,
                    source_text=text,
                    index=i,
                    verified=verified,
                    evidence=evidence,
                ))
            
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
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
