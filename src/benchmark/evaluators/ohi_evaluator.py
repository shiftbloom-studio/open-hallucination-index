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
    
    def __init__(
        self,
        config: ComparisonBenchmarkConfig,
        *,
        name_override: str | None = None,
        strategy_override: str | None = None,
        target_sources_override: int | None = None,
    ) -> None:
        self.config = config
        self.base_url = config.ohi_api_base_url
        self.verify_url = config.ohi_verify_url
        self.strategy = strategy_override or config.ohi_strategy
        if name_override:
            self.name = name_override
        self.timeout = config.timeout_seconds
        self._target_sources = target_sources_override or 10
        
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
                "target_sources": self._target_sources,
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
                # Treat filter rejections as non-error unverifiable for benchmarks
                if response.status_code in {403, 422}:
                    return EvaluatorResult(
                        claim=claim,
                        verdict=VerificationVerdict.UNVERIFIABLE,
                        trust_score=0.0,
                        latency_ms=latency_ms,
                        evaluator=self.name,
                        metadata={"api_status": response.status_code},
                    )

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
            trust_score_raw = data.get("trust_score", 0.0)
            if isinstance(trust_score_raw, dict):
                trust_score = float(
                    trust_score_raw.get("overall")
                    or trust_score_raw.get("score")
                    or 0.0
                )
            else:
                trust_score = float(trust_score_raw)
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
            
            # Extract evidence from trace
            evidence: list[EvidenceItem] = []
            for claim_data in claims_data:
                trace = claim_data.get("trace") or {}
                supporting = trace.get("supporting_evidence", []) or []
                refuting = trace.get("refuting_evidence", []) or []
                for ev in [*supporting, *refuting]:
                    evidence.append(
                        EvidenceItem(
                            text=(ev.get("content") or ev.get("text") or "")[:500],
                            source=ev.get("source", "unknown"),
                            similarity_score=ev.get("similarity_score", 0.0),
                            metadata=ev.get("structured_data", ev.get("metadata", {})),
                        )
                    )
            
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
                    "trust_score_raw": trust_score_raw,
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
                "use_cache": False,  # Disable cache for accurate benchmarking
            }
            
            headers = {}
            if self.config.ohi_api_key:
                headers["X-API-Key"] = self.config.ohi_api_key
            
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
                claim_status = claim_data.get("status", "unverifiable")

                # Map OHI status to verified boolean
                verified = str(claim_status).lower() == "supported"

                # Extract evidence from trace
                trace = claim_data.get("trace") or {}
                supporting = trace.get("supporting_evidence", []) or []
                refuting = trace.get("refuting_evidence", []) or []
                evidence = [
                    EvidenceItem(
                        text=(ev.get("content") or ev.get("text") or "")[:500],
                        source=ev.get("source", "unknown"),
                        similarity_score=ev.get("similarity_score", 0.0),
                        metadata=ev.get("structured_data", ev.get("metadata", {})),
                    )
                    for ev in [*supporting, *refuting]
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
