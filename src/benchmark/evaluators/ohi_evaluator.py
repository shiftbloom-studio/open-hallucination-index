"""
OHI Evaluator
=============

Evaluator using the Open Hallucination Index API.
This is our primary system - expected to outperform baselines.
"""

from __future__ import annotations

import asyncio
import logging
import time

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

logger = logging.getLogger(__name__)


class OHIEvaluator(BaseEvaluator):
    """
    Evaluator using the OHI API for claim verification.
    
    Uses the full OHI pipeline:
    - Hybrid graph + vector search
    - MCP knowledge sources (based on tier)
    - Claim decomposition
    - Evidence aggregation
    
    Tiers:
    - local: Only local sources (Neo4j + Qdrant) - fastest
    - default: Local first, MCP fallback if insufficient
    - max: All sources for maximum evidence coverage
    """
    
    name = "OHI"
    
    def __init__(
        self,
        config: ComparisonBenchmarkConfig,
        *,
        name_override: str | None = None,
        strategy_override: str | None = None,
        target_sources_override: int | None = None,
        tier: str = "default",  # local, default, or max
    ) -> None:
        self.config = config
        self.base_url = config.ohi_api_base_url
        self.verify_url = config.ohi_verify_url
        self.strategy = strategy_override or config.ohi_strategy
        self.tier = tier
        if name_override:
            self.name = name_override
        self.timeout = config.timeout_seconds
        self._target_sources = target_sources_override or 5  # Reduced for stability
        # Keep connection pool small for stability
        self.max_concurrency = min(5, config.concurrency)
        self.max_retries = 3  # Retry failed requests
        self.retry_delay = 0.5  # Initial retry delay in seconds
        
        # Log API key status
        if config.ohi_api_key:
            logger.info(f"{self.name}: API key configured (length: {len(config.ohi_api_key)})")
        else:
            logger.warning(f"{self.name}: No API key configured - ensure API_API_KEY env var is set")
        
        # Persistent HTTP client
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()  # Ensure thread-safe client access
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with proper connection pooling."""
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                logger.info(
                    f"{self.name}: Creating HTTP client (max_connections={self.max_concurrency}, "
                    f"timeout={self.timeout}s, base_url={self.base_url})"
                )
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0,  # Connection timeout
                        read=self.timeout,  # Read timeout
                        write=30.0,  # Write timeout
                        pool=5.0,  # Pool acquisition timeout
                    ),
                    limits=httpx.Limits(
                        max_connections=self.max_concurrency,
                        max_keepalive_connections=self.max_concurrency // 2,
                        keepalive_expiry=30.0,  # Keep connections alive for 30s
                    ),
                    # Follow redirects
                    follow_redirects=True,
                    # Retry on connection errors
                    transport=httpx.AsyncHTTPTransport(retries=1),
                )
            return self._client
    
    async def close(self) -> None:
        """Close the HTTP client and free resources."""
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                logger.info(f"{self.name}: Closing HTTP client")
                await self._client.aclose()
                self._client = None
    
    async def health_check(self) -> bool:
        """Check if OHI API is healthy."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def _make_request(
        self,
        payload: dict,
        headers: dict,
        attempt: int = 1,
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            payload: Request payload
            headers: Request headers
            attempt: Current retry attempt (1-indexed)
            
        Returns:
            httpx.Response object
            
        Raises:
            httpx.HTTPError: If all retries fail
        """
        client = await self._get_client()
        
        try:
            response = await client.post(
                self.verify_url,
                json=payload,
                headers=headers,
            )
            return response
            
        except (httpx.ConnectError, httpx.PoolTimeout, httpx.ConnectTimeout) as e:
            if attempt >= self.max_retries:
                logger.error(
                    f"{self.name}: Connection failed after {attempt} attempts: {type(e).__name__}: {e}"
                )
                raise
            
            # Exponential backoff
            delay = self.retry_delay * (2 ** (attempt - 1))
            logger.warning(
                f"{self.name}: Connection error (attempt {attempt}/{self.max_retries}), "
                f"retrying in {delay:.1f}s: {type(e).__name__}: {e}"
            )
            await asyncio.sleep(delay)
            return await self._make_request(payload, headers, attempt + 1)
        
        except httpx.TimeoutException as e:
            logger.error(f"{self.name}: Request timeout: {e}")
            raise
        
        except Exception as e:
            logger.error(f"{self.name}: Unexpected error: {type(e).__name__}: {e}")
            raise

    async def verify(self, claim: str) -> EvaluatorResult:
        """
        Verify a claim using OHI API with retry logic.
        
        Args:
            claim: The claim text to verify.
            
        Returns:
            EvaluatorResult with OHI verification results.
        """
        start_time = time.perf_counter()
        
        try:
            payload = {
                "text": claim,
                "strategy": self.strategy,
                "tier": self.tier,  # Evidence collection tier (local, default, max)
                "return_evidence": True,
                "target_sources": self._target_sources,
                "use_cache": False,  # Disable cache for accurate benchmarking
                "skip_decomposition": True,  # Treat input as single claim, no decomposition
            }
            
            headers = {}
            if self.config.ohi_api_key:
                headers["X-API-Key"] = self.config.ohi_api_key
            else:
                logger.debug(f"{self.name}: Making request without API key")
            
            # Disable pre-checks for benchmarking
            headers["X-Benchmark-Mode"] = "true"
            
            response = await self._make_request(payload, headers)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if response.status_code != 200:
                # Log response details for debugging
                try:
                    error_detail = response.text[:500]  # First 500 chars
                except Exception:
                    error_detail = "<unable to read response>"
                
                logger.error(
                    f"{self.name}: API error - Status {response.status_code}, "
                    f"URL: {self.verify_url}, Detail: {error_detail}"
                )
                
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
                
                # 401 indicates authentication failure
                if response.status_code == 401:
                    return EvaluatorResult(
                        claim=claim,
                        verdict=VerificationVerdict.UNVERIFIABLE,
                        trust_score=0.0,
                        latency_ms=latency_ms,
                        evaluator=self.name,
                        error="Authentication failed (401) - check API_API_KEY environment variable",
                    )

                return EvaluatorResult(
                    claim=claim,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    trust_score=0.0,
                    latency_ms=latency_ms,
                    evaluator=self.name,
                    error=f"API error {response.status_code}: {error_detail[:100]}",
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
            
        except httpx.TimeoutException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{self.name}: Request timeout after {latency_ms:.0f}ms: {e}"
            )
            return EvaluatorResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                trust_score=0.0,
                latency_ms=latency_ms,
                evaluator=self.name,
                error=f"Request timeout after {latency_ms:.0f}ms",
            )
        except (httpx.ConnectError, httpx.PoolTimeout, httpx.NetworkError) as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{self.name}: Connection error: {type(e).__name__}: {e}"
            )
            return EvaluatorResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                trust_score=0.0,
                latency_ms=latency_ms,
                evaluator=self.name,
                error=f"Connection error: {type(e).__name__}",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{self.name}: Unexpected error: {type(e).__name__}: {e}",
                exc_info=True,  # Include stack trace
            )
            return EvaluatorResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                trust_score=0.0,
                latency_ms=latency_ms,
                evaluator=self.name,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
    
    async def decompose_and_verify(self, text: str) -> FActScoreResult:
        """
        Decompose text and verify each atomic fact using OHI.
        
        OHI's claim decomposer extracts atomic facts,
        then each is verified individually.
        """
        start_time = time.perf_counter()
        
        try:
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
            
            # Disable pre-checks for benchmarking
            headers["X-Benchmark-Mode"] = "true"
            
            response = await self._make_request(payload, headers)
            
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
                # Both "supported" and "partially_supported" count as verified for FActScore
                status_lower = str(claim_status).lower().replace("_", "-")
                verified = status_lower in ("supported", "partially-supported", "partially_supported")

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
