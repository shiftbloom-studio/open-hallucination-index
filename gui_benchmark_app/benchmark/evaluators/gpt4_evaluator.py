"""
GPT-4 Evaluator
===============

Direct LLM-based claim verification using OpenAI GPT-4.
This is a baseline comparator - expected to be slower and
less accurate than specialized verification systems like OHI.

Uses GPT-4 for both:
- Claim decomposition (atomic fact extraction)
- Fact verification (checking against its training data)
"""

from __future__ import annotations

import asyncio
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

# Prompt templates
VERIFICATION_PROMPT = """You are a fact-checking assistant. Analyze the following claim and determine if it is factually accurate.

Claim: {claim}

Respond with a JSON object containing:
- "verdict": one of "supported", "refuted", or "unverifiable"
- "confidence": a number between 0.0 and 1.0 indicating your confidence
- "reasoning": a brief explanation (max 100 words)

Important: Only mark as "supported" if you are highly confident the claim is factually accurate. Be conservative - when uncertain, use "unverifiable".

Respond ONLY with the JSON object, no other text."""

DECOMPOSITION_PROMPT = """Break down the following text into atomic facts. Each atomic fact should be:
- A single, self-contained statement
- Verifiable independently
- As simple as possible

Text: {text}

Respond with a JSON object containing:
- "facts": an array of strings, each being one atomic fact

Extract ALL factual claims from the text. Respond ONLY with the JSON object."""

FACT_VERIFICATION_PROMPT = """Verify if the following atomic fact is accurate:

Fact: {fact}

Respond with a JSON object containing:
- "verified": true if the fact is accurate, false if inaccurate or uncertain
- "confidence": a number between 0.0 and 1.0

Be strict: only mark as verified if you are confident the fact is correct.
Respond ONLY with the JSON object."""


class GPT4Evaluator(BaseEvaluator):
    """
    Evaluator using OpenAI GPT-4 for claim verification.

    Characteristics:
    - Uses LLM knowledge directly (no external retrieval)
    - Slower due to API latency
    - May hallucinate or have outdated information
    - Rate-limited by OpenAI API
    """

    name = "GPT-4"

    def __init__(self, config: ComparisonBenchmarkConfig) -> None:
        self.config = config
        self.openai_config = config.openai

        if not self.openai_config.is_configured:
            raise ValueError("OpenAI API key not configured")

        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = asyncio.Semaphore(
            self.openai_config.requests_per_minute // 3  # Conservative limit
        )
        self._last_request_time = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for OpenAI API."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.openai_config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.openai_config.timeout_seconds),
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        min_interval = 60.0 / self.openai_config.requests_per_minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    async def _chat_completion(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict | None:
        """
        Make a chat completion request to OpenAI.

        Returns parsed JSON response or None on error.
        """
        async with self._rate_limiter:
            await self._rate_limit()

            client = await self._get_client()

            payload = {
                "model": model or self.openai_config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
                if temperature is not None
                else self.openai_config.temperature,
                "max_tokens": self.openai_config.max_tokens,
                "response_format": {"type": "json_object"},
            }

            for attempt in range(self.openai_config.max_retries):
                try:
                    response = await client.post(
                        "/chat/completions",
                        json=payload,
                    )

                    if response.status_code == 429:
                        # Rate limited - exponential backoff
                        wait_time = (2**attempt) * 5
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        return None

                    data = response.json()
                    content = data["choices"][0]["message"]["content"]

                    # Parse JSON response
                    import json

                    return json.loads(content)

                except Exception:
                    if attempt == self.openai_config.max_retries - 1:
                        return None
                    await asyncio.sleep(2**attempt)

            return None

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.openai_config.is_configured:
            return False
        try:
            result = await self._chat_completion(
                'Respond with: {"status": "ok"}',
                temperature=0.0,
            )
            return result is not None
        except Exception:
            return False

    async def verify(self, claim: str) -> EvaluatorResult:
        """
        Verify a claim using GPT-4.

        GPT-4 relies on its training data knowledge,
        which may be outdated or hallucinated.
        """
        start_time = time.perf_counter()

        prompt = VERIFICATION_PROMPT.format(claim=claim)
        result = await self._chat_completion(prompt)

        latency_ms = (time.perf_counter() - start_time) * 1000

        if result is None:
            return EvaluatorResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                trust_score=0.0,
                latency_ms=latency_ms,
                evaluator=self.name,
                error="OpenAI API request failed",
            )

        # Parse result
        verdict_str = result.get("verdict", "unverifiable").lower()
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        # Map to verdict enum
        verdict_map = {
            "supported": VerificationVerdict.SUPPORTED,
            "refuted": VerificationVerdict.REFUTED,
            "unverifiable": VerificationVerdict.UNVERIFIABLE,
        }
        verdict = verdict_map.get(verdict_str, VerificationVerdict.UNVERIFIABLE)

        return EvaluatorResult(
            claim=claim,
            verdict=verdict,
            trust_score=confidence
            if verdict == VerificationVerdict.SUPPORTED
            else 1.0 - confidence,
            latency_ms=latency_ms,
            evaluator=self.name,
            evidence=[
                EvidenceItem(
                    text=reasoning,
                    source="GPT-4 reasoning",
                    similarity_score=confidence,
                )
            ]
            if reasoning
            else [],
            metadata={
                "model": self.openai_config.model,
                "raw_response": result,
            },
        )

    async def decompose_and_verify(self, text: str) -> FActScoreResult:
        """
        Decompose text into atomic facts using GPT-4,
        then verify each fact individually.

        This is a two-step process:
        1. GPT-4 decomposes text → list of atomic facts
        2. GPT-4 verifies each atomic fact
        """
        start_time = time.perf_counter()

        # Step 1: Decompose text into atomic facts
        decomp_prompt = DECOMPOSITION_PROMPT.format(text=text)
        decomp_result = await self._chat_completion(
            decomp_prompt,
            model=self.openai_config.decomposition_model,
            temperature=self.openai_config.decomposition_temperature,
        )

        if decomp_result is None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return FActScoreResult(
                original_text=text,
                atomic_facts=[],
                latency_ms=latency_ms,
                evaluator=self.name,
                error="Failed to decompose text",
            )

        facts_list = decomp_result.get("facts", [])

        if not facts_list:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return FActScoreResult(
                original_text=text,
                atomic_facts=[],
                latency_ms=latency_ms,
                evaluator=self.name,
            )

        # Step 2: Verify each atomic fact
        atomic_facts: list[AtomicFact] = []

        for i, fact_text in enumerate(facts_list):
            verify_prompt = FACT_VERIFICATION_PROMPT.format(fact=fact_text)
            verify_result = await self._chat_completion(verify_prompt)

            verified = False
            confidence = 0.0

            if verify_result:
                verified = verify_result.get("verified", False)
                confidence = float(verify_result.get("confidence", 0.5))

            atomic_facts.append(
                AtomicFact(
                    text=fact_text,
                    source_text=text,
                    index=i,
                    verified=verified,
                    evidence=[
                        EvidenceItem(
                            text=f"GPT-4 verification (confidence: {confidence:.2f})",
                            source="GPT-4",
                            similarity_score=confidence,
                        )
                    ]
                    if verified
                    else [],
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
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
