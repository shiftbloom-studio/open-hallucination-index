"""
VerifyTextUseCase
=================

Primary application use-case: orchestrates the full verification pipeline.

Flow:
1. Check cache for existing result
2. Decompose text into claims
3. Verify each claim against knowledge stores
4. Compute trust score
5. Cache and return result
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING

from open_hallucination_index.domain.results import (
    ClaimVerification,
    TrustScore,
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from open_hallucination_index.ports.cache import CacheProvider
    from open_hallucination_index.ports.claim_decomposer import ClaimDecomposer
    from open_hallucination_index.ports.scorer import Scorer
    from open_hallucination_index.ports.verification_oracle import (
        VerificationOracle,
        VerificationStrategy,
    )


class VerifyTextUseCase:
    """
    Orchestrates text verification through the hexagonal architecture ports.

    This use-case is the primary entry point for the verification pipeline.
    It coordinates:
    - ClaimDecomposer: Text → Claims
    - VerificationOracle: Claims → Verification statuses + traces
    - Scorer: Verifications → Trust score
    - CacheProvider: Result caching

    No concrete infrastructure dependencies are injected directly;
    only abstract ports are used.
    """

    def __init__(
        self,
        decomposer: ClaimDecomposer,
        oracle: VerificationOracle,
        scorer: Scorer,
        cache: CacheProvider | None = None,
    ) -> None:
        """
        Initialize the use-case with required ports.

        Args:
            decomposer: Claim extraction service.
            oracle: Claim verification service.
            scorer: Trust score computation service.
            cache: Optional result cache.
        """
        self._decomposer = decomposer
        self._oracle = oracle
        self._scorer = scorer
        self._cache = cache

    async def execute(
        self,
        text: str,
        *,
        strategy: VerificationStrategy | None = None,
        use_cache: bool = True,
        context: str | None = None,
    ) -> VerificationResult:
        """
        Execute the full verification pipeline.

        Args:
            text: Input text to verify.
            strategy: Optional verification strategy override.
            use_cache: Whether to check/update cache.
            context: Optional context for claim decomposition.

        Returns:
            Complete verification result with trust score and traces.
        """
        start_time = time.perf_counter()
        input_hash = self._compute_hash(text)

        # Step 1: Check cache
        if use_cache and self._cache is not None:
            cached = await self._cache.get(input_hash)
            if cached is not None:
                return VerificationResult(
                    id=cached.id,
                    input_hash=input_hash,
                    input_length=len(text),
                    trust_score=cached.trust_score,
                    claim_verifications=cached.claim_verifications,
                    summary=cached.summary,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    cached=True,
                )

        # Step 2: Decompose text into claims
        if context:
            claims = await self._decomposer.decompose_with_context(text, context)
        else:
            claims = await self._decomposer.decompose(text)

        # Step 3: Verify each claim
        if strategy:
            verification_results = await self._oracle.verify_claims(claims, strategy)
        else:
            verification_results = await self._oracle.verify_claims(claims)

        # Step 4: Build ClaimVerification objects with score contributions (parallel)
        # Create preliminary verifications for parallel score computation
        preliminaries = [
            ClaimVerification(
                claim=claim,
                status=status,
                trace=trace,
                score_contribution=0.0,  # Placeholder
            )
            for claim, (status, trace) in zip(claims, verification_results, strict=True)
        ]

        # Parallel computation of all claim contributions
        contributions = await asyncio.gather(
            *[self._scorer.compute_claim_contribution(p) for p in preliminaries]
        )

        # Build final ClaimVerification objects with actual contributions
        claim_verifications: list[ClaimVerification] = [
            ClaimVerification(
                claim=p.claim,
                status=p.status,
                trace=p.trace,
                score_contribution=contribution,
            )
            for p, contribution in zip(preliminaries, contributions, strict=True)
        ]

        # Step 5: Compute overall trust score
        trust_score = await self._scorer.compute_score(claim_verifications)

        # Step 6: Generate summary
        summary = self._generate_summary(claim_verifications, trust_score)

        # Step 7: Build result
        processing_time = (time.perf_counter() - start_time) * 1000
        result = VerificationResult(
            input_hash=input_hash,
            input_length=len(text),
            trust_score=trust_score,
            claim_verifications=claim_verifications,
            summary=summary,
            processing_time_ms=processing_time,
            cached=False,
        )

        # Step 8: Cache result
        if use_cache and self._cache is not None:
            await self._cache.set(input_hash, result)

        return result

    def _compute_hash(self, text: str) -> str:
        """Compute deterministic hash of input text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _generate_summary(
        self,
        verifications: list[ClaimVerification],
        score: TrustScore,
    ) -> str:
        """Generate human-readable summary of verification results."""
        if not verifications:
            return "No verifiable claims found in the input text."

        total = len(verifications)
        supported = sum(
            1 for v in verifications if v.status == VerificationStatus.SUPPORTED
        )
        refuted = sum(
            1 for v in verifications if v.status == VerificationStatus.REFUTED
        )
        unverifiable = sum(
            1 for v in verifications if v.status == VerificationStatus.UNVERIFIABLE
        )

        trust_level = (
            "high" if score.overall >= 0.8
            else "moderate" if score.overall >= 0.5
            else "low"
        )

        return (
            f"Analyzed {total} claim(s): {supported} supported, "
            f"{refuted} refuted, {unverifiable} unverifiable. "
            f"Overall trust level: {trust_level} ({score.overall:.2f})."
        )
