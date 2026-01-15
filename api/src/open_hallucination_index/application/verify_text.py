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
    from open_hallucination_index.ports.knowledge_tracker import KnowledgeTracker
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
        trace_store: KnowledgeTracker | None = None,
    ) -> None:
        """
        Initialize the use-case with required ports.

        Args:
            decomposer: Claim extraction service.
            oracle: Claim verification service.
            scorer: Trust score computation service.
            cache: Optional result cache.
            trace_store: Optional trace storage for knowledge-track.
        """
        self._decomposer = decomposer
        self._oracle = oracle
        self._scorer = scorer
        self._cache = cache
        self._trace_store = trace_store

    async def execute(
        self,
        text: str,
        *,
        strategy: VerificationStrategy | None = None,
        use_cache: bool = True,
        context: str | None = None,
        target_sources: int | None = None,
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

        # Step 2b: Claim-level cache lookup (by claim hash)
        claim_hashes = [self._compute_claim_hash(claim.text) for claim in claims]
        claim_hash_by_id = {
            claim.id: claim_hash for claim, claim_hash in zip(claims, claim_hashes, strict=True)
        }
        cached_claims: dict[str, ClaimVerification] = {}

        if use_cache and self._cache is not None and claim_hashes:
            cached_map = await self._cache.get_claims_batch(claim_hashes)
            cached_claims = {h: v for h, v in cached_map.items() if v is not None}

        # Step 3: Verify each claim (skip cached claims)
        claims_to_verify: list = []
        results_by_claim_id: dict = {}
        cached_claim_ids: set = set()

        for claim, claim_hash in zip(claims, claim_hashes, strict=True):
            cached = cached_claims.get(claim_hash)
            if cached is not None:
                trace = cached.trace.model_copy(update={"claim_id": claim.id})
                results_by_claim_id[claim.id] = (cached.status, trace)
                cached_claim_ids.add(claim.id)
            else:
                claims_to_verify.append(claim)

        if claims_to_verify:
            if strategy:
                verification_results = await self._oracle.verify_claims(
                    claims_to_verify,
                    strategy,
                    target_sources=target_sources,
                )
            else:
                verification_results = await self._oracle.verify_claims(
                    claims_to_verify,
                    target_sources=target_sources,
                )

            for claim, (status, trace) in zip(claims_to_verify, verification_results, strict=True):
                results_by_claim_id[claim.id] = (status, trace)

        # Step 4: Build ClaimVerification objects with score contributions (parallel)
        # Create preliminary verifications for parallel score computation
        preliminaries = [
            ClaimVerification(
                claim=claim,
                status=status,
                trace=trace,
                score_contribution=0.0,  # Placeholder
            )
            for claim in claims
            for (status, trace) in [results_by_claim_id[claim.id]]
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

        # Step 8: Cache result + claim-level cache
        if use_cache and self._cache is not None:
            await self._cache.set(input_hash, result)

            claim_cache_entries = []
            for verification in claim_verifications:
                claim_hash = claim_hash_by_id.get(verification.claim.id)
                if claim_hash is None:
                    continue
                if claim_hash in cached_claims:
                    continue
                claim_cache_entries.append((claim_hash, verification))

            if claim_cache_entries:
                await self._cache.set_claims_batch(claim_cache_entries)

        # Step 9: Record traces for knowledge-track (12h TTL)
        if self._trace_store is not None:
            asyncio.create_task(
                self._record_traces(claim_verifications, processing_time, cached_claim_ids)
            )

        return result

    async def _record_traces(
        self,
        verifications: list[ClaimVerification],
        total_time_ms: float,
        cached_claim_ids: set,
    ) -> None:
        """Record traces for all verified claims."""
        from open_hallucination_index.domain.knowledge_track import TraceData

        if self._trace_store is None:
            return

        traces = []
        for v in verifications:
            supporting = [ev.model_dump(mode="json") for ev in v.trace.supporting_evidence]
            refuting = [ev.model_dump(mode="json") for ev in v.trace.refuting_evidence]
            evidence_sources = self._extract_sources_from_evidence(
                v.trace.supporting_evidence + v.trace.refuting_evidence
            )
            contributed_sources = self._extract_sources_from_evidence(
                v.trace.supporting_evidence + v.trace.refuting_evidence,
                include_local=True,
            )

            if v.claim.id in cached_claim_ids:
                evidence_sources.add("redis")
                contributed_sources.add("redis")

            mcp_calls = self._build_mcp_calls(
                v.trace.supporting_evidence, "supporting"
            ) + self._build_mcp_calls(v.trace.refuting_evidence, "refuting")

            trace = TraceData(
                claim_id=v.claim.id,
                claim_text=v.claim.text,
                verification_status=v.status.value,
                verification_confidence=v.trace.confidence,
                verification_strategy=v.trace.verification_strategy,
                supporting_evidence=supporting,
                refuting_evidence=refuting,
                mcp_calls=mcp_calls,
                sources_queried=sorted(evidence_sources),
                sources_contributed=sorted(contributed_sources),
                query_times_ms={},
                total_time_ms=total_time_ms,
                reasoning=v.trace.reasoning,
            )
            traces.append(trace)

        if traces:
            await self._trace_store.record_traces_batch(traces)

    def _extract_sources_from_evidence(
        self,
        evidence_list: list,
        *,
        include_local: bool = False,
    ) -> set[str]:
        sources: set[str] = set()
        for ev in evidence_list:
            source_name = self._normalize_source_name(ev)
            if source_name:
                sources.add(source_name)

            structured = ev.structured_data or {}
            aliases = structured.get("source_aliases")
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias:
                        sources.add(alias.lower().replace("-", "_"))

            if include_local:
                local_source = self._map_local_source(ev)
                if local_source:
                    sources.add(local_source)

        return sources

    def _normalize_source_name(self, evidence) -> str | None:
        structured = evidence.structured_data or {}
        mcp_source = structured.get("mcp_source") or structured.get("original_source")
        if isinstance(mcp_source, str) and mcp_source:
            return mcp_source.lower().replace("-", "_")

        if evidence.source.value == "mcp_wikipedia":
            return "wikipedia"
        if evidence.source.value == "mcp_context7":
            return "context7"
        if evidence.source.value == "pubmed":
            return "pubmed"

        return None

    def _map_local_source(self, evidence) -> str | None:
        if evidence.source.value in {"graph_exact", "graph_inferred"}:
            return "neo4j"
        if evidence.source.value == "vector_semantic":
            return "qdrant"
        if evidence.source.value == "cached":
            return "redis"
        return None

    def _build_mcp_calls(
        self,
        evidence_list: list,
        evidence_type: str,
    ) -> list[dict]:
        calls: list[dict] = []
        for ev in evidence_list:
            source_name = self._normalize_source_name(ev)
            if not source_name:
                source_name = self._map_local_source(ev) or "unknown"

            structured = ev.structured_data or {}
            calls.append(
                {
                    "source": source_name,
                    "tool": structured.get("tool"),
                    "content": ev.content,
                    "url": ev.source_uri,
                    "confidence": ev.similarity_score,
                    "match_type": ev.match_type,
                    "evidence_type": evidence_type,
                }
            )

        return calls

    def _compute_hash(self, text: str) -> str:
        """Compute deterministic hash of input text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _compute_claim_hash(self, claim_text: str) -> str:
        """Compute deterministic hash of a claim text."""
        normalized = " ".join(claim_text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:24]

    def _generate_summary(
        self,
        verifications: list[ClaimVerification],
        score: TrustScore,
    ) -> str:
        """Generate human-readable summary of verification results."""
        if not verifications:
            return "No verifiable claims found in the input text."

        total = len(verifications)
        supported = sum(1 for v in verifications if v.status == VerificationStatus.SUPPORTED)
        refuted = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
        unverifiable = sum(1 for v in verifications if v.status == VerificationStatus.UNVERIFIABLE)

        trust_level = (
            "high" if score.overall >= 0.8 else "moderate" if score.overall >= 0.5 else "low"
        )

        return (
            f"Analyzed {total} claim(s): {supported} supported, "
            f"{refuted} refuted, {unverifiable} unverifiable. "
            f"Overall trust level: {trust_level} ({score.overall:.2f})."
        )
