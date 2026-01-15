"""
Knowledge Track Service
=======================

Application service for retrieving knowledge provenance tracks.
Handles LLM-generated detail text and mesh building.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from open_hallucination_index.domain.knowledge_track import (
    KnowledgeTrackResult,
    MCP_SOURCE_DESCRIPTIONS,
    SourceReference,
    TraceData,
)
from open_hallucination_index.ports.llm_provider import LLMMessage

if TYPE_CHECKING:
    from open_hallucination_index.domain.services.mesh_builder import (
        KnowledgeMeshBuilder,
    )
    from open_hallucination_index.ports.knowledge_tracker import KnowledgeTracker
    from open_hallucination_index.ports.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# Default trace TTL: 12 hours
DEFAULT_TRACE_TTL = 43200


class KnowledgeTrackService:
    """
    Service for retrieving and building knowledge tracks.

    Orchestrates:
    - Trace retrieval from Redis
    - 3D mesh building
    - LLM detail text generation
    - Source reference aggregation
    """

    def __init__(
        self,
        trace_store: KnowledgeTracker,
        mesh_builder: KnowledgeMeshBuilder,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """
        Initialize the service.

        Args:
            trace_store: Redis trace storage adapter.
            mesh_builder: Service for building 3D knowledge meshes.
            llm_provider: Optional LLM for generating detail text.
        """
        self._trace_store = trace_store
        self._mesh_builder = mesh_builder
        self._llm = llm_provider

    async def get_knowledge_track(
        self,
        claim_id: UUID,
        depth: int = 2,
        generate_detail: bool = True,
    ) -> KnowledgeTrackResult | None:
        """
        Retrieve a complete knowledge track for a verified claim.

        Args:
            claim_id: UUID of the claim to retrieve track for.
            depth: Mesh depth for 3D visualization (1-5).
            generate_detail: Whether to generate LLM detail text.

        Returns:
            KnowledgeTrackResult or None if claim not found.
        """
        start_time = time.perf_counter()

        # Get trace from Redis
        trace = await self._trace_store.get_trace(claim_id)
        if trace is None:
            logger.debug(f"No trace found for claim {claim_id}")
            return None

        # Build 3D mesh
        mesh = await self._mesh_builder.build_mesh(claim_id, depth)
        if mesh is None:
            logger.warning(f"Failed to build mesh for claim {claim_id}")
            return None

        # Build source references
        source_references = self._build_source_references(trace)

        # Generate detail text on demand
        if generate_detail and self._llm is not None:
            detail_text = await self._generate_detail_text(trace, source_references)
        else:
            detail_text = self._build_fallback_detail_text(trace)

        # Calculate cache expiry (estimate based on TTL)
        cached_until = datetime.now(UTC) + timedelta(seconds=DEFAULT_TRACE_TTL)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Built knowledge track for claim {claim_id} in {elapsed_ms:.1f}ms"
        )

        return KnowledgeTrackResult(
            claim_id=claim_id,
            claim_text=trace.claim_text,
            detail_text=detail_text,
            verification_status=trace.verification_status,
            verification_confidence=trace.verification_confidence,
            source_references=source_references,
            mesh=mesh,
            cached_until=cached_until,
        )

    def _build_source_references(self, trace: TraceData) -> list[SourceReference]:
        """Build source references from trace data."""
        references: list[SourceReference] = []
        seen_sources: set[str] = set()

        # Add references from MCP call logs
        for call in trace.mcp_calls:
            source_name = call.get("source", call.get("tool", "unknown"))
            normalized_name = source_name.lower().replace("search_", "").replace(
                "get_", ""
            )

            if normalized_name in seen_sources:
                continue
            seen_sources.add(normalized_name)

            url = call.get("url") or call.get("source_uri")
            snippet = str(call.get("content", call.get("response", "")))[:500]
            query_time = call.get("duration_ms") or 0.0
            confidence_val = call.get("confidence")
            confidence = float(confidence_val) if confidence_val is not None else 0.5

            references.append(
                SourceReference(
                    mcp_source=normalized_name,
                    source_description=MCP_SOURCE_DESCRIPTIONS.get(
                        normalized_name,
                        f"{source_name} - External knowledge source",
                    ),
                    url=url if url else None,
                    evidence_snippet=snippet if snippet else None,
                    confidence=confidence,
                    contributed=source_name in trace.sources_contributed
                    or normalized_name in trace.sources_contributed,
                    query_time_ms=float(query_time) if query_time else None,
                )
            )

        # Add all queried sources not in MCP calls
        for source_name in trace.sources_queried:
            normalized_name = source_name.lower()
            if normalized_name not in seen_sources:
                seen_sources.add(normalized_name)
                references.append(
                    SourceReference(
                        mcp_source=normalized_name,
                        source_description=MCP_SOURCE_DESCRIPTIONS.get(
                            normalized_name,
                            f"{source_name} - External knowledge source",
                        ),
                        url=None,
                        evidence_snippet=None,
                        confidence=0.0,
                        contributed=source_name in trace.sources_contributed,
                        query_time_ms=trace.query_times_ms.get(source_name),
                    )
                )

        # Always include local sources
        local_sources = ["neo4j", "qdrant", "redis"]
        for local in local_sources:
            if local not in seen_sources:
                references.append(
                    SourceReference(
                        mcp_source=local,
                        source_description=MCP_SOURCE_DESCRIPTIONS.get(
                            local, f"{local} - Local knowledge store"
                        ),
                        url=None,
                        evidence_snippet=None,
                        confidence=0.5,
                        contributed=local in trace.sources_contributed,
                        query_time_ms=trace.query_times_ms.get(local),
                    )
                )

        # Sort: contributing sources first, then by confidence
        references.sort(key=lambda r: (not r.contributed, -r.confidence))

        return references

    async def _generate_detail_text(
        self,
        trace: TraceData,
        sources: list[SourceReference],
    ) -> str:
        """Generate LLM detail text explaining the verification."""
        if self._llm is None:
            return self._build_fallback_detail_text(trace)

        try:
            # Build prompt for LLM
            prompt = self._build_detail_prompt(trace, sources)

            messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are an expert fact-checker. Provide a clear, concise "
                        "explanation of which sources prove or deny parts of a claim. "
                        "Use specific evidence and cite sources by name. "
                        "Keep the explanation under 500 words."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ]

            response = await self._llm.complete(
                messages,
                temperature=0.3,
                max_tokens=800,
            )

            return response.content

        except Exception as e:
            logger.warning(f"LLM detail generation failed: {e}")
            return self._build_fallback_detail_text(trace)

    def _build_detail_prompt(
        self,
        trace: TraceData,
        sources: list[SourceReference],
    ) -> str:
        """Build the prompt for LLM detail text generation."""
        supporting_summary = self._summarize_evidence(trace.supporting_evidence)
        refuting_summary = self._summarize_evidence(trace.refuting_evidence)
        source_list = ", ".join(s.mcp_source for s in sources if s.contributed)

        prompt = f"""Explain the verification of this claim:

CLAIM: "{trace.claim_text}"

VERIFICATION STATUS: {trace.verification_status}
CONFIDENCE: {trace.verification_confidence:.0%}

SOURCES CONSULTED: {source_list or "local knowledge bases only"}

SUPPORTING EVIDENCE:
{supporting_summary}

REFUTING EVIDENCE:
{refuting_summary}

Explain which sources provided what evidence and how they support or refute the claim.
Be specific about which parts of the claim are verified and by which sources."""

        return prompt

    def _summarize_evidence(
        self,
        evidence_list: list[dict],
        max_items: int = 5,
    ) -> str:
        """Summarize evidence list for prompt."""
        if not evidence_list:
            return "None found."

        summaries = []
        for ev in evidence_list[:max_items]:
            source = ev.get("source", "unknown")
            content = str(ev.get("content", ""))[:200]
            score = ev.get("similarity_score", "N/A")
            summaries.append(f"- [{source}] (score: {score}): {content}")

        return "\n".join(summaries)

    def _build_fallback_detail_text(self, trace: TraceData) -> str:
        """Build fallback detail text without LLM."""
        status = trace.verification_status.upper()
        confidence = trace.verification_confidence * 100

        supporting_count = len(trace.supporting_evidence)
        refuting_count = len(trace.refuting_evidence)

        sources_used = trace.sources_contributed or trace.sources_queried[:5]
        sources_str = ", ".join(sources_used) if sources_used else "local stores"

        text = (
            f"The claim was verified as {status} with {confidence:.0f}% confidence. "
            f"Evidence was gathered from {sources_str}. "
        )

        if supporting_count > 0:
            text += f"Found {supporting_count} supporting evidence item(s). "

        if refuting_count > 0:
            text += f"Found {refuting_count} refuting evidence item(s). "

        if trace.reasoning:
            text += f"\n\nReasoning: {trace.reasoning}"

        return text

    async def trace_exists(self, claim_id: UUID) -> bool:
        """Check if a trace exists for the given claim."""
        return await self._trace_store.trace_exists(claim_id)

    @staticmethod
    def list_available_sources() -> list[SourceReference]:
        """List all available MCP sources with descriptions."""
        return [
            SourceReference(
                mcp_source=name,
                source_description=desc,
                url=None,
                evidence_snippet=None,
                confidence=0.0,
                contributed=False,
                query_time_ms=None,
            )
            for name, desc in MCP_SOURCE_DESCRIPTIONS.items()
        ]
