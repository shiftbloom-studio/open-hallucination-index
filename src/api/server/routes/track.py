"""
Knowledge Track API Endpoints
==============================

API for retrieving knowledge provenance tracks for verified claims.
Provides detailed explanation of sources and 3D mesh for visualization.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from config.dependencies import (
    get_knowledge_track_service,
)
from models.track import (
    EdgeType,
    KnowledgeMesh,
    KnowledgeTrackResult,
    NodeType,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Response Schemas
# -----------------------------------------------------------------------------


class KnowledgeNodeResponse(BaseModel):
    """A node in the 3D knowledge mesh."""

    id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Type of node")
    label: str = Field(..., description="Display label")
    source: str | None = Field(None, description="Origin source")
    source_uri: str | None = Field(None, description="Link to source")
    depth_level: int = Field(..., description="Depth from central claim")
    weight: float = Field(..., description="Node importance (0-1)")
    confidence: float | None = Field(None, description="Confidence score")
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeEdgeResponse(BaseModel):
    """An edge connecting two knowledge nodes."""

    id: str = Field(..., description="Unique edge identifier")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    edge_type: EdgeType = Field(..., description="Relationship type")
    weight: float = Field(..., description="Edge strength (0-1)")
    label: str | None = Field(None, description="Edge label")


class KnowledgeMeshResponse(BaseModel):
    """3D-visualization-ready knowledge graph."""

    center_claim_id: str = Field(..., description="Central claim ID")
    nodes: list[KnowledgeNodeResponse] = Field(default_factory=list)
    edges: list[KnowledgeEdgeResponse] = Field(default_factory=list)
    depth_levels: int = Field(..., description="Max depth traversed")
    total_sources_queried: int = Field(..., description="Sources consulted")
    total_evidence_found: int = Field(..., description="Evidence items found")
    layout_algorithm: str = Field(default="force-directed")
    node_count: int = Field(..., description="Total nodes")
    edge_count: int = Field(..., description="Total edges")


class SourceReferenceResponse(BaseModel):
    """Reference to an MCP source."""

    mcp_source: str = Field(..., description="Source name")
    source_description: str = Field(..., description="Human-readable description")
    url: str | None = Field(None, description="Direct URL to content")
    evidence_snippet: str | None = Field(None, max_length=500, description="Evidence excerpt")
    confidence: float = Field(..., ge=0.0, le=1.0)
    contributed: bool = Field(..., description="Whether source contributed evidence")
    query_time_ms: float | None = Field(None, description="Query latency")


class KnowledgeTrackResponse(BaseModel):
    """Complete knowledge track response."""

    id: str = Field(..., description="Track ID")
    claim_id: str = Field(..., description="Original claim ID")
    claim_text: str = Field(..., description="Original claim text")
    detail_text: str = Field(
        ...,
        description="LLM-generated explanation of which sources prove/deny the claim",
    )
    verification_status: str = Field(..., description="Verification outcome")
    verification_confidence: float = Field(..., ge=0.0, le=1.0)
    source_references: list[SourceReferenceResponse] = Field(
        ..., description="All MCP sources with URLs and descriptions"
    )
    mesh: KnowledgeMeshResponse = Field(..., description="3D knowledge graph for visualization")
    cached_until: str | None = Field(None, description="Cache expiry timestamp")
    generated_at: str = Field(..., description="Generation timestamp")

    @classmethod
    def from_domain(cls, result: KnowledgeTrackResult) -> KnowledgeTrackResponse:
        """Convert domain model to API response."""
        return cls(
            id=str(result.id),
            claim_id=str(result.claim_id),
            claim_text=result.claim_text,
            detail_text=result.detail_text,
            verification_status=result.verification_status,
            verification_confidence=result.verification_confidence,
            source_references=[
                SourceReferenceResponse(
                    mcp_source=ref.mcp_source,
                    source_description=ref.source_description,
                    url=ref.url,
                    evidence_snippet=ref.evidence_snippet,
                    confidence=ref.confidence,
                    contributed=ref.contributed,
                    query_time_ms=ref.query_time_ms,
                )
                for ref in result.source_references
            ],
            mesh=_mesh_to_response(result.mesh),
            cached_until=(result.cached_until.isoformat() if result.cached_until else None),
            generated_at=result.generated_at.isoformat(),
        )


class AvailableSourcesResponse(BaseModel):
    """List of available MCP sources."""

    sources: list[SourceReferenceResponse] = Field(..., description="All available MCP sources")
    total: int = Field(..., description="Total number of sources")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _mesh_to_response(mesh: KnowledgeMesh) -> KnowledgeMeshResponse:
    """Convert domain mesh to API response."""
    return KnowledgeMeshResponse(
        center_claim_id=str(mesh.center_claim_id),
        nodes=[
            KnowledgeNodeResponse(
                id=str(n.id),
                node_type=n.node_type,
                label=n.label,
                source=n.source,
                source_uri=n.source_uri,
                depth_level=n.depth_level,
                weight=n.weight,
                confidence=n.confidence,
                metadata=n.metadata,
            )
            for n in mesh.nodes
        ],
        edges=[
            KnowledgeEdgeResponse(
                id=str(e.id),
                source_node_id=str(e.source_node_id),
                target_node_id=str(e.target_node_id),
                edge_type=e.edge_type,
                weight=e.weight,
                label=e.label,
            )
            for e in mesh.edges
        ],
        depth_levels=mesh.depth_levels,
        total_sources_queried=mesh.total_sources_queried,
        total_evidence_found=mesh.total_evidence_found,
        layout_algorithm=mesh.layout_algorithm,
        node_count=mesh.node_count,
        edge_count=mesh.edge_count,
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.get(
    "/knowledge-track/{claim_id}",
    response_model=KnowledgeTrackResponse,
    status_code=status.HTTP_200_OK,
    summary="Get knowledge provenance track for a verified claim",
    description=(
        "Retrieves detailed provenance information for a previously verified claim. "
        "Returns LLM-generated explanation of which sources proved or denied the claim, "
        "URLs to all consulted MCP sources, and a 3D-visualization-ready knowledge mesh."
    ),
    responses={
        404: {
            "description": "Claim trace not found or expired",
            "content": {
                "application/json": {
                    "example": {"detail": "Knowledge track not found for claim ID"}
                }
            },
        },
    },
)
async def get_knowledge_track(
    claim_id: UUID,
    depth: Annotated[
        int,
        Query(
            ge=1,
            le=5,
            description="Depth of knowledge mesh (1-5, default 2)",
        ),
    ] = 2,
    generate_detail: Annotated[
        bool,
        Query(
            description="Generate LLM detail text (default true)",
        ),
    ] = True,
    service: Annotated[
        Any,  # KnowledgeTrackService
        Depends(get_knowledge_track_service),
    ] = None,
) -> KnowledgeTrackResponse:
    """
    Get knowledge provenance track for a verified claim.

    The track includes:
    - **detail_text**: LLM-generated explanation of which sources
      prove or deny which parts of the claim
    - **source_references**: All MCP sources consulted with URLs,
      descriptions, and evidence snippets
    - **mesh**: 3D-visualization-ready knowledge graph with nodes
      and edges representing claims, evidence, sources, and entities

    The track is cached for 12 hours after verification.
    """
    from services.track import (
        KnowledgeTrackService,
    )

    track_service: KnowledgeTrackService = service

    result = await track_service.get_knowledge_track(
        claim_id=claim_id,
        depth=depth,
        generate_detail=generate_detail,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Knowledge track not found for claim ID {claim_id}. "
                "The trace may have expired (12 hour TTL) or the claim was never verified."
            ),
        )

    return KnowledgeTrackResponse.from_domain(result)


@router.get(
    "/knowledge-track/sources/available",
    response_model=AvailableSourcesResponse,
    status_code=status.HTTP_200_OK,
    summary="List all available MCP sources",
    description="Returns a list of all MCP knowledge sources that can be consulted.",
)
async def list_available_sources(
    service: Annotated[
        Any,  # KnowledgeTrackService
        Depends(get_knowledge_track_service),
    ] = None,
) -> AvailableSourcesResponse:
    """
    List all available MCP knowledge sources.

    Returns descriptions and metadata for all sources that may be
    consulted during claim verification, including:
    - Wikipedia cluster (Wikipedia, Wikidata, DBpedia)
    - Academic sources (OpenAlex, Crossref, Europe PMC)
    - Medical sources (PubMed, ClinicalTrials.gov)
    - News (GDELT)
    - Economic (World Bank)
    - Security (OSV)
    - Local stores (Neo4j, Qdrant, Redis)
    """
    from services.track import (
        KnowledgeTrackService,
    )

    track_service: KnowledgeTrackService = service

    sources = track_service.list_available_sources()

    return AvailableSourcesResponse(
        sources=[
            SourceReferenceResponse(
                mcp_source=s.mcp_source,
                source_description=s.source_description,
                url=s.url,
                evidence_snippet=s.evidence_snippet,
                confidence=s.confidence,
                contributed=s.contributed,
                query_time_ms=s.query_time_ms,
            )
            for s in sources
        ],
        total=len(sources),
    )


@router.head(
    "/knowledge-track/{claim_id}",
    status_code=status.HTTP_200_OK,
    summary="Check if knowledge track exists",
    description="Check if a knowledge track exists for a claim without retrieving it.",
    responses={
        404: {"description": "Claim trace not found or expired"},
    },
)
async def check_knowledge_track_exists(
    claim_id: UUID,
    service: Annotated[
        Any,  # KnowledgeTrackService
        Depends(get_knowledge_track_service),
    ] = None,
) -> None:
    """
    Check if a knowledge track exists for a claim.

    Returns 200 if the track exists, 404 if not found or expired.
    """
    from services.track import (
        KnowledgeTrackService,
    )

    track_service: KnowledgeTrackService = service

    exists = await track_service.trace_exists(claim_id)

    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge track not found for claim ID {claim_id}",
        )
