"""
Knowledge Track Domain Models
==============================

Domain entities for knowledge provenance tracking and 3D mesh visualization.
These models capture the full trace of how claims are verified, including
all MCP sources consulted and their relationships.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum, auto
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Type classification for knowledge graph nodes."""

    CLAIM = auto()  # The central claim being verified
    ENTITY = auto()  # Named entity (person, place, organization, etc.)
    EVIDENCE = auto()  # Supporting or refuting evidence
    SOURCE = auto()  # MCP source or knowledge base
    CONCEPT = auto()  # Abstract concept or topic
    FACT = auto()  # Known fact from knowledge base


class EdgeType(StrEnum):
    """Relationship types between knowledge nodes."""

    SUPPORTS = auto()  # Evidence supports claim
    REFUTES = auto()  # Evidence refutes claim
    MENTIONS = auto()  # Source mentions entity
    DERIVED_FROM = auto()  # Evidence derived from source
    RELATED_TO = auto()  # General semantic relation
    PART_OF = auto()  # Hierarchical relationship
    VERIFIED_BY = auto()  # Claim verified by source
    CONFLICTS_WITH = auto()  # Contradicting evidence


class KnowledgeNode(BaseModel):
    """
    A node in the knowledge provenance graph.

    Represents claims, entities, evidence, or sources that can be
    rendered in 3D visualization.
    """

    id: UUID = Field(default_factory=uuid4)
    node_type: NodeType
    label: str = Field(..., description="Display label for the node")
    source: str | None = Field(
        default=None, description="Origin source (e.g., 'wikipedia', 'neo4j')"
    )
    source_uri: str | None = Field(default=None, description="URI link to source")

    # 3D positioning hints (for visualization layout)
    depth_level: int = Field(default=0, ge=0, description="Depth from center claim (0 = claim)")
    weight: float = Field(default=1.0, ge=0.0, description="Node importance weight")

    # Metadata for rendering
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class KnowledgeEdge(BaseModel):
    """
    An edge connecting two knowledge nodes.

    Represents relationships between claims, evidence, sources, and entities.
    """

    id: UUID = Field(default_factory=uuid4)
    source_node_id: UUID = Field(..., description="ID of source node")
    target_node_id: UUID = Field(..., description="ID of target node")
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Edge strength/confidence")
    label: str | None = Field(default=None, description="Optional edge label")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class KnowledgeMesh(BaseModel):
    """
    3D-visualization-ready knowledge graph structure.

    Contains nodes and edges representing the provenance network
    for a verified claim. Designed to be rendered in WebGL/Three.js.
    """

    center_claim_id: UUID = Field(..., description="ID of the central claim")
    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)

    # Graph metadata
    depth_levels: int = Field(default=2, ge=1, le=5, description="Max depth from claim")
    total_sources_queried: int = Field(default=0, ge=0)
    total_evidence_found: int = Field(default=0, ge=0)

    # Layout hints for 3D rendering
    layout_algorithm: str = Field(
        default="force-directed", description="Suggested layout algorithm"
    )

    model_config = {"frozen": True}

    @property
    def node_count(self) -> int:
        """Total number of nodes in mesh."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Total number of edges in mesh."""
        return len(self.edges)


class MCPSource(StrEnum):
    """Available MCP sources with their descriptions."""

    # Wikipedia cluster
    WIKIPEDIA = "wikipedia"
    WIKIDATA = "wikidata"
    DBPEDIA = "dbpedia"
    MEDIAWIKI = "mediawiki"
    WIKIMEDIA_REST = "wikimedia_rest"

    # Academic cluster
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    EUROPEPMC = "europepmc"
    OPENCITATIONS = "opencitations"

    # Medical cluster
    PUBMED = "pubmed"
    NCBI = "ncbi"
    CLINICAL_TRIALS = "clinical_trials"

    # Documentation source
    CONTEXT7 = "context7"

    # Infrastructure / storage cluster
    NEO4J = "neo4j"
    QDRANT = "qdrant"
    REDIS = "redis"

# Human-readable descriptions for each MCP source
MCP_SOURCE_DESCRIPTIONS: dict[str, str] = {
    "wikipedia": "Wikipedia - Free encyclopedia with crowd-sourced knowledge",
    "wikidata": "Wikidata - Structured knowledge base with linked data",
    "dbpedia": "DBpedia - Structured information extracted from Wikipedia",
    "mediawiki": "MediaWiki API - Wikipedia's native API for article content",
    "wikimedia_rest": "Wikimedia REST API - Wikipedia summaries and metadata",
    "openalex": "OpenAlex - Open catalog of scholarly works and citations",
    "crossref": "Crossref - DOI metadata and academic publication records",
    "europepmc": "Europe PMC - European life sciences literature database",
    "opencitations": "OpenCitations - Open citation data for academic works",
    "pubmed": "PubMed - Biomedical literature from MEDLINE and life science journals",
    "ncbi": "NCBI - National Center for Biotechnology Information databases",
    "clinical_trials": "ClinicalTrials.gov - Registry of clinical studies",
    "context7": "Context7 - Library and framework documentation",
    "neo4j": "Neo4j Graph - Local knowledge graph database",
    "qdrant": "Qdrant Vector - Semantic similarity search",
    "redis": "Redis Cache - Cached verification results",
}


class SourceReference(BaseModel):
    """
    Reference to an MCP source that contributed to verification.

    Contains URL, description, and evidence snippet from the source.
    """

    mcp_source: MCPSource | str = Field(
        ...,
        description="Name of the MCP source (known MCPSource or custom string)",
    )
    source_description: str = Field(..., description="Human-readable description of the source")
    url: str | None = Field(default=None, description="Direct URL to the source content")
    evidence_snippet: str | None = Field(
        default=None, max_length=500, description="Short excerpt of relevant evidence"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in this source")
    contributed: bool = Field(
        default=False, description="Whether this source contributed to the final decision"
    )
    query_time_ms: float | None = Field(
        default=None, ge=0.0, description="Time to query this source"
    )

    model_config = {"frozen": True}


class KnowledgeTrackResult(BaseModel):
    """
    Complete knowledge track response for a verified claim.

    Contains LLM-generated explanation, all source references,
    and 3D mesh structure for visualization.
    """

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID = Field(..., description="ID of the original verified claim")
    claim_text: str = Field(..., description="Original claim text")

    # LLM-generated detailed explanation
    detail_text: str = Field(
        ...,
        description="LLM-generated explanation of which sources prove/deny which parts",
    )

    # Verification outcome reference
    verification_status: str = Field(..., description="Original verification status")
    verification_confidence: float = Field(..., ge=0.0, le=1.0)

    # All MCP sources with references
    source_references: list[SourceReference] = Field(
        default_factory=list, description="All MCP sources consulted with URLs and descriptions"
    )

    # 3D mesh for visualization
    mesh: KnowledgeMesh = Field(..., description="3D-ready knowledge graph structure")

    # Cache metadata
    cached_until: datetime | None = Field(
        default=None, description="When this track expires from cache"
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}

    @property
    def contributing_sources(self) -> list[SourceReference]:
        """Sources that contributed to the verification decision."""
        return [ref for ref in self.source_references if ref.contributed]

    @property
    def source_count(self) -> int:
        """Total number of sources consulted."""
        return len(self.source_references)


class TraceData(BaseModel):
    """
    Internal trace data stored in Redis during verification.

    Captures all MCP calls, evidence, and timing for later retrieval.
    """

    claim_id: UUID
    claim_text: str
    verification_status: str
    verification_confidence: float
    verification_strategy: str

    # Evidence chains
    supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    refuting_evidence: list[dict[str, Any]] = Field(default_factory=list)

    # MCP call logs
    mcp_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log of all MCP tool calls with responses",
    )

    # Source metadata
    sources_queried: list[str] = Field(default_factory=list)
    sources_contributed: list[str] = Field(default_factory=list)

    # Timing
    query_times_ms: dict[str, float] = Field(default_factory=dict)
    total_time_ms: float = Field(default=0.0, ge=0.0)

    # Neo4j/Qdrant relationships for mesh building
    entity_relationships: list[dict[str, Any]] = Field(default_factory=list)
    semantic_neighbors: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reasoning: str = Field(default="")

    model_config = {"frozen": False}  # Mutable for building during verification
