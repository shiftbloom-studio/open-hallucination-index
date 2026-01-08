"""
Domain Entities
===============

Core business objects representing claims and evidence.
These are immutable value objects with no infrastructure dependencies.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum, auto
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ClaimType(StrEnum):
    """Classification of claim structure."""

    SUBJECT_PREDICATE_OBJECT = auto()  # "Paris is the capital of France"
    TEMPORAL = auto()  # "Einstein was born in 1879"
    QUANTITATIVE = auto()  # "The Eiffel Tower is 330 meters tall"
    COMPARATIVE = auto()  # "Python is faster than Ruby for X"
    CAUSAL = auto()  # "Smoking causes lung cancer"
    DEFINITIONAL = auto()  # "A mammal is a warm-blooded vertebrate"
    EXISTENTIAL = auto()  # "There exists a planet with water"
    UNCLASSIFIED = auto()


class Claim(BaseModel):
    """
    An atomic, verifiable assertion extracted from text.

    Represents a single factual claim in subject-predicate-object form
    or other structured representation that can be checked against
    a knowledge base.
    """

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., description="Original claim text as extracted")
    claim_type: ClaimType = Field(default=ClaimType.UNCLASSIFIED)

    # Structured representation (if decomposed to triplet)
    subject: str | None = Field(default=None, description="Subject entity")
    predicate: str | None = Field(default=None, description="Relationship/property")
    object: str | None = Field(default=None, description="Object entity or value")

    # Metadata
    source_span: tuple[int, int] | None = Field(
        default=None, description="Character offsets in original text"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Decomposer's confidence in extraction"
    )
    normalized_form: str | None = Field(
        default=None, description="Canonicalized form for matching"
    )

    model_config = {"frozen": True}


class EvidenceSource(StrEnum):
    """Origin of supporting/refuting evidence."""

    GRAPH_EXACT = auto()  # Exact match in knowledge graph
    GRAPH_INFERRED = auto()  # Inferred via graph traversal
    VECTOR_SEMANTIC = auto()  # Semantic similarity match
    EXTERNAL_API = auto()  # External fact-check API
    CACHED = auto()  # Retrieved from semantic cache
    MCP_WIKIPEDIA = auto()  # Wikipedia via MCP server
    MCP_CONTEXT7 = auto()  # Context7 documentation via MCP server
    
    # Unified OHI MCP Server sources
    WIKIPEDIA = auto()  # Wikipedia/Wikidata/DBpedia
    KNOWLEDGE_GRAPH = auto()  # Generic knowledge graph
    ACADEMIC = auto()  # OpenAlex, Crossref, Europe PMC
    PUBMED = auto()  # PubMed/NCBI
    CLINICAL_TRIALS = auto()  # ClinicalTrials.gov
    NEWS = auto()  # GDELT news
    WORLD_BANK = auto()  # World Bank economic data
    OSV = auto()  # Open Source Vulnerabilities


class Evidence(BaseModel):
    """
    Supporting or refuting evidence for a claim.

    Retrieved from knowledge stores and used to justify
    verification decisions.
    """

    id: UUID = Field(default_factory=uuid4)
    source: EvidenceSource
    source_id: str | None = Field(
        default=None, description="ID in source system (e.g., Neo4j node ID)"
    )

    # Content
    content: str = Field(..., description="Evidence text or statement")
    structured_data: dict[str, Any] | None = Field(
        default=None, description="Structured evidence (graph triplet, etc.)"
    )

    # Matching metadata
    similarity_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Semantic similarity if vector-based"
    )
    match_type: str | None = Field(
        default=None, description="Type of match (exact, partial, semantic)"
    )

    # Provenance
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    source_uri: str | None = Field(
        default=None, description="URI/IRI of source document or entity"
    )

    model_config = {"frozen": True}
