"""
KnowledgeStore Port
===================

Abstract interfaces for knowledge retrieval backends.
Supports both graph-based (exact) and vector-based (semantic) stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim, Evidence


@dataclass(frozen=True, slots=True)
class GraphQuery:
    """Query specification for graph-based lookup."""

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    max_hops: int = 2  # For graph traversal
    limit: int = 10


@dataclass(frozen=True, slots=True)
class VectorQuery:
    """
    Query specification for vector-based semantic search.
    
    Supports hybrid search (dense + sparse vectors) matching 
    the ingestion structure with BM25 sparse vectors.
    """

    text: str
    embedding: list[float] | None = None  # Pre-computed dense embedding
    sparse_indices: list[int] | None = None  # Sparse vector indices (BM25)
    sparse_values: list[float] | None = None  # Sparse vector values (BM25)
    top_k: int = 5
    min_similarity: float = 0.5  # Lowered for better evidence recall
    filter_metadata: dict[str, Any] | None = None
    # Extended filters matching Wikipedia ingestion metadata
    infobox_types: list[str] | None = None  # Filter by entity types
    categories: list[str] | None = None      # Filter by Wikipedia categories
    section_filter: str | None = None        # Filter by section name


class KnowledgeStore(ABC):
    """
    Base port for knowledge retrieval.

    Defines common lifecycle and health check operations.
    Specific query methods are defined in subclasses.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the knowledge store."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the knowledge store."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the store is operational."""
        ...


class GraphKnowledgeStore(KnowledgeStore):
    """
    Port for graph-based knowledge retrieval.

    Implementations query knowledge graphs (Neo4j, etc.)
    for exact or inferred fact matches.

    Responsibilities:
    - Exact triplet matching
    - Graph traversal for inference
    - Entity resolution and linking
    """

    @abstractmethod
    async def query_triplet(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[Evidence]:
        """
        Query for matching triplets in the graph.

        At least one of subject, predicate, or obj must be provided.

        Args:
            subject: Subject entity to match.
            predicate: Relationship/property to match.
            obj: Object entity or value to match.

        Returns:
            List of evidence from matching graph entries.
        """
        ...

    @abstractmethod
    async def find_evidence_for_claim(
        self,
        claim: Claim,
        max_hops: int = 2,
    ) -> list[Evidence]:
        """
        Find graph evidence supporting or refuting a claim.

        May perform multi-hop traversal to find indirect evidence.

        Args:
            claim: The claim to verify.
            max_hops: Maximum graph traversal depth.

        Returns:
            List of relevant evidence from the graph.
        """
        ...

    @abstractmethod
    async def entity_exists(self, entity: str) -> bool:
        """Check if an entity exists in the knowledge graph."""
        ...

    @abstractmethod
    async def get_entity_properties(
        self,
        entity: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve all properties of an entity.

        Args:
            entity: Entity identifier.

        Returns:
            Dictionary of properties, or None if entity not found.
        """
        ...


class VectorKnowledgeStore(KnowledgeStore):
    """
    Port for vector-based semantic knowledge retrieval.

    Implementations query vector databases (Qdrant, etc.)
    for semantically similar facts.

    Responsibilities:
    - Semantic similarity search
    - Embedding-based retrieval
    - Metadata filtering
    """

    @abstractmethod
    async def search_similar(
        self,
        query: VectorQuery,
    ) -> list[Evidence]:
        """
        Search for semantically similar content.

        Args:
            query: Vector query specification.

        Returns:
            List of evidence ranked by similarity.
        """
        ...

    @abstractmethod
    async def find_evidence_for_claim(
        self,
        claim: Claim,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[Evidence]:
        """
        Find vector evidence supporting or refuting a claim.

        Args:
            claim: The claim to verify.
            top_k: Maximum number of results.
            min_similarity: Minimum similarity threshold (lowered to 0.5 for better recall).

        Returns:
            List of semantically similar evidence.
        """
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        May delegate to an embedding model or use store's built-in.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        ...
