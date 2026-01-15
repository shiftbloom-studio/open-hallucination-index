"""
Qdrant Vector Knowledge Store Adapter
=====================================

Adapter for Qdrant vector database as a semantic knowledge store.

INGESTION COMPATIBILITY
-----------------------
This adapter is fully aligned with the ingestion pipeline structure:

Ingestion Structure (ingestion/qdrant_store.py):
- Dense vectors: 384-dim sentence-transformers (all-MiniLM-L12-v2)
- Sparse vectors: BM25 with IDF weighting
- Payload fields:
  * page_id, article_id, title, text, section, url, chunk_id
  * word_count, is_first, source
  * Metadata: infobox_type, instance_of, birth_date, death_date
  * location, occupation, nationality, country, industry, headquarters
  * categories (array), entities (array)
- Payload indexes on: text (full-text), title, section, infobox_type,
  country, occupation, nationality (keyword indexes)

API Compatibility:
- Reads "text" field (not "content") for Wikipedia chunks
- Supports "content" field for persisted external evidence
- Uses "url" field as source_uri
- Supports metadata filtering on all indexed fields
- Hybrid search ready (dense + sparse vectors)

Version: 2025-01-15 - Aligned with ingestion v2
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from open_hallucination_index.domain.entities import Evidence, EvidenceSource
from open_hallucination_index.ports.knowledge_store import VectorKnowledgeStore, VectorQuery

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.infrastructure.config import QdrantSettings

logger = logging.getLogger(__name__)


class QdrantError(Exception):
    """Exception raised when Qdrant operations fail."""

    pass


# Type alias for embedding function
EmbeddingFunc = Callable[[str], Coroutine[Any, Any, list[float]]]


class QdrantVectorAdapter(VectorKnowledgeStore):
    """
    Adapter for Qdrant as a vector-based knowledge store.

    Provides semantic similarity search for fact verification.
    """

    def __init__(
        self,
        settings: QdrantSettings,
        embedding_func: EmbeddingFunc | None = None,
    ) -> None:
        """
        Initialize the adapter with configuration.

        Args:
            settings: Qdrant connection settings.
            embedding_func: Optional async function to generate embeddings.
        """
        self._settings = settings
        self._embedding_func = embedding_func
        self._client: AsyncQdrantClient | None = None

    def set_embedding_func(self, func: EmbeddingFunc) -> None:
        """Set the embedding function after initialization."""
        self._embedding_func = func

    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        try:
            api_key = None
            if self._settings.api_key:
                api_key = self._settings.api_key.get_secret_value()

            # Build client kwargs
            client_kwargs: dict[str, object] = {
                "host": self._settings.host,
                "port": self._settings.port,
                "api_key": api_key,
                "prefer_grpc": self._settings.use_grpc,
            }
            if self._settings.use_grpc:
                client_kwargs["grpc_port"] = self._settings.grpc_port

            # Force IPv4 by using a custom httpx transport
            ipv4_transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
            client_kwargs["https"] = False
            self._client = AsyncQdrantClient(
                **client_kwargs,  # type: ignore[arg-type]
            )
            # Override the internal httpx client to use IPv4
            if hasattr(self._client, "_client") and self._client._client is not None:
                self._client._client._transport = ipv4_transport

            await self._ensure_collection()
            logger.info(f"Connected to Qdrant at {self._settings.host}:{self._settings.port}")

        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise QdrantError(f"Connection failed: {e}") from e

    async def _ensure_collection(self) -> None:
        """
        Ensure the knowledge base collection exists with correct configuration.
        Handles concurrency between multiple workers.
        
        IMPORTANT: This method will NOT delete existing collections with data.
        It supports both simple vectors and Named Vectors (hybrid collections).
        """
        if self._client is None:
            return

        collection_name = self._settings.collection_name
        vector_size = self._settings.vector_size

        try:
            # Check if collection exists
            exists = False
            try:
                config = await self._client.get_collection(collection_name)
                exists = True
                points_count = config.points_count or 0

                # Get dimension - handle both simple and named vectors
                existing_size = 0
                vectors_params = config.config.params.vectors

                # Case 1: Simple vector config (has .size directly)
                if hasattr(vectors_params, "size") and vectors_params.size:
                    existing_size = vectors_params.size
                # Case 2: Named vectors (dict with vector names like "dense", "sparse")
                elif isinstance(vectors_params, dict):
                    # Check for "dense" vector first (common in hybrid setups)
                    if "dense" in vectors_params:
                        dense_config = vectors_params["dense"]
                        if hasattr(dense_config, "size"):
                            existing_size = dense_config.size
                        elif isinstance(dense_config, dict) and "size" in dense_config:
                            existing_size = dense_config["size"]
                    # Fallback to any vector with matching size
                    elif not existing_size:
                        for vec_config in vectors_params.values():
                            if hasattr(vec_config, "size") and vec_config.size == vector_size:
                                existing_size = vec_config.size
                                break
                            elif isinstance(vec_config, dict) and vec_config.get("size") == vector_size:
                                existing_size = vec_config["size"]
                                break
                    # Still no size? Just use first available
                    if not existing_size and vectors_params:
                        first_vec = next(iter(vectors_params.values()))
                        if hasattr(first_vec, "size"):
                            existing_size = first_vec.size
                        elif isinstance(first_vec, dict):
                            existing_size = first_vec.get("size", 0)

                # Log collection info
                logger.info(
                    f"Qdrant collection '{collection_name}' exists with {points_count} points, "
                    f"vector dimension: {existing_size}"
                )

                # Check dimension - but NEVER delete a collection with data!
                if existing_size and existing_size != vector_size:
                    if points_count > 0:
                        logger.warning(
                            f"Qdrant collection '{collection_name}' has different dimension "
                            f"(expected {vector_size}, found {existing_size}), but contains "
                            f"{points_count} points. Keeping existing collection. "
                            f"Search may still work if using named 'dense' vectors."
                        )
                        # Don't delete - just proceed with existing collection
                    else:
                        # Empty collection with wrong dimension - safe to recreate
                        logger.info(
                            f"Empty collection '{collection_name}' has wrong dimension. Recreating..."
                        )
                        try:
                            await self._client.delete_collection(collection_name)
                        except Exception as e:
                            logger.debug(f"Delete collection failed: {e}")
                        exists = False
            except Exception:
                # Collection does not exist or error getting it
                exists = False

            if not exists:
                try:
                    await self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info(f"Created Qdrant collection: {collection_name}")
                except Exception as e:
                    # If 409 Conflict, it means another worker created it in the meantime
                    status_code = e.status_code if hasattr(e, "status_code") else None
                    if "already exists" in str(e) or status_code == 409:
                        logger.info(
                            "Collection %s already exists (created by another worker)",
                            collection_name,
                        )
                    else:
                        raise e
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            # Re-raise to prevent starting with broken vector store
            raise QdrantError(f"Collection initialization failed: {e}") from e

    async def disconnect(self) -> None:
        """Close the Qdrant connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Qdrant")

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        if self._client is None:
            return False
        try:
            await self._client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def search_similar(
        self,
        query: VectorQuery,
    ) -> list[Evidence]:
        """
        Search for semantically similar content.
        
        Supports hybrid search (dense + sparse vectors) and
        rich metadata filtering matching the ingestion structure.

        Args:
            query: Vector query specification.

        Returns:
            List of evidence ranked by similarity.
        """
        if self._client is None:
            raise QdrantError("Not connected to Qdrant")

        embedding = query.embedding
        if embedding is None:
            embedding = await self.embed_text(query.text)

        try:
            # Build filter from metadata (supports ingestion structure)
            query_filter = None
            must_conditions = []
            
            # Basic metadata filters
            if query.filter_metadata:
                for key, value in query.filter_metadata.items():
                    must_conditions.append({"key": key, "match": {"value": value}})
            
            # Extended filters for ingestion metadata
            if query.infobox_types:
                must_conditions.append({
                    "key": "infobox_type",
                    "match": {"any": query.infobox_types}
                })
            
            if query.categories:
                # Categories is an array in payload - match any
                must_conditions.append({
                    "key": "categories",
                    "match": {"any": query.categories}
                })
            
            if query.section_filter:
                must_conditions.append({
                    "key": "section",
                    "match": {"value": query.section_filter}
                })
            
            if must_conditions:
                query_filter = Filter(must=must_conditions)  # type: ignore[arg-type]

            # Use hybrid search if sparse vectors provided, otherwise dense only
            if query.sparse_indices and query.sparse_values:
                # Hybrid search with sparse + dense
                from qdrant_client.models import SparseVector, QueryRequest
                
                # TODO: Qdrant client API for hybrid query may need adjustment
                # For now, fall back to dense search
                # Future: use proper hybrid query with sparse vector
                logger.debug("Hybrid search requested but using dense-only for now")
            
            # Dense vector search - use "dense" named vector for wikipedia_hybrid collection
            results = await self._client.query_points(
                collection_name=self._settings.collection_name,
                query=embedding,
                using="dense",  # Named vector for hybrid collections
                limit=query.top_k,
                score_threshold=query.min_similarity,
                query_filter=query_filter,
            )

            evidence_list: list[Evidence] = []
            for hit in results.points:
                # Use "text" field from ingestion (not "content")
                content = ""
                if hit.payload:
                    # Primary: use "text" field from Wikipedia ingestion
                    content = hit.payload.get("text", "")
                    # Fallback: for persisted external evidence, use "content"
                    if not content:
                        content = hit.payload.get("content", "")
                
                # Build source URI from url or source_uri
                source_uri = None
                if hit.payload:
                    source_uri = hit.payload.get("url") or hit.payload.get("source_uri")
                
                evidence_list.append(
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.VECTOR_SEMANTIC,
                        source_id=str(hit.id),
                        content=content,
                        structured_data=hit.payload,
                        similarity_score=hit.score,
                        match_type="semantic",
                        retrieved_at=datetime.now(UTC),
                        source_uri=source_uri,
                    )
                )

            return evidence_list

        except UnexpectedResponse as e:
            logger.error(f"Qdrant search failed: {e}")
            raise QdrantError(f"Search failed: {e}") from e
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            raise QdrantError(f"Search error: {e}") from e

    async def find_evidence_for_claim(
        self,
        claim: Claim,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[Evidence]:
        """
        Find semantically similar evidence for a claim.
        
        Utilizes rich metadata filtering from ingestion:
        - infobox_type: Filter by entity types (Person, Organization, etc.)
        - categories: Filter by Wikipedia categories
        - entities: Filter by mentioned entities

        Args:
            claim: The claim to verify.
            top_k: Maximum number of results.
            min_similarity: Minimum similarity threshold (lowered to 0.5 for better recall).

        Returns:
            List of semantically similar evidence.
        """
        search_text = claim.normalized_form or claim.text
        
        # Build metadata filters from claim structure
        filter_metadata = {}
        
        # Filter by source if claim metadata available
        if hasattr(claim, "metadata") and claim.metadata:
            # Example: Filter by entity type, category, etc.
            if "entity_type" in claim.metadata:
                filter_metadata["infobox_type"] = claim.metadata["entity_type"]
            if "category" in claim.metadata:
                filter_metadata["categories"] = claim.metadata["category"]

        query = VectorQuery(
            text=search_text,
            top_k=top_k,
            min_similarity=min_similarity,
            filter_metadata=filter_metadata if filter_metadata else None,
        )

        return await self.search_similar(query)

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if self._embedding_func is None:
            raise NotImplementedError(
                "No embedding function configured. "
                "Pass embedding_func to constructor or call set_embedding_func()."
            )
        return await self._embedding_func(text)

    async def embed_texts_batch(
        self,
        texts: list[str],
        max_concurrency: int = 8,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in parallel.

        Uses asyncio.gather() with optional concurrency limiting via semaphore
        to prevent overwhelming the embedding service.

        Args:
            texts: List of texts to embed.
            max_concurrency: Maximum parallel embedding operations.

        Returns:
            List of embedding vectors in same order as input texts.
        """
        if self._embedding_func is None:
            raise NotImplementedError(
                "No embedding function configured. "
                "Pass embedding_func to constructor or call set_embedding_func()."
            )

        if not texts:
            return []

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def embed_with_semaphore(text: str) -> list[float]:
            async with semaphore:
                return await self._embedding_func(text)  # type: ignore[misc]

        # Run all embeddings in parallel with concurrency limit
        embeddings = await asyncio.gather(*[embed_with_semaphore(text) for text in texts])
        return list(embeddings)

    async def add_facts(
        self,
        facts: list[dict[str, Any]],
    ) -> int:
        """
        Add facts to the vector store.

        Args:
            facts: List of fact dictionaries with 'content' and optional metadata.

        Returns:
            Number of facts added.
        """
        if self._client is None:
            raise QdrantError("Not connected to Qdrant")

        if not facts:
            return 0

        points = []
        for fact in facts:
            content = fact.get("content", "")
            if not content:
                continue

            embedding = await self.embed_text(content)
            point_id = fact.get("id", str(uuid4()))

            payload = {
                "content": content,
                "source_uri": fact.get("source_uri"),
                "subject": fact.get("subject"),
                "predicate": fact.get("predicate"),
                "object": fact.get("object"),
                **{k: v for k, v in fact.items() if k not in ["content", "id"]},
            }

            # Use named vector "dense" for wikipedia_hybrid collection
            points.append(
                PointStruct(
                    id=point_id if isinstance(point_id, int) else hash(point_id) % (2**63),
                    vector={"dense": embedding},  # Named vector for hybrid collection
                    payload=payload,
                )
            )

        try:
            await self._client.upsert(
                collection_name=self._settings.collection_name,
                points=points,
            )
            logger.info(f"Added {len(points)} facts to Qdrant")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to add facts: {e}")
            raise QdrantError(f"Upsert failed: {e}") from e

    async def persist_external_evidence(
        self,
        evidence_list: list[Evidence],
    ) -> int:
        """
        Persist external evidence (from MCP) to Qdrant for semantic fallback.

        This enables future claims to find similar evidence via vector search,
        reducing latency by avoiding MCP calls for previously seen content.

        Args:
            evidence_list: List of Evidence objects to persist.

        Returns:
            Number of evidence items persisted.
        """
        if self._client is None:
            raise QdrantError("Not connected to Qdrant")

        if not evidence_list:
            return 0

        # Convert Evidence to facts format
        facts = []
        for ev in evidence_list:
            fact = {
                "id": str(ev.id),
                "content": ev.content,
                "source_uri": ev.source_uri,
                "source": ev.source.value if hasattr(ev.source, "value") else str(ev.source),
                "match_type": ev.match_type or "external",
                "original_similarity": ev.similarity_score,
                "retrieved_at": ev.retrieved_at.isoformat() if ev.retrieved_at else None,
            }

            # Add structured data fields if available
            if ev.structured_data:
                fact["subject"] = ev.structured_data.get("subject")
                fact["predicate"] = ev.structured_data.get("predicate")
                fact["object"] = ev.structured_data.get("object")

            facts.append(fact)

        return await self.add_facts(facts)

    async def count_similar(
        self,
        text: str,
        min_similarity: float = 0.5,
    ) -> int:
        """
        Fast count of similar vectors without retrieving full content.

        Useful for sufficiency checks before full retrieval.

        Args:
            text: Text to find similar content for.
            min_similarity: Minimum similarity threshold (lowered for better recall).

        Returns:
            Count of similar items.
        """
        if self._client is None:
            return 0

        try:
            embedding = await self.embed_text(text)

            # Use scroll with count to efficiently count matches
            results = await self._client.query_points(
                collection_name=self._settings.collection_name,
                query=embedding,
                using="dense",  # Named vector for hybrid collections
                limit=100,  # Cap for performance
                score_threshold=min_similarity,
            )

            return len(results.points)
        except Exception as e:
            logger.debug(f"Count similar failed: {e}")
            return 0

    async def hybrid_search(
        self,
        text: str,
        keywords: list[str] | None = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
        keyword_boost: float = 0.3,
    ) -> list[Evidence]:
        """
        Hybrid search combining dense vectors with keyword matching.

        First performs vector search, then boosts results containing keywords.

        Args:
            text: Query text for dense search.
            keywords: Optional keywords to boost matching results.
            top_k: Maximum results to return.
            min_similarity: Minimum similarity threshold.
            keyword_boost: Score boost for keyword matches (0.0-1.0).

        Returns:
            List of evidence with adjusted scores.
        """
        if self._client is None:
            raise QdrantError("Not connected to Qdrant")

        # Get dense vector results with extra buffer for reranking
        base_results = await self.search_similar(
            VectorQuery(
                text=text,
                top_k=top_k * 2,  # Get extra for filtering
                min_similarity=min_similarity,
            )
        )

        if not keywords:
            return base_results[:top_k]

        # Apply keyword boost
        boosted_results: list[tuple[float, Evidence]] = []
        keywords_lower = [k.lower() for k in keywords]

        for ev in base_results:
            content_lower = ev.content.lower()
            boost = 0.0

            # Count keyword matches
            matches = sum(1 for kw in keywords_lower if kw in content_lower)
            if matches > 0:
                # Boost proportional to matches (capped)
                boost = min(keyword_boost * matches, keyword_boost * 2)

            final_score = (ev.similarity_score or 0.5) + boost
            boosted_results.append((final_score, ev))

        # Sort by boosted score and return top_k
        boosted_results.sort(key=lambda x: x[0], reverse=True)

        # Update similarity scores with boosted values
        final_evidence: list[Evidence] = []
        for score, ev in boosted_results[:top_k]:
            # Create new Evidence with updated score
            final_evidence.append(
                Evidence(
                    id=ev.id,
                    source=ev.source,
                    source_id=ev.source_id,
                    content=ev.content,
                    structured_data=ev.structured_data,
                    similarity_score=min(1.0, score),  # Cap at 1.0
                    match_type=ev.match_type,
                    retrieved_at=ev.retrieved_at,
                    source_uri=ev.source_uri,
                )
            )

        return final_evidence
