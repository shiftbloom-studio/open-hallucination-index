"""
Qdrant Vector Knowledge Store Adapter
=====================================

Adapter for Qdrant vector database as a semantic knowledge store.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine
from uuid import uuid4

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

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
        """Ensure the knowledge base collection exists."""
        if self._client is None:
            return

        try:
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self._settings.collection_name in collection_names:
                # Check existing collection config
                config = await self._client.get_collection(self._settings.collection_name)
                # Pydantic models for Qdrant API
                existing_size = 0
                if hasattr(config.config.params, "vectors") and hasattr(config.config.params.vectors, "size"):
                    existing_size = config.config.params.vectors.size
                elif isinstance(config.config.params.vectors, dict) and "size" in config.config.params.vectors:
                    existing_size = config.config.params.vectors["size"]

                if existing_size != self._settings.vector_size:
                    logger.warning(
                        f"Qdrant collection '{self._settings.collection_name}' has wrong dimension: "
                        f"expected {self._settings.vector_size}, found {existing_size}. Recreating..."
                    )
                    await self._client.delete_collection(self._settings.collection_name)
                    collection_names.remove(self._settings.collection_name)

            if self._settings.collection_name not in collection_names:
                await self._client.create_collection(
                    collection_name=self._settings.collection_name,
                    vectors_config=VectorParams(
                        size=self._settings.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self._settings.collection_name}")
        except Exception as e:
            logger.warning(f"Collection check/creation failed: {e}")

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
            query_filter = None
            if query.filter_metadata:
                must_conditions = []
                for key, value in query.filter_metadata.items():
                    must_conditions.append({"key": key, "match": {"value": value}})
                if must_conditions:
                    query_filter = Filter(must=must_conditions)  # type: ignore[arg-type]

            results = await self._client.query_points(
                collection_name=self._settings.collection_name,
                query=embedding,
                limit=query.top_k,
                score_threshold=query.min_similarity,
                query_filter=query_filter,
            )

            evidence_list: list[Evidence] = []
            for hit in results.points:
                evidence_list.append(
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.VECTOR_SEMANTIC,
                        source_id=str(hit.id),
                        content=hit.payload.get("content", "") if hit.payload else "",
                        structured_data=hit.payload,
                        similarity_score=hit.score,
                        match_type="semantic",
                        retrieved_at=datetime.utcnow(),
                        source_uri=hit.payload.get("source_uri") if hit.payload else None,
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
        min_similarity: float = 0.7,
    ) -> list[Evidence]:
        """
        Find semantically similar evidence for a claim.

        Args:
            claim: The claim to verify.
            top_k: Maximum number of results.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of semantically similar evidence.
        """
        search_text = claim.normalized_form or claim.text

        query = VectorQuery(
            text=search_text,
            top_k=top_k,
            min_similarity=min_similarity,
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
        embeddings = await asyncio.gather(
            *[embed_with_semaphore(text) for text in texts]
        )
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

            points.append(
                PointStruct(
                    id=point_id if isinstance(point_id, int) else hash(point_id) % (2**63),
                    vector=embedding,
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
