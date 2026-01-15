"""
Qdrant hybrid vector store with async parallel uploads.

Features:
- Async uploads with AsyncQdrantClient
- Hybrid search: Dense (sentence-transformers) + Sparse (BM25)
- GPU-accelerated embeddings with multi-process support
- Non-blocking batch uploads for pipeline efficiency
- Full-text payload indexing
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import TYPE_CHECKING

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    KeywordIndexParams,
    KeywordIndexType,
    Modifier,
    OptimizersConfigDiff,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from ingestion.models import ProcessedArticle, ProcessedChunk
from ingestion.preprocessor import BM25Tokenizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger("ingestion.qdrant")

# Constants
DENSE_VECTOR_SIZE = 384
MAX_TEXT_LENGTH = 4000
MAX_POINTS_PER_UPLOAD = 200


class QdrantHybridStore:
    """
    Qdrant store with hybrid search support and async uploads.

    Features:
    - Dense vectors (sentence-transformers) with GPU acceleration
    - Sparse vectors (BM25) for keyword matching
    - Full-text payload index for fallback search
    - Non-blocking upload queue for pipeline efficiency
    - Multi-GPU support for embedding computation

    Architecture:
    - Embedding thread: Computes embeddings from queue
    - Upload thread: Uploads batches to Qdrant
    - Main thread: Enqueues work, non-blocking
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int | None = 6334,
        collection: str = "wikipedia_hybrid",
        embedding_model: str = "all-MiniLM-L12-v2",
        embedding_batch_size: int = 512,
        upload_workers: int = 4,
        embedding_workers: int = 2,
        prefer_grpc: bool = True,
        embedding_device: str = "auto",
    ):
        self.collection = collection
        self.embedding_batch_size = embedding_batch_size

        # Initialize Qdrant client (sync for uploads - more stable)
        # Try gRPC first if preferred, fall back to HTTP if connection fails
        client: QdrantClient | None = None
        
        if prefer_grpc and grpc_port is not None:
            try:
                # Try gRPC connection first
                grpc_client = QdrantClient(
                    host=host,
                    port=port,
                    grpc_port=grpc_port,
                    timeout=30,
                    prefer_grpc=True,
                )
                # Test connection
                grpc_client.get_collections()
                client = grpc_client
                logger.info(f"✅ Connected to Qdrant via gRPC (port {grpc_port})")
            except Exception as e:
                logger.warning(f"⚠️ gRPC connection failed ({e}), falling back to HTTP")
        
        if client is None:
            # Fall back to HTTP
            client = QdrantClient(
                host=host,
                port=port,
                timeout=120,
                prefer_grpc=False,
            )
            logger.info(f"✅ Connected to Qdrant via HTTP (port {port})")
        
        self.client: QdrantClient = client

        # Initialize embedding model with explicit device selection
        cuda_available = torch.cuda.is_available()
        if embedding_device == "cuda" and not cuda_available:
            logger.warning("⚠️ CUDA requested but not available. Falling back to CPU.")
        if embedding_device == "cpu":
            self.device = "cpu"
        elif embedding_device == "cuda" and cuda_available:
            self.device = "cuda"
        else:
            self.device = "cuda" if cuda_available else "cpu"

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.info("💻 Using CPU for embeddings")

        self.model = SentenceTransformer(embedding_model, device=self.device)

        # BM25 tokenizer for sparse vectors
        self.tokenizer = BM25Tokenizer(vocab_size=50000)

        # Shutdown flag - MUST be set before starting any threads
        self._shutdown = False

        # Embedding queue and workers (GPU-accelerated)
        self._embedding_workers = max(1, embedding_workers)
        self._embed_queue: Queue[
            tuple[list[ProcessedArticle] | None, threading.Event]
        ] = Queue(maxsize=max(4, self._embedding_workers * 2))
        self._embed_threads = []
        for i in range(self._embedding_workers):
            t = threading.Thread(
                target=self._embed_worker,
                daemon=True,
                name=f"qdrant_embedder_{i}",
            )
            t.start()
            self._embed_threads.append(t)

        # Upload queue and workers
        self._upload_queue: Queue[tuple[list[PointStruct], threading.Event]] = Queue(
            maxsize=upload_workers * 2
        )
        self._upload_workers = upload_workers
        self._upload_executor = ThreadPoolExecutor(
            max_workers=upload_workers, thread_name_prefix="qdrant_upload"
        )

        # Start upload worker threads
        self._upload_threads = []
        for i in range(upload_workers):
            t = threading.Thread(target=self._upload_worker, daemon=True, name=f"qdrant_uploader_{i}")
            t.start()
            self._upload_threads.append(t)

        # Initialize collection
        self._init_collection()

    def _init_collection(self):
        """Initialize collection with hybrid vector configuration."""
        if self.client.collection_exists(self.collection):
            logger.info(f"Collection '{self.collection}' exists, using it.")
            self._create_payload_indexes()
            return

        logger.info(f"Creating Qdrant collection: {self.collection}")

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    modifier=Modifier.IDF,
                ),
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=128,
                full_scan_threshold=10000,
                on_disk=True,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
        )

        # Create payload indexes for filtering and full-text search
        self._create_payload_indexes()

    def _create_payload_indexes(self):
        """Create indexes on payload fields for efficient filtering."""
        try:
            # Full-text index on text content
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="text",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                    min_token_len=2,
                    max_token_len=20,
                ),
            )

            # Keyword index on title
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="title",
                field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
            )

            # Keyword index on section
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="section",
                field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
            )

            # Keyword indexes for structured metadata
            for field_name in [
                "infobox_type",
                "instance_of",
                "country",
                "occupation",
                "nationality",
                # NEW: Additional filterable fields
                "industry",
                "location",
                "headquarters",
                "spouse",
                "founded_by",
                "capital_of",
                "part_of",
                "predecessor",
                "successor",
            ]:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
                )

            logger.info("✅ Payload indexes created")
        except Exception as e:
            logger.warning(f"Payload index creation: {e}")

    def _embed_worker(self):
        """Worker thread that computes embeddings and enqueues uploads."""
        done_event: threading.Event | None = None
        while not self._shutdown:
            try:
                articles, done_event = self._embed_queue.get(timeout=1.0)
                if articles is None:  # Shutdown signal
                    if done_event is not None:
                        done_event.set()
                    break

                chunk_items = [(a, c) for a in articles for c in a.chunks]
                chunks = [c for _, c in chunk_items]
                if not chunks:
                    done_event.set()
                    done_event = None
                    continue

                dense_vectors, sparse_vectors = self.compute_embeddings(chunks)
                points = self.prepare_points(chunk_items, dense_vectors, sparse_vectors)

                self._upload_queue.put((points, done_event))
                done_event = None  # Ownership transferred to upload queue
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Embedding worker error: {e}")
                if done_event is not None:
                    done_event.set()
                    done_event = None

    def _upload_worker(self):
        """Worker thread that processes upload queue."""
        while not self._shutdown:
            try:
                points, done_event = self._upload_queue.get(timeout=1.0)
                if points is None:  # Shutdown signal
                    break
                self._do_upload(points)
                done_event.set()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Upload worker error: {e}")

    def _do_upload(self, points: list[PointStruct]):
        """Actually upload points to Qdrant."""
        for i in range(0, len(points), MAX_POINTS_PER_UPLOAD):
            chunk = points[i : i + MAX_POINTS_PER_UPLOAD]
            try:
                self.client.upsert(
                    collection_name=self.collection,
                    points=chunk,
                    wait=False,  # Don't wait for indexing
                )
            except Exception as e:
                logger.error(f"Qdrant upsert error: {e}")

    def compute_embeddings(
        self, chunks: list[ProcessedChunk]
    ) -> tuple[np.ndarray, list[tuple[list[int], list[float]]]]:
        """
        Compute dense and sparse embeddings for chunks.

        Returns:
            (dense_embeddings, sparse_vectors) where sparse_vectors is
            list of (indices, values) tuples
        """
        # Prepare texts (with truncation)
        texts = []
        for chunk in chunks:
            ctx_text = chunk.contextualized_text
            if len(ctx_text) > MAX_TEXT_LENGTH + 200:
                ctx_text = ctx_text[: MAX_TEXT_LENGTH + 200] + "..."
            texts.append(ctx_text)

        # Update IDF scores
        self.tokenizer.update_idf(texts)

        # Compute dense embeddings with GPU
        dense_vectors = self.model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Compute sparse vectors (CPU, fast)
        sparse_vectors = [self.tokenizer.to_sparse_vector(t) for t in texts]

        return dense_vectors, sparse_vectors

    def prepare_points(
        self,
        chunk_items: list[tuple[ProcessedArticle, ProcessedChunk]],
        dense_vectors: np.ndarray,
        sparse_vectors: list[tuple[list[int], list[float]]],
    ) -> list[PointStruct]:
        """Build PointStruct objects for upload."""
        points = []

        for i, (article, chunk) in enumerate(chunk_items):
            # Generate deterministic point ID
            point_id = abs(hash(chunk.chunk_id)) % (2**63)

            indices, values = sparse_vectors[i]

            # Truncate text for payload
            text = chunk.text
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH] + "..."

            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vectors[i].tolist(),
                    "sparse": SparseVector(indices=indices, values=values),
                },
                payload={
                    "page_id": chunk.page_id,
                    "article_id": chunk.page_id,
                    "title": chunk.title,
                    "text": text,
                    "section": chunk.section,
                    "url": chunk.url,
                    "chunk_id": chunk.chunk_id,
                    "word_count": chunk.word_count,
                    "is_first": chunk.is_first_chunk,
                    "source": "wikipedia",
                    "infobox_type": article.article.infobox.type if article.article.infobox else None,
                    "instance_of": article.article.instance_of,
                    # Person-related metadata
                    "birth_date": article.article.birth_date,
                    "death_date": article.article.death_date,
                    "location": article.article.location,
                    "occupation": article.article.occupation,
                    "nationality": article.article.nationality,
                    "spouse": article.article.spouse,
                    "children": list(article.article.children)[:10],
                    "parents": list(article.article.parents)[:10],
                    "education": list(article.article.education)[:10],
                    "employer": list(article.article.employer)[:10],
                    "awards": list(article.article.awards)[:10],
                    # Creative/Influence metadata
                    "author_of": list(article.article.author_of)[:10],
                    "genre": list(article.article.genre)[:10],
                    "influenced_by": list(article.article.influenced_by)[:10],
                    "influenced": list(article.article.influenced)[:10],
                    # Organization metadata
                    "founded_by": article.article.founded_by,
                    "founding_date": article.article.founding_date,
                    "headquarters": article.article.headquarters,
                    "industry": article.article.industry,
                    # Geographic metadata
                    "country": article.article.country,
                    "capital_of": article.article.capital_of,
                    "part_of": article.article.part_of,
                    # Temporal metadata
                    "predecessor": article.article.predecessor,
                    "successor": article.article.successor,
                    # Categories and entities
                    "categories": list(article.article.categories)[:20],
                    "entities": list(article.article.entities)[:20],
                },
            )
            points.append(point)

        return points

    def upload_batch_async(
        self, articles: list[ProcessedArticle]
    ) -> threading.Event:
        """
        Upload a batch of articles asynchronously.

        Enqueues articles for embedding computation by worker threads,
        then automatic upload. Returns an Event that will be set when
        the full pipeline (embed -> upload) completes.

        This is fully non-blocking - embedding happens in separate threads.
        """
        # Check for empty batch
        all_chunks = [c for a in articles for c in a.chunks]
        if not all_chunks:
            event = threading.Event()
            event.set()
            return event

        # Queue for async embedding -> upload pipeline
        done_event = threading.Event()
        self._embed_queue.put((articles, done_event))

        return done_event

    def upload_batch_sync(self, articles: list[ProcessedArticle]) -> int:
        """
        Upload a batch of articles synchronously.

        Returns the number of chunks uploaded.
        """
        chunk_items = [(a, c) for a in articles for c in a.chunks]
        chunks = [c for _, c in chunk_items]
        if not chunks:
            return 0

        # Compute embeddings
        dense_vectors, sparse_vectors = self.compute_embeddings(chunks)

        # Build and upload points
        points = self.prepare_points(chunk_items, dense_vectors, sparse_vectors)
        self._do_upload(points)

        return len(points)

    def flush(self):
        """Wait for all pending embeddings and uploads to complete."""
        # Drain embedding queue first (process synchronously)
        while not self._embed_queue.empty():
            try:
                articles, event = self._embed_queue.get_nowait()
                if articles is None:
                    if event is not None:
                        event.set()
                    continue
                chunk_items = [(a, c) for a in articles for c in a.chunks]
                chunks = [c for _, c in chunk_items]
                if not chunks:
                    event.set()
                    continue
                dense_vectors, sparse_vectors = self.compute_embeddings(chunks)
                points = self.prepare_points(chunk_items, dense_vectors, sparse_vectors)
                self._do_upload(points)
                event.set()
            except Empty:
                break

        # Drain upload queue
        while not self._upload_queue.empty():
            try:
                points, event = self._upload_queue.get_nowait()
                if points is None:
                    if event is not None:
                        event.set()
                    continue
                self._do_upload(points)
                event.set()
            except Empty:
                break

    def close(self):
        """Clean up resources."""
        self._shutdown = True

        # Signal embedding workers to stop
        for _ in self._embed_threads:
            try:
                self._embed_queue.put((None, threading.Event()), timeout=1.0)
            except Exception:
                pass

        # Signal upload workers to stop
        for _ in self._upload_threads:
            try:
                self._upload_queue.put((None, threading.Event()), timeout=1.0)  # type: ignore[arg-type]
            except Exception:
                pass

        # Wait for embedding workers
        for t in self._embed_threads:
            t.join(timeout=5.0)

        # Wait for upload workers
        for t in self._upload_threads:
            t.join(timeout=5.0)

        self._upload_executor.shutdown(wait=False)

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection)
            return {
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
                "status": info.status.name,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
