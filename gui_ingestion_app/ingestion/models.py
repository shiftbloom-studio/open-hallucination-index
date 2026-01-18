"""
Data models for the Wikipedia ingestion pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================


class IngestionMode(Enum):
    """Ingestion mode selection."""

    CHUNKED = "chunked"  # Download files then process (default, robust)
    STREAMING = "streaming"  # Stream directly from server (less robust)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""

    # Connection settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_https: bool = False
    qdrant_tls_ca_cert: str | None = None
    neo4j_uri: str = "bolt://localhost:7687"

    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # Collection/database names
    qdrant_collection: str = "wikipedia_hybrid"

    # Processing settings
    limit: int | None = None
    batch_size: int = 256  # Balanced throughput/stability for 64GB RAM
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 100

    # Parallelism settings - aggressive defaults for maximum throughput
    download_workers: int = 4  # Parallel downloads
    dump_workers: int = 3  # Parallel dump file processing workers
    preprocess_workers: int = 8  # Text preprocessing threads
    embedding_workers: int = 1  # GPU embedding workers
    upload_workers: int = 2  # Parallel DB uploads (reduced to avoid Neo4j deadlocks)
    embedding_batch_size: int = 768  # Tuned for RTX 4090

    # Queue sizes for producer-consumer pattern
    download_queue_size: int = 8  # Downloaded files awaiting processing
    preprocess_queue_size: int = 1024  # Articles awaiting embedding
    upload_queue_size: int = 12  # Batches awaiting upload
    max_pending_batches: int = 4  # Backpressure to prevent batch pile-up

    # Download settings
    download_dir: str = ".wiki_dumps"
    keep_downloads: bool = False
    max_concurrent_downloads: int = 4

    # Checkpoint settings
    checkpoint_file: str | None = None
    auto_save_interval: int = 100  # Save checkpoint every N batches

    # Retry settings
    endless_mode: bool = False
    endless_retry_delay: int = 30

    # BM25 settings
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_vocab_size: int = 50000

    # Embedding model
    embedding_model: str = "all-MiniLM-L12-v2"
    embedding_dim: int = 384
    # Embedding device selection: "auto" | "cuda" | "cpu"
    embedding_device: str = "auto"


# =============================================================================
# ARTICLE DATA MODELS
# =============================================================================


@dataclass
class WikiSection:
    """Represents a section within an article."""

    title: str
    level: int
    content: str
    start_pos: int
    end_pos: int


@dataclass
class WikiInfobox:
    """Parsed infobox data."""

    type: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class WikiArticle:
    """Rich article representation with extracted metadata."""

    id: int
    title: str
    text: str
    url: str
    links: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    sections: list[WikiSection] = field(default_factory=list)
    infobox: WikiInfobox | None = None
    first_paragraph: str = ""
    word_count: int = 0
    entities: set[str] = field(default_factory=set)
    # Additional metadata for rich graph relationships
    disambiguation_links: set[str] = field(default_factory=set)
    see_also_links: set[str] = field(default_factory=set)
    external_links: set[str] = field(default_factory=set)
    redirect_from: str | None = None
    # Extracted facts for knowledge graph (existing)
    birth_date: str | None = None
    death_date: str | None = None
    location: str | None = None
    occupation: str | None = None
    nationality: str | None = None
    # NEW: Additional relationship metadata for 15 more relationships
    # Person-related
    spouse: str | None = None
    children: set[str] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)
    education: set[str] = field(default_factory=set)  # Schools/universities
    employer: set[str] = field(default_factory=set)  # Companies/organizations
    awards: set[str] = field(default_factory=set)
    # Work-related
    author_of: set[str] = field(default_factory=set)  # Books, papers, etc.
    genre: set[str] = field(default_factory=set)  # For creative works/artists
    influenced_by: set[str] = field(default_factory=set)
    influenced: set[str] = field(default_factory=set)
    # Organization/Place-related
    founded_by: str | None = None
    founding_date: str | None = None
    headquarters: str | None = None
    industry: str | None = None
    # Geographic
    country: str | None = None
    capital_of: str | None = None
    part_of: str | None = None  # e.g., city part of state
    # Temporal
    predecessor: str | None = None
    successor: str | None = None
    # Classification
    instance_of: str | None = None  # Type classification (person, city, company, etc.)

    # =========================================================================
    # NEW: Enhanced metadata from SQL dumps (pre-computed by Wikipedia)
    # =========================================================================

    # Wikidata integration - links to structured knowledge base
    wikidata_id: str | None = None  # Q-ID like 'Q42' for Douglas Adams

    # Geographic coordinates (from geo_tags.sql.gz)
    latitude: float | None = None
    longitude: float | None = None
    geo_type: str | None = None  # 'city', 'country', 'landmark', 'mountain', etc.
    geo_country_code: str | None = None  # ISO country code like 'US', 'GB'
    geo_dimension: int | None = None  # Size in meters (useful for zoom level)

    # Page metadata (from page.sql.gz)
    is_redirect: bool = False
    redirect_target: str | None = None  # Title of target page if redirect
    page_length: int = 0  # Article length in bytes (proxy for importance)
    is_disambiguation: bool = False  # Disambiguation page flag

    # Category hierarchy (enhanced from categorylinks.sql.gz)
    parent_categories: set[str] = field(default_factory=set)  # Direct parent categories
    category_depth: int | None = None  # Depth in category tree (lower = more general)

    # Link statistics (from pagelinks.sql.gz)
    incoming_link_count: int = 0  # Number of pages linking TO this page
    outgoing_link_count: int = 0  # Number of pages this page links to

    # Quality indicators
    has_infobox: bool = False
    has_coordinates: bool = False
    has_wikidata: bool = False
    quality_score: float = 0.0  # Computed importance/quality score (0-1)


@dataclass
class ProcessedChunk:
    """A processed chunk ready for embedding."""

    chunk_id: str
    text: str
    contextualized_text: str  # With title/section prefix for better embeddings
    section: str
    start_char: int
    end_char: int
    page_id: int
    title: str
    url: str
    word_count: int
    is_first_chunk: bool = False
    # Pre-computed embeddings (filled during pipeline)
    dense_embedding: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None


@dataclass
class ProcessedArticle:
    """Article with its processed chunks, ready for upload."""

    article: WikiArticle
    chunks: list[ProcessedChunk]
    clean_text: str


# =============================================================================
# STATISTICS
# =============================================================================


@dataclass
class PipelineStats:
    """Real-time pipeline statistics."""

    # Counters
    articles_processed: int = 0
    articles_skipped: int = 0
    chunks_created: int = 0
    links_extracted: int = 0
    categories_extracted: int = 0
    entities_extracted: int = 0
    errors: int = 0

    # Timing
    start_time: float = 0.0
    download_time: float = 0.0
    preprocess_time: float = 0.0
    embedding_time: float = 0.0
    upload_time: float = 0.0

    # Queue depths (for monitoring)
    download_queue_depth: int = 0
    preprocess_queue_depth: int = 0
    upload_queue_depth: int = 0

    # Parts tracking
    parts_downloaded: int = 0
    parts_processed: int = 0
    parts_total: int = 0

    def rate(self) -> float:
        """Calculate articles per second."""
        elapsed = time.time() - self.start_time
        return self.articles_processed / elapsed if elapsed > 0 else 0.0

    def eta_seconds(self, total: int | None) -> float | None:
        """Estimate time remaining."""
        if total is None or self.articles_processed == 0:
            return None
        rate = self.rate()
        if rate == 0:
            return None
        remaining = total - self.articles_processed
        return remaining / rate

    def summary(self) -> str:
        """Generate summary string."""
        elapsed = time.time() - self.start_time
        rate = self.rate()
        return (
            f"\n{'='*70}\n"
            f"📊 INGESTION SUMMARY\n"
            f"{'='*70}\n"
            f"⏱️  Duration: {elapsed:.1f}s\n"
            f"📄 Articles: {self.articles_processed:,} "
            f"(skipped {self.articles_skipped:,})\n"
            f"🧩 Chunks: {self.chunks_created:,}\n"
            f"🔗 Links: {self.links_extracted:,}\n"
            f"🏷️  Categories: {self.categories_extracted:,}\n"
            f"👤 Entities: {self.entities_extracted:,}\n"
            f"⚡ Rate: {rate:.1f} articles/sec\n"
            f"❌ Errors: {self.errors}\n"
            f"📦 Parts: {self.parts_processed}/{self.parts_total}\n"
            f"{'='*70}"
        )


# =============================================================================
# BATCH CONTAINERS
# =============================================================================


@dataclass
class ArticleBatch:
    """A batch of articles ready for processing."""

    articles: list[ProcessedArticle]
    batch_id: int

    @property
    def size(self) -> int:
        return len(self.articles)

    @property
    def chunk_count(self) -> int:
        return sum(len(a.chunks) for a in self.articles)


@dataclass
class EmbeddedBatch:
    """A batch with computed embeddings, ready for upload."""

    articles: list[ProcessedArticle]
    batch_id: int

    @property
    def size(self) -> int:
        return len(self.articles)

    @property
    def all_chunks(self) -> list[ProcessedChunk]:
        return [c for a in self.articles for c in a.chunks]
