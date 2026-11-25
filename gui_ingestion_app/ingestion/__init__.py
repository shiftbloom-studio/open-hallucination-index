"""
High-Performance Wikipedia Ingestion Pipeline
=============================================

A modular, highly optimized ingestion system for Wikipedia data into
Neo4j (graph) and Qdrant (vector) databases with maximum throughput.

Architecture:
- Producer-Consumer pattern with async queues
- Multi-threaded downloads with parallel processing
- GPU-accelerated embeddings with batching
- Async database uploads for both stores
- Rich relationship extraction for graph traversal
- SQL dump integration for complete metadata (Wikidata, geo, categories)
- Post-ingestion optimization (PageRank, link resolution, geo proximity)

Modules:
- models: Data classes for articles, chunks, and configs
- downloader: Async Wikipedia dump file downloading
- preprocessor: Text cleaning, chunking, and BM25 tokenization
- qdrant_store: Async vector store with hybrid search + geo
- neo4j_store: Graph store with rich relationship modeling
- pipeline: Producer-consumer orchestration
- checkpoint: Resumable ingestion state management
- sql_parsers: Parsers for Wikipedia SQL dump files
- optimizer: Post-ingestion database optimization

New in v4.0.0:
- Geographic coordinate support (from geo_tags.sql.gz)
- Wikidata Q-ID integration (from page_props.sql.gz)
- Quality scoring for evidence ranking
- PageRank computation for article importance
- Category hierarchy relationships
- Geographic proximity relationships
"""

from ingestion.checkpoint import CheckpointManager
from ingestion.downloader import ChunkedWikiDownloader, DumpFile, LocalFileParser
from ingestion.models import (
    IngestionConfig,
    PipelineStats,
    ProcessedChunk,
    WikiArticle,
    WikiInfobox,
    WikiSection,
)
from ingestion.neo4j_store import Neo4jGraphStore
from ingestion.pipeline import IngestionPipeline, run_ingestion
from ingestion.preprocessor import AdvancedTextPreprocessor, BM25Tokenizer
from ingestion.qdrant_store import QdrantHybridStore

# Optional: SQL dump parsers (may have additional dependencies)
try:
    from ingestion.sql_parsers import (
        WikipediaLookupTables,
        load_all_lookup_tables,
        parse_categorylinks,
        parse_geo_tags,
        parse_page_props,
    )
    SQL_PARSERS_AVAILABLE = True
except ImportError:
    SQL_PARSERS_AVAILABLE = False
    WikipediaLookupTables = None  # type: ignore
    load_all_lookup_tables = None  # type: ignore

# Optional: Optimizer (requires Neo4j connection)
try:
    from ingestion.optimizer import Neo4jOptimizer, run_optimization
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    Neo4jOptimizer = None  # type: ignore
    run_optimization = None  # type: ignore

__all__ = [
    # Models
    "WikiArticle",
    "WikiSection",
    "WikiInfobox",
    "ProcessedChunk",
    "IngestionConfig",
    "PipelineStats",
    # Downloader
    "ChunkedWikiDownloader",
    "DumpFile",
    "LocalFileParser",
    # Preprocessor
    "AdvancedTextPreprocessor",
    "BM25Tokenizer",
    # Stores
    "QdrantHybridStore",
    "Neo4jGraphStore",
    # Pipeline
    "IngestionPipeline",
    "CheckpointManager",
    "run_ingestion",
    # SQL Parsers (optional)
    "WikipediaLookupTables",
    "load_all_lookup_tables",
    "SQL_PARSERS_AVAILABLE",
    # Optimizer (optional)
    "Neo4jOptimizer",
    "run_optimization",
    "OPTIMIZER_AVAILABLE",
]

__version__ = "4.0.0"
