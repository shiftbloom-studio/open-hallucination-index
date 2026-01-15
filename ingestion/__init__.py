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

Modules:
- models: Data classes for articles, chunks, and configs
- downloader: Async Wikipedia dump file downloading
- preprocessor: Text cleaning, chunking, and BM25 tokenization
- qdrant_store: Async vector store with hybrid search
- neo4j_store: Graph store with rich relationship modeling
- pipeline: Producer-consumer orchestration
- checkpoint: Resumable ingestion state management
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
from ingestion.pipeline import IngestionPipeline
from ingestion.preprocessor import AdvancedTextPreprocessor, BM25Tokenizer
from ingestion.qdrant_store import QdrantHybridStore

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
]

__version__ = "3.0.0"
