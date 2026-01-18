#!/usr/bin/env python3
"""
High-Performance Wikipedia Dual-Ingestion Tool v3.0
====================================================

Downloads, parses, and ingests Wikipedia articles into BOTH:
1. Qdrant (Vector Store): Hybrid search with Dense + Sparse (BM25) vectors
2. Neo4j (Graph Store): Rich knowledge graph with many relationship types

Key Features v3.0:
- Producer-Consumer architecture for maximum throughput
- Separate download threads (no blocking)
- GPU-accelerated embeddings with large batches
- Async non-blocking uploads to both stores
- Rich relationship extraction (10+ relationship types)
- Real-time progress with responsive UI

Performance Improvements:
- 10-50x faster than v2.0 (expected 50-200 articles/sec with GPU)
- Non-blocking downloads run ahead of processing
- Preprocessing parallelized across CPU cores
- GPU always saturated with embedding work
- Database uploads happen async

Usage:
    python -m ingestion --limit 10000 --batch-size 256
    python ingestion/__main__.py --endless --keep-downloads
"""

from __future__ import annotations

import argparse
import logging
import sys

from ingestion.models import IngestionConfig
from ingestion.pipeline import run_ingestion

logger = logging.getLogger("ingest_wiki")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="High-Performance Wikipedia Dual Ingestion (Neo4j + Qdrant) v3.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Connection settings
    conn_group = parser.add_argument_group("Connection Settings")
    conn_group.add_argument(
        "--qdrant-host", default="localhost", help="Qdrant host"
    )
    conn_group.add_argument(
        "--qdrant-port", type=int, default=6333, help="Qdrant HTTP port"
    )
    conn_group.add_argument(
        "--qdrant-grpc-port", type=int, default=6334, help="Qdrant gRPC port"
    )
    conn_group.add_argument(
        "--qdrant-https",
        action="store_true",
        help="Use HTTPS/TLS when connecting to Qdrant",
    )
    conn_group.add_argument(
        "--qdrant-tls-ca-cert",
        default=None,
        help="Path to Qdrant TLS CA certificate",
    )
    conn_group.add_argument(
        "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI"
    )

    conn_group.add_argument(
        "--neo4j-user", default="neo4j", help="Neo4j username"
    )
    conn_group.add_argument(
        "--neo4j-pass", default="password123", help="Neo4j password"
    )
    conn_group.add_argument(
        "--neo4j-db", default="neo4j", help="Neo4j database name"
    )

    # Processing settings
    proc_group = parser.add_argument_group("Processing Settings")
    proc_group.add_argument(
        "--limit", type=int, default=None, help="Maximum articles to process"
    )
    proc_group.add_argument(
        "--batch-size",
        type=int,
        default=384,
        help="Articles per batch (larger = faster, more memory)",
    )
    proc_group.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in characters",
    )
    proc_group.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Chunk overlap in characters",
    )

    # Parallelism settings
    parallel_group = parser.add_argument_group("Parallelism Settings")
    parallel_group.add_argument(
        "--preprocess-workers",
        type=int,
        default=10,
        help="Number of preprocessing worker threads",
    )
    parallel_group.add_argument(
        "--dump-workers",
        type=int,
        default=8,
        help="Number of dump file worker threads",
    )
    parallel_group.add_argument(
        "--upload-workers",
        type=int,
        default=24,
        help="Number of upload worker threads per store",
    )
    parallel_group.add_argument(
        "--embedding-workers",
        type=int,
        default=12,
        help="Number of embedding worker threads",
    )
    parallel_group.add_argument(
        "--embedding-batch-size",
        type=int,
        default=768,
        help="Batch size for GPU embedding (larger = faster)",
    )
    parallel_group.add_argument(
        "--embedding-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Embedding device selection (auto = use CUDA if available)",
    )

    # Download settings
    dl_group = parser.add_argument_group("Download Settings")
    dl_group.add_argument(
        "--download-dir",
        type=str,
        default=".wiki_dumps",
        help="Directory for temporary dump file downloads",
    )
    dl_group.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded files after processing",
    )
    dl_group.add_argument(
        "--max-concurrent-downloads",
        type=int,
        default=4,
        help="Maximum concurrent dump file downloads",
    )

    # Collection settings
    coll_group = parser.add_argument_group("Collection Settings")
    coll_group.add_argument(
        "--collection",
        default="wikipedia_hybrid",
        help="Qdrant collection name",
    )

    # Checkpoint settings
    ckpt_group = parser.add_argument_group("Checkpoint Settings")
    ckpt_group.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint file path (default: .ingest_checkpoint.json)",
    )
    ckpt_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Reset statistics but keep processed article IDs (prevents duplicates)",
    )
    ckpt_group.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="DELETE all checkpoint data including processed IDs (allows re-import)",
    )

    # Retry settings
    retry_group = parser.add_argument_group("Retry Settings")
    retry_group.add_argument(
        "--endless",
        action="store_true",
        help="Keep retrying on network errors until complete",
    )
    retry_group.add_argument(
        "--endless-retry-delay",
        type=int,
        default=30,
        help="Seconds to wait before retrying after error",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build configuration from args
    config = IngestionConfig(
        # Connection
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_grpc_port=args.qdrant_grpc_port,
        qdrant_https=args.qdrant_https,
        qdrant_tls_ca_cert=args.qdrant_tls_ca_cert,
        neo4j_uri=args.neo4j_uri,

        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_pass,
        neo4j_database=args.neo4j_db,
        # Collection
        qdrant_collection=args.collection,
        # Processing
        limit=args.limit,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        # Parallelism
        dump_workers=args.dump_workers,
        preprocess_workers=args.preprocess_workers,
        upload_workers=args.upload_workers,
        embedding_batch_size=args.embedding_batch_size,
        embedding_workers=args.embedding_workers,
        embedding_device=args.embedding_device,
        # Download
        download_dir=args.download_dir,
        keep_downloads=args.keep_downloads,
        max_concurrent_downloads=args.max_concurrent_downloads,
        # Checkpoint
        checkpoint_file=args.checkpoint_file,
        # Retry
        endless_mode=args.endless,
        endless_retry_delay=args.endless_retry_delay,
    )

    # Run the pipeline
    stats = run_ingestion(
        config=config,
        no_resume=args.no_resume,
        clear_checkpoint=args.clear_checkpoint,
        limit=args.limit,
    )

    # Exit with appropriate code
    if stats.errors > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
