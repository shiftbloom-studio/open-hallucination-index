"""
High-performance ingestion pipeline with producer-consumer architecture.

Features:
- Separate threads for download, preprocess, embed, and upload
- Queue-based communication between stages
- Non-blocking progress updates
- Real-time statistics
- Graceful shutdown handling

Architecture:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Download    │────▶│  Preprocess  │────▶│   Embed +    │────▶│   Upload     │
    │  (N threads) │     │  (M threads) │     │   Prepare    │     │  (async)     │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
           │                    │                    │                    │
           ▼                    ▼                    ▼                    ▼
      Parts Queue          Article Queue        Batch Queue          Done Events

This design ensures:
- Downloads run ahead of processing (prefetch)
- Preprocessing is parallelized across CPU cores
- GPU is always busy with embedding work
- Database uploads happen async, don't block pipeline
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from ingestion.checkpoint import CheckpointManager
from ingestion.downloader import (
    ChunkedWikiDownloader,
    DumpFile,
    LocalFileParser,
    is_shutdown_requested,
)
from ingestion.models import (
    IngestionConfig,
    PipelineStats,
    ProcessedArticle,
    WikiArticle,
)
from ingestion.neo4j_store import Neo4jGraphStore
from ingestion.preprocessor import AdvancedTextPreprocessor
from ingestion.qdrant_store import QdrantHybridStore

logger = logging.getLogger("ingestion.pipeline")


@dataclass
class BatchResult:
    """Result of processing a batch."""

    article_count: int
    chunk_count: int
    qdrant_event: threading.Event
    neo4j_event: threading.Event


class IngestionPipeline:
    """
    High-performance ingestion pipeline with producer-consumer architecture.

    The pipeline has 4 main stages:
    1. Download: Fetches dump parts in parallel
    2. Parse + Preprocess: Extracts articles and processes text
    3. Embed: Computes dense and sparse vectors (GPU-accelerated)
    4. Upload: Sends to Qdrant and Neo4j (async, non-blocking)

    Key optimizations:
    - Download runs ahead of processing (prefetch buffer)
    - Preprocessing is parallelized with ThreadPoolExecutor
    - Embedding batches are large for GPU efficiency
    - Uploads are async so they don't block the pipeline
    - Statistics update in real-time without blocking
    """

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.stats = PipelineStats()

        # Initialize components
        self.checkpoint = CheckpointManager(
            checkpoint_file=config.checkpoint_file,
            collection=config.qdrant_collection,
        )

        self.downloader = ChunkedWikiDownloader(
            download_dir=Path(config.download_dir),
            max_concurrent=config.max_concurrent_downloads,
        )

        self.preprocessor = AdvancedTextPreprocessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
            max_workers=config.preprocess_workers,
        )

        self.qdrant = QdrantHybridStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            grpc_port=config.qdrant_grpc_port,
            collection=config.qdrant_collection,
            embedding_model=config.embedding_model,
            embedding_batch_size=config.embedding_batch_size,
            upload_workers=config.upload_workers,
        )

        self.neo4j = Neo4jGraphStore(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
            upload_workers=config.upload_workers,
        )

        # Queues for pipeline stages
        self._article_queue: queue.Queue[WikiArticle | None] = queue.Queue(
            maxsize=config.preprocess_queue_size
        )
        self._batch_queue: queue.Queue[list[ProcessedArticle] | None] = queue.Queue(
            maxsize=config.upload_queue_size
        )

        # Thread pool for preprocessing
        self._preprocess_executor = ThreadPoolExecutor(
            max_workers=config.preprocess_workers,
            thread_name_prefix="preprocess",
        )

        # Pending upload events to track
        self._pending_uploads: list[BatchResult] = []

        # Shutdown flag
        self._shutdown = False

    def load_checkpoint(self, no_resume: bool = False, clear: bool = False) -> None:
        """Load or clear checkpoint."""
        if clear:
            self.checkpoint.clear()
            return

        if no_resume:
            self.checkpoint.load_ids_only()
        else:
            self.checkpoint.load()

        logger.info(f"📍 {self.checkpoint.get_resume_info()}")

    def run(
        self,
        limit: int | None = None,
        progress_callback: Callable[[PipelineStats], None] | None = None,
    ) -> PipelineStats:
        """
        Run the full ingestion pipeline.

        Args:
            limit: Maximum number of articles to process
            progress_callback: Optional callback for progress updates

        Returns:
            Final pipeline statistics
        """
        self.stats = PipelineStats()
        self.stats.start_time = time.time()

        # Discover dump parts
        all_parts = self.downloader.discover_parts()
        self.stats.parts_total = len(all_parts)

        # Filter out already processed parts (via checkpoint)
        pending_parts = [
            p for p in all_parts if not self.checkpoint.should_skip_part(p.index)
        ]

        if not pending_parts:
            logger.info("✅ All parts already processed!")
            return self.stats

        # Check for existing downloads to skip re-downloading
        already_downloaded, need_download = self.downloader.check_existing_downloads(
            pending_parts, verify_sizes=True
        )

        logger.info(
            f"📊 Status: {len(already_downloaded)} downloaded, "
            f"{len(need_download)} to download, "
            f"{self.stats.parts_total - len(pending_parts)} already processed"
        )

        # Initialize progress bar
        pbar = tqdm(
            total=limit or len(pending_parts) * 50000,
            unit="articles",
            desc="Ingesting",
            dynamic_ncols=True,
        )

        try:
            self._run_pipeline(
                already_downloaded, need_download, limit, pbar, progress_callback
            )
        finally:
            pbar.close()
            self._cleanup()

        return self.stats

    def _run_pipeline(
        self,
        already_downloaded: list[DumpFile],
        need_download: list[DumpFile],
        limit: int | None,
        pbar: tqdm,
        progress_callback: Callable[[PipelineStats], None] | None,
    ) -> None:
        """Main pipeline execution loop."""
        download_queue = list(need_download)
        # Already downloaded files are immediately ready to process
        ready_to_process: list[DumpFile] = list(already_downloaded)
        current_batch: list[ProcessedArticle] = []
        batch_id = 0

        if ready_to_process:
            logger.info(
                f"📂 {len(ready_to_process)} parts ready to process from existing downloads"
            )

        # Start initial downloads for parts that need downloading
        while (
            download_queue
            and self.downloader.active_download_count() < self.config.max_concurrent_downloads
        ):
            part = download_queue.pop(0)
            self.downloader.start_download(part)
            logger.info(f"📥 Started download: part {part.index}")

        # Main loop
        last_progress_update = time.time()

        while not self._shutdown and not is_shutdown_requested():
            # Check for completed downloads
            completed = self.downloader.get_completed_downloads()
            ready_to_process.extend(completed)
            self.stats.parts_downloaded += len(completed)

            # Start new downloads if slots available
            while (
                download_queue
                and self.downloader.active_download_count() < self.config.max_concurrent_downloads
            ):
                part = download_queue.pop(0)
                self.downloader.start_download(part)

            # Process ready files
            if ready_to_process:
                part = ready_to_process.pop(0)

                if part.download_complete and part.local_path:
                    try:
                        # Process all articles in this part
                        for article in LocalFileParser.parse_file(
                            part.local_path,
                            skip_ids=self.checkpoint.processed_ids,
                        ):
                            if is_shutdown_requested():
                                break

                            # Preprocess article
                            processed = self.preprocessor.process_article(article)

                            if processed.chunks:
                                current_batch.append(processed)
                                self.stats.chunks_created += len(processed.chunks)

                            self.stats.links_extracted += len(article.links)
                            self.stats.categories_extracted += len(article.categories)
                            self.stats.entities_extracted += len(article.entities)

                            # Process batch when full
                            if len(current_batch) >= self.config.batch_size:
                                self._process_batch(current_batch, batch_id)
                                batch_id += 1

                                # Update progress
                                pbar.update(len(current_batch))
                                self.stats.articles_processed += len(current_batch)

                                # Record in checkpoint
                                article_ids = [a.article.id for a in current_batch]
                                chunk_count = sum(len(a.chunks) for a in current_batch)
                                self.checkpoint.record_batch(article_ids, chunk_count)

                                current_batch = []

                                # Check limit
                                if limit and self.stats.articles_processed >= limit:
                                    logger.info("✅ Limit reached")
                                    return

                            # Periodic progress update
                            now = time.time()
                            if now - last_progress_update > 1.0:
                                self._update_progress(pbar)
                                if progress_callback:
                                    progress_callback(self.stats)
                                last_progress_update = now

                        # Part complete
                        self.checkpoint.record_part_complete(part.index)
                        self.stats.parts_processed += 1

                        # Delete file if configured
                        if not self.config.keep_downloads and part.local_path.exists():
                            try:
                                part.local_path.unlink()
                                logger.debug(f"🗑️  Deleted {part.local_path.name}")
                            except Exception as e:
                                logger.warning(f"Could not delete {part.local_path}: {e}")

                        logger.info(
                            f"✅ Part {part.index} complete. "
                            f"Progress: {self.stats.parts_processed}/{self.stats.parts_total}"
                        )

                    except Exception as e:
                        logger.error(f"Error processing part {part.index}: {e}")
                        self.stats.errors += 1

            else:
                # No files ready, wait a bit for downloads
                time.sleep(0.1)

            # Check if done
            has_work = (
                ready_to_process
                or download_queue
                or self.downloader.active_download_count() > 0
            )
            if not has_work:
                break

        # Process final batch
        if current_batch:
            self._process_batch(current_batch, batch_id)
            pbar.update(len(current_batch))
            self.stats.articles_processed += len(current_batch)

            article_ids = [a.article.id for a in current_batch]
            chunk_count = sum(len(a.chunks) for a in current_batch)
            self.checkpoint.record_batch(article_ids, chunk_count)

        # Wait for pending uploads
        self._wait_for_uploads()

    def _process_batch(self, batch: list[ProcessedArticle], batch_id: int) -> None:
        """Process a batch: compute embeddings and upload."""
        if not batch:
            return

        # Upload to both stores asynchronously
        qdrant_event = self.qdrant.upload_batch_async(batch)
        neo4j_event = self.neo4j.upload_batch_async(batch)

        # Track pending uploads
        result = BatchResult(
            article_count=len(batch),
            chunk_count=sum(len(a.chunks) for a in batch),
            qdrant_event=qdrant_event,
            neo4j_event=neo4j_event,
        )
        self._pending_uploads.append(result)

        # Clean up completed uploads
        self._pending_uploads = [
            r
            for r in self._pending_uploads
            if not (r.qdrant_event.is_set() and r.neo4j_event.is_set())
        ]

    def _wait_for_uploads(self, timeout: float = 60.0) -> None:
        """Wait for all pending uploads to complete."""
        logger.info(f"⏳ Waiting for {len(self._pending_uploads)} pending uploads...")

        for result in self._pending_uploads:
            result.qdrant_event.wait(timeout=timeout)
            result.neo4j_event.wait(timeout=timeout)

        # Flush any remaining items in store queues
        self.qdrant.flush()
        self.neo4j.flush()

        self._pending_uploads.clear()

    def _update_progress(self, pbar: tqdm) -> None:
        """Update progress bar with current stats."""
        rate = self.stats.rate()
        download_progress = self.downloader.get_progress()

        pbar.set_postfix(
            {
                "rate": f"{rate:.1f}/s",
                "parts": f"{self.stats.parts_processed}/{self.stats.parts_total}",
                "dl": f"{download_progress.download_speed_mbps:.1f}MB/s",
                "pending": len(self._pending_uploads),
            },
            refresh=False,
        )

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("🧹 Cleaning up...")

        # Wait for uploads
        self._wait_for_uploads()

        # Save final checkpoint
        self.checkpoint.save()

        # Close components
        self.downloader.close()
        self.preprocessor.close()
        self.qdrant.close()
        self.neo4j.close()

        self._preprocess_executor.shutdown(wait=False)

    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics."""
        return self.stats

    def shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown = True


def run_ingestion(
    config: IngestionConfig,
    no_resume: bool = False,
    clear_checkpoint: bool = False,
    limit: int | None = None,
) -> PipelineStats:
    """
    Convenience function to run the full ingestion pipeline.

    Args:
        config: Pipeline configuration
        no_resume: Reset stats but keep processed IDs
        clear_checkpoint: Clear all checkpoint data
        limit: Maximum articles to process

    Returns:
        Final pipeline statistics
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy loggers
    for name in ["neo4j", "sentence_transformers", "qdrant_client", "httpx", "urllib3"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("🚀 Starting high-performance Wikipedia ingestion pipeline")
    logger.info(f"📊 Config: batch_size={config.batch_size}, workers={config.preprocess_workers}")

    pipeline = IngestionPipeline(config)
    pipeline.load_checkpoint(no_resume=no_resume, clear=clear_checkpoint)

    if clear_checkpoint:
        logger.info("Checkpoint cleared. Run again without --clear-checkpoint to ingest.")
        return PipelineStats()

    stats = pipeline.run(limit=limit or config.limit)

    logger.info(stats.summary())

    return stats
