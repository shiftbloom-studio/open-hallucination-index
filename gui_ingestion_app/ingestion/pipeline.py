"""
High-performance ingestion pipeline with producer-consumer architecture.

Features:
- Parallel dump file processing with configurable workers
- Separate threads for download, preprocess, embed, and upload
- Queue-based communication between stages
- Non-blocking progress updates
- Real-time statistics with thread-safe counters
- Graceful shutdown handling

Architecture:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Download    │────▶│  Dump Worker │────▶│   Embed +    │────▶│   Upload     │
    │  (N threads) │     │  (K workers) │     │   Prepare    │     │  (async)     │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
           │                    │                    │                    │
           ▼                    ▼                    ▼                    ▼
      Parts Queue          Preprocess Pool       Batch Queue          Done Events

This design ensures:
- Downloads run ahead of processing (prefetch)
- Multiple dump files are processed in parallel
- Preprocessing is parallelized across CPU cores
- GPU is always busy with embedding work
- Database uploads happen async, don't block pipeline
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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

# Optional: SQL dump parsers for enhanced metadata
SQL_PARSERS_AVAILABLE = False
load_all_lookup_tables = None  # type: ignore
WikipediaLookupTables = None  # type: ignore

try:
    from ingestion.sql_parsers import (
        WikipediaLookupTables as _WikipediaLookupTables,
        load_all_lookup_tables as _load_all_lookup_tables,
    )
    SQL_PARSERS_AVAILABLE = True
    load_all_lookup_tables = _load_all_lookup_tables
    WikipediaLookupTables = _WikipediaLookupTables
except ImportError:
    pass

logger = logging.getLogger("ingestion.pipeline")


@dataclass
class BatchResult:
    """Result of processing a batch with upload tracking."""

    article_count: int
    chunk_count: int
    qdrant_event: threading.Event
    neo4j_event: threading.Event
    batch_data: list[ProcessedArticle]  # Keep data for retry if needed
    article_ids: list[int]  # For checkpoint recording
    upload_attempts: int = 0
    max_attempts: int = 3
    checkpoint_recorded: bool = False


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

    def __init__(self, config: IngestionConfig, lookup_tables = None):
        self.config = config
        self.stats = PipelineStats()
        self.lookup_tables = lookup_tables

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
            lookup_tables=lookup_tables,  # Pass lookup tables to preprocessor
        )

        self.qdrant = QdrantHybridStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            grpc_port=config.qdrant_grpc_port,
            collection=config.qdrant_collection,
            embedding_model=config.embedding_model,
            embedding_batch_size=config.embedding_batch_size,
            upload_workers=config.upload_workers,
            embedding_workers=config.embedding_workers,
            embedding_device=config.embedding_device,
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

        # Thread pool for parallel dump file processing
        self._dump_executor = ThreadPoolExecutor(
            max_workers=config.dump_workers,
            thread_name_prefix="dump_worker",
        )

        # Pending upload events to track
        self._pending_uploads: list[BatchResult] = []

        # Thread-safe synchronization
        self._stats_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._batch_id_counter = itertools.count()
        self._last_progress_update = 0.0

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
        """Main pipeline execution loop with parallel dump workers."""
        download_queue = list(need_download)
        ready_to_process: list[DumpFile] = list(already_downloaded)
        part_futures: dict[Future, DumpFile] = {}

        self._stop_event.clear()
        self._last_progress_update = time.time()

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

        # Main coordination loop
        while not self._shutdown and not is_shutdown_requested():
            # Check for completed downloads
            completed = self.downloader.get_completed_downloads()
            ready_to_process.extend(completed)
            if ready_to_process:
                ready_to_process.sort(key=lambda p: p.index)
            with self._stats_lock:
                self.stats.parts_downloaded += len(completed)

            # Start new downloads if slots available and not stopping
            if not self._stop_event.is_set():
                while (
                    download_queue
                    and self.downloader.active_download_count() < self.config.max_concurrent_downloads
                ):
                    part = download_queue.pop(0)
                    self.downloader.start_download(part)

            # Schedule dump workers for ready files
            while (
                ready_to_process
                and len(part_futures) < self.config.dump_workers
                and not self._stop_event.is_set()
            ):
                part = ready_to_process.pop(0)
                if part.download_complete and part.local_path:
                    future = self._dump_executor.submit(
                        self._process_dump_part, part, limit, pbar, progress_callback
                    )
                    part_futures[future] = part

            # Handle completed futures
            completed_futures = [f for f in part_futures if f.done()]
            for future in completed_futures:
                part = part_futures.pop(future)
                try:
                    future.result()  # Raises exception if worker failed
                except Exception as e:
                    logger.error(f"Worker error for part {part.index}: {e}")
                    with self._stats_lock:
                        self.stats.errors += 1

            # Periodic progress update from main thread
            now = time.time()
            if now - self._last_progress_update > 1.0:
                self._update_progress(pbar)
                if progress_callback:
                    with self._stats_lock:
                        progress_callback(self.stats)
                self._last_progress_update = now

            # Check if done
            has_work = (
                ready_to_process
                or download_queue
                or self.downloader.active_download_count() > 0
                or part_futures
            )
            if not has_work:
                break

            time.sleep(0.05)  # Small sleep to prevent busy-waiting

        # Wait for pending uploads
        self._wait_for_uploads()

    def _process_dump_part(
        self,
        part: DumpFile,
        limit: int | None,
        pbar: tqdm,
        progress_callback: Callable[[PipelineStats], None] | None,
    ) -> None:
        """
        Process a single dump part in a worker thread.

        This method is designed to run in parallel with other dump workers.
        All shared state access is thread-safe.
        """
        if not part.download_complete or not part.local_path:
            return

        try:
            # Get snapshot of processed IDs for skip checking
            with self._checkpoint_lock:
                skip_ids = set(self.checkpoint.processed_ids)

            article_batch: list[WikiArticle] = []

            for article in LocalFileParser.parse_file(
                part.local_path,
                skip_ids=skip_ids,
            ):
                if self._shutdown or self._stop_event.is_set() or is_shutdown_requested():
                    break

                article_batch.append(article)

                # Process batch when full
                if len(article_batch) >= self.config.batch_size:
                    self._process_article_batch(
                        article_batch, limit, pbar, progress_callback
                    )
                    article_batch = []

                    # Check if limit reached
                    if self._stop_event.is_set():
                        break

            # Process remaining articles in partial batch
            if article_batch and not self._stop_event.is_set():
                self._process_article_batch(
                    article_batch, limit, pbar, progress_callback
                )

            # Mark part complete (thread-safe)
            with self._checkpoint_lock:
                self.checkpoint.record_part_complete(part.index)
            with self._stats_lock:
                self.stats.parts_processed += 1

            # Delete file if configured
            if not self.config.keep_downloads and part.local_path.exists():
                try:
                    part.local_path.unlink()
                    logger.debug(f"🗑️  Deleted {part.local_path.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {part.local_path}: {e}")

            with self._stats_lock:
                parts_done = self.stats.parts_processed
                parts_total = self.stats.parts_total
            logger.info(
                f"✅ Part {part.index} complete. Progress: {parts_done}/{parts_total}"
            )

        except Exception as e:
            logger.error(f"Error processing part {part.index}: {e}")
            with self._stats_lock:
                self.stats.errors += 1
            raise  # Re-raise so future.result() captures it

    def _process_article_batch(
        self,
        articles: list[WikiArticle],
        limit: int | None,
        pbar: tqdm,
        progress_callback: Callable[[PipelineStats], None] | None,
    ) -> None:
        """
        Preprocess a batch of articles and enqueue for upload.

        Uses the preprocessor's thread pool for parallel preprocessing,
        then enqueues for async embedding and upload.
        """
        # Parallel preprocessing
        processed = self.preprocessor.process_batch(articles)
        if not processed:
            return

        # Filter articles with chunks for upload
        upload_batch = [pa for pa in processed if pa.chunks]
        if upload_batch:
            batch_id = next(self._batch_id_counter)
            self._process_batch(upload_batch, batch_id)

        # Update stats and checkpoint (thread-safe)
        with self._stats_lock:
            processed_count = len(processed)
            chunk_count = sum(len(pa.chunks) for pa in processed)

            self.stats.articles_processed += processed_count
            self.stats.articles_skipped += processed_count - len(upload_batch)
            self.stats.chunks_created += chunk_count
            self.stats.links_extracted += sum(len(pa.article.links) for pa in processed)
            self.stats.categories_extracted += sum(
                len(pa.article.categories) for pa in processed
            )
            self.stats.entities_extracted += sum(
                len(pa.article.entities) for pa in processed
            )

            pbar.update(processed_count)

            # Check limit
            if limit and self.stats.articles_processed >= limit:
                logger.info("✅ Limit reached")
                self._stop_event.set()
        
        # Note: Checkpoint recording now happens in _process_batch after upload verification

    def _process_batch(self, batch: list[ProcessedArticle], batch_id: int) -> None:
        """Process a batch: compute embeddings and upload."""
        if not batch:
            return

        # Backpressure: avoid stacking too many pending uploads
        self._throttle_pending_uploads()

        # Upload to both stores asynchronously
        qdrant_event = self.qdrant.upload_batch_async(batch)
        neo4j_event = self.neo4j.upload_batch_async(batch)

        # Extract article IDs for checkpoint
        article_ids = [pa.article.id for pa in batch]

        # Track pending uploads with batch data for retry
        result = BatchResult(
            article_count=len(batch),
            chunk_count=sum(len(a.chunks) for a in batch),
            qdrant_event=qdrant_event,
            neo4j_event=neo4j_event,
            batch_data=batch,  # Keep data for potential retry
            article_ids=article_ids,
        )
        self._pending_uploads.append(result)

        # Clean up completed uploads and record in checkpoint
        self._drain_completed_uploads()

    def _drain_completed_uploads(self) -> None:
        """Record and remove completed uploads from the pending list."""
        completed = []
        for r in self._pending_uploads:
            if r.qdrant_event.is_set() and r.neo4j_event.is_set() and not r.checkpoint_recorded:
                with self._checkpoint_lock:
                    self.checkpoint.record_batch(r.article_ids, r.chunk_count)
                r.checkpoint_recorded = True
                completed.append(r)

        if completed:
            self._pending_uploads = [r for r in self._pending_uploads if r not in completed]

    def _throttle_pending_uploads(self) -> None:
        """Apply backpressure to avoid unbounded pending upload growth."""
        max_pending = getattr(self.config, "max_pending_batches", 0)
        if not max_pending or max_pending <= 0:
            return

        while (
            len(self._pending_uploads) >= max_pending
            and not self._shutdown
            and not self._stop_event.is_set()
        ):
            self._drain_completed_uploads()
            if len(self._pending_uploads) < max_pending:
                break

            # Wait briefly for the oldest batch to finish
            oldest = self._pending_uploads[0]
            oldest.qdrant_event.wait(timeout=1.0)
            oldest.neo4j_event.wait(timeout=1.0)

    def _wait_for_uploads(self, timeout: float = 60.0) -> None:
        """
        Wait for all pending uploads to complete with retry logic.
        
        This ensures all batches are successfully uploaded before checkpointing.
        Failed batches are retried up to max_attempts times.
        Only successful uploads are recorded in the checkpoint.
        """
        logger.info(f"⏳ Waiting for {len(self._pending_uploads)} pending uploads...")

        failed_batches = []
        successful_batches = []
        
        for result in self._pending_uploads:
            # Wait for both uploads to complete
            qdrant_done = result.qdrant_event.wait(timeout=timeout)
            neo4j_done = result.neo4j_event.wait(timeout=timeout)
            
            # Check if uploads succeeded
            if qdrant_done and neo4j_done:
                successful_batches.append(result)
            else:
                # Upload timed out or failed
                if result.upload_attempts < result.max_attempts:
                    logger.warning(
                        f"⚠️ Upload timeout for batch (attempt {result.upload_attempts + 1}/"
                        f"{result.max_attempts}), will retry"
                    )
                    failed_batches.append(result)
                else:
                    logger.error(
                        f"❌ Upload failed for batch after {result.max_attempts} attempts, "
                        f"skipping {result.article_count} articles"
                    )

        # Flush any remaining items in store queues
        self.qdrant.flush()
        self.neo4j.flush()

        # Retry failed batches with proper re-queuing for multiple attempts
        while failed_batches:
            logger.info(f"🔄 Retrying {len(failed_batches)} failed batches...")
            still_failed = []
            
            for result in failed_batches:
                result.upload_attempts += 1
                logger.info(
                    f"Retrying batch (attempt {result.upload_attempts}/{result.max_attempts})..."
                )
                
                # Re-upload with fresh events
                result.qdrant_event = self.qdrant.upload_batch_async(result.batch_data)
                result.neo4j_event = self.neo4j.upload_batch_async(result.batch_data)
                
                # Wait for retry
                qdrant_done = result.qdrant_event.wait(timeout=timeout)
                neo4j_done = result.neo4j_event.wait(timeout=timeout)
                
                if qdrant_done and neo4j_done:
                    successful_batches.append(result)
                    logger.info("✅ Batch retry succeeded")
                else:
                    if result.upload_attempts < result.max_attempts:
                        logger.warning(
                            f"⚠️ Retry failed, will try again "
                            f"({result.upload_attempts}/{result.max_attempts})"
                        )
                        still_failed.append(result)  # Re-queue for another retry
                    else:
                        logger.error(
                            f"❌ Batch upload failed permanently after {result.max_attempts} attempts"
                        )
            
            failed_batches = still_failed  # Continue with remaining failed batches

        # Record successful uploads in checkpoint (only after verification!)
        if successful_batches:
            with self._checkpoint_lock:
                for result in successful_batches:
                    if not result.checkpoint_recorded:
                        self.checkpoint.record_batch(result.article_ids, result.chunk_count)
                        result.checkpoint_recorded = True
            
            logger.info(f"💾 Recorded {len(successful_batches)} successful batches in checkpoint")

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

        self._preprocess_executor.shutdown(wait=True, cancel_futures=True)
        self._dump_executor.shutdown(wait=True, cancel_futures=True)

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
    load_sql_dumps: bool = False,
    run_post_optimization: bool = False,
) -> PipelineStats:
    """
    Convenience function to run the full ingestion pipeline.

    Args:
        config: Pipeline configuration
        no_resume: Reset stats but keep processed IDs
        clear_checkpoint: Clear all checkpoint data
        limit: Maximum articles to process
        load_sql_dumps: Load SQL dump lookup tables for enrichment
        run_post_optimization: Run post-ingestion optimization (PageRank, etc.)

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
    logger.info(
        f"📊 Config: batch_size={config.batch_size}, "
        f"dump_workers={config.dump_workers}, "
        f"preprocess_workers={config.preprocess_workers}, "
        f"embedding_workers={config.embedding_workers}"
    )

    # Load SQL dump lookup tables if requested
    lookup_tables = None
    if load_sql_dumps and SQL_PARSERS_AVAILABLE and load_all_lookup_tables is not None:
        logger.info("📚 Loading SQL dump lookup tables for enrichment...")
        try:
            lookup_tables = load_all_lookup_tables(Path(config.download_dir))
            if lookup_tables is not None:
                logger.info(
                    f"✅ Loaded: {len(lookup_tables.wikidata_ids):,} Wikidata IDs, "
                    f"{len(lookup_tables.geo_coordinates):,} geo coords, "
                    f"{len(lookup_tables.categories):,} category assignments"
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to load SQL dumps: {e}")
            lookup_tables = None
    elif load_sql_dumps:
        logger.warning("⚠️ SQL parsers not available, skipping lookup table loading")

    pipeline = IngestionPipeline(config, lookup_tables=lookup_tables)
    pipeline.load_checkpoint(no_resume=no_resume, clear=clear_checkpoint)

    if clear_checkpoint:
        logger.info("Checkpoint cleared. Run again without --clear-checkpoint to ingest.")
        return PipelineStats()

    stats = pipeline.run(limit=limit or config.limit)

    logger.info(stats.summary())

    # Run post-ingestion optimization if requested
    if run_post_optimization:
        logger.info("🔧 Running post-ingestion optimization...")
        try:
            from ingestion.optimizer import run_optimization
            opt_stats = run_optimization(
                neo4j_uri=config.neo4j_uri,
                neo4j_user=config.neo4j_user,
                neo4j_password=config.neo4j_password,
                neo4j_database=config.neo4j_database,
            )
            logger.info(opt_stats.summary())
        except Exception as e:
            logger.error(f"Optimization failed: {e}")

    return stats
