"""
Checkpoint manager for resumable ingestion.

Features:
- Atomic checkpoint saves
- Processed ID tracking for duplicate prevention
- Session statistics
- Emergency save on crash
- Mid-file progress tracking for recovery within BZ2 files
- Configurable save intervals with force-save capability
"""

from __future__ import annotations

import atexit
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("ingestion.checkpoint")


class CheckpointManager:
    """
    Manages checkpoint state for resumable ingestion.

    Saves progress after each batch so ingestion can be resumed from
    the last successful batch if interrupted.

    IMPORTANT: processed_ids are ALWAYS preserved unless --clear-checkpoint
    is used. This ensures no duplicate articles are ever imported, even
    across multiple runs with different settings.
    
    NEW: Mid-file progress tracking allows recovery within a BZ2 file,
    not just at file boundaries. This prevents having to re-process
    large files from the beginning after a crash.
    """

    DEFAULT_CHECKPOINT_FILE = ".ingest_checkpoint.json"
    VERSION = "4.0"  # Bumped for mid-file tracking support

    def __init__(
        self,
        checkpoint_file: str | None = None,
        collection: str = "wikipedia_hybrid",
        auto_save: bool = True,
        auto_save_interval: int = 50,  # Reduced from 100 for more frequent saves
        force_save_interval: float = 60.0,  # Force save every N seconds regardless of batches
    ):
        self.checkpoint_file = Path(checkpoint_file or self.DEFAULT_CHECKPOINT_FILE)
        self.collection = collection
        self.auto_save = auto_save
        self.auto_save_interval = max(1, auto_save_interval)
        self.force_save_interval = force_save_interval

        # Checkpoint state
        self.processed_ids: set[int] = set()
        self.last_article_id: int = 0
        self.articles_processed: int = 0
        self.chunks_created: int = 0
        self.session_start: float = time.time()
        self.total_elapsed: float = 0.0

        # Parts tracking
        self.processed_parts: set[int] = set()
        
        # NEW: Mid-file progress tracking
        # Maps part_index -> last_processed_article_id within that file
        self.part_progress: dict[int, int] = {}

        # Batch counter for throttled saves
        self._batch_counter: int = 0
        self._last_save_time: float = time.time()
        self._pending_changes: bool = False

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        # Register atexit handler for emergency saves
        atexit.register(self._emergency_save)

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful checkpoint saving on termination."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"🛑 Received signal {signum}, saving checkpoint...")
            self._emergency_save()
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            # Windows doesn't have SIGHUP
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)
        except (ValueError, OSError):
            # Signal handling may not work in all contexts (e.g., threads)
            pass

    def _emergency_save(self) -> None:
        """Save checkpoint on unexpected exit."""
        if self.processed_ids or self._pending_changes:
            try:
                self.save()
                logger.info("💾 Emergency checkpoint saved")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

    def load(self) -> bool:
        """
        Load checkpoint from file if it exists and matches current collection.

        Returns:
            True if checkpoint was loaded, False otherwise.
        """
        if not self.checkpoint_file.exists():
            logger.info("📍 No checkpoint found, starting fresh")
            return False

        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)

            # Verify collection matches
            if data.get("collection") != self.collection:
                logger.warning(
                    f"⚠️  Checkpoint is for different collection "
                    f"('{data.get('collection')}' vs '{self.collection}'). "
                    f"Starting fresh."
                )
                return False

            self.processed_ids = set(data.get("processed_ids", []))
            self.last_article_id = data.get("last_article_id", 0)
            self.articles_processed = data.get("articles_processed", 0)
            self.chunks_created = data.get("chunks_created", 0)
            self.total_elapsed = data.get("total_elapsed", 0.0)
            self.processed_parts = set(data.get("processed_parts", []))
            
            # NEW: Load mid-file progress (convert string keys back to int)
            raw_progress = data.get("part_progress", {})
            self.part_progress = {int(k): v for k, v in raw_progress.items()}

            logger.info(
                f"✅ Checkpoint loaded: {self.articles_processed:,} articles processed, "
                f"{len(self.processed_parts)} parts complete, "
                f"{len(self.part_progress)} parts in progress"
            )
            return True

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}. Starting fresh.")
            return False

    def save(self) -> None:
        """Save current state to checkpoint file."""
        # Calculate total elapsed time including previous sessions
        current_session_time = time.time() - self.session_start
        total_time = self.total_elapsed + current_session_time

        data = {
            "collection": self.collection,
            "processed_ids": list(self.processed_ids),
            "last_article_id": self.last_article_id,
            "articles_processed": self.articles_processed,
            "chunks_created": self.chunks_created,
            "total_elapsed": total_time,
            "processed_parts": list(self.processed_parts),
            # NEW: Mid-file progress tracking
            "part_progress": self.part_progress,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": self.VERSION,
        }

        # Write atomically using temp file
        # Use compact JSON (no indent) for faster serialization with large ID sets
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))  # Compact format, ~2-3x smaller

            # Atomic rename
            temp_file.replace(self.checkpoint_file)
            self._pending_changes = False
            self._last_save_time = time.time()

        except OSError as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def should_skip(self, article_id: int) -> bool:
        """
        Check if article has already been processed.

        This is the PRIMARY mechanism for preventing duplicate imports.
        """
        return article_id in self.processed_ids

    def should_skip_part(self, part_index: int) -> bool:
        """Check if a dump part has already been fully processed."""
        return part_index in self.processed_parts

    def get_part_resume_position(self, part_index: int) -> int | None:
        """
        Get the last processed article ID for a part that was interrupted.
        
        Returns:
            The last article ID processed, or None if starting from beginning.
        """
        return self.part_progress.get(part_index)

    def record_batch(
        self,
        article_ids: list[int],
        chunk_count: int,
        part_index: int | None = None,
    ) -> None:
        """
        Record a successfully processed batch.

        Args:
            article_ids: List of processed article IDs
            chunk_count: Number of chunks created from this batch
            part_index: Optional part index for mid-file progress tracking
        """
        for article_id in article_ids:
            self.processed_ids.add(article_id)
            self.last_article_id = max(self.last_article_id, article_id)

        self.articles_processed += len(article_ids)
        self.chunks_created += chunk_count
        self._pending_changes = True
        
        # NEW: Track mid-file progress
        if part_index is not None and article_ids:
            self.part_progress[part_index] = max(article_ids)

        # Throttled auto-save: save every N batches or every M seconds
        if self.auto_save:
            self._batch_counter += 1
            now = time.time()
            time_since_save = now - self._last_save_time
            
            should_save = (
                self._batch_counter >= self.auto_save_interval or
                time_since_save >= self.force_save_interval
            )
            
            if should_save:
                self.save()
                self._batch_counter = 0

    def force_save(self) -> None:
        """Force an immediate checkpoint save regardless of interval."""
        if self._pending_changes:
            self.save()
            self._batch_counter = 0

    def record_part_complete(self, part_index: int) -> None:
        """Record that a dump part has been fully processed."""
        self.processed_parts.add(part_index)
        # Remove from in-progress tracking since it's complete
        self.part_progress.pop(part_index, None)
        self._pending_changes = True
        if self.auto_save:
            self.save()

    def load_ids_only(self) -> bool:
        """
        Load ONLY the processed_ids from checkpoint, reset all other stats.

        This is used with --no-resume to start fresh statistics but still
        preserve the list of already-processed articles to prevent duplicates.

        Returns:
            True if IDs were loaded, False otherwise.
        """
        if not self.checkpoint_file.exists():
            logger.info("📍 No checkpoint found, starting completely fresh")
            return False

        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)

            # Load ONLY the processed IDs - these are sacred and must persist
            self.processed_ids = set(data.get("processed_ids", []))

            # Reset all other stats to zero (fresh run)
            self.last_article_id = 0
            self.articles_processed = 0
            self.chunks_created = 0
            self.total_elapsed = 0.0
            self.processed_parts = set()
            self.part_progress = {}  # Reset mid-file progress too

            if self.processed_ids:
                logger.info(
                    f"🛡️  Loaded {len(self.processed_ids):,} previously processed "
                    f"article IDs (will be skipped to prevent duplicates)"
                )
                logger.info("📊 Statistics reset to zero for fresh run")
            return True

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}. Starting fresh.")
            return False

    def clear(self) -> None:
        """
        Clear checkpoint file completely (for TRULY fresh start).

        WARNING: This will remove ALL processed article IDs!
        Articles may be re-imported if the databases are not also cleared.
        """
        if self.checkpoint_file.exists():
            # Load current state for logging
            try:
                with open(self.checkpoint_file, encoding="utf-8") as f:
                    data = json.load(f)
                old_count = len(data.get("processed_ids", []))
                logger.warning(
                    f"⚠️  Deleting {old_count:,} processed article IDs - "
                    f"articles may be re-imported!"
                )
            except Exception:
                pass

            self.checkpoint_file.unlink()
            logger.info("🗑️  Checkpoint file deleted")

        self.processed_ids.clear()
        self.last_article_id = 0
        self.articles_processed = 0
        self.chunks_created = 0
        self.total_elapsed = 0.0
        self.processed_parts.clear()
        self.part_progress.clear()
        self._pending_changes = False

        logger.info("✅ Checkpoint completely cleared")

    def get_resume_info(self) -> str:
        """Get human-readable resume information."""
        if not self.processed_ids:
            return "Starting fresh ingestion"

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(self.total_elapsed))
        in_progress = len(self.part_progress)
        info = (
            f"Resuming: {self.articles_processed:,} articles already processed, "
            f"{self.chunks_created:,} chunks created, "
            f"{len(self.processed_parts)} parts complete"
        )
        if in_progress:
            info += f", {in_progress} parts in progress"
        info += f", previous runtime: {elapsed_str}"
        return info

    def sync_from_database(self, article_ids: set[int]) -> int:
        """
        Sync checkpoint with article IDs retrieved from database.
        
        This is used for recovery when checkpoint file is lost but data
        exists in Qdrant/Neo4j. It merges the provided IDs with any
        existing processed_ids.
        
        Args:
            article_ids: Set of article IDs found in the database
            
        Returns:
            Number of new IDs added to the checkpoint
        """
        original_count = len(self.processed_ids)
        self.processed_ids.update(article_ids)
        new_count = len(self.processed_ids) - original_count
        
        if new_count > 0:
            logger.info(
                f"🔄 Synced {new_count:,} article IDs from database "
                f"(total now: {len(self.processed_ids):,})"
            )
            self._pending_changes = True
            self.save()
        
        return new_count

    def recover_from_database(
        self,
        qdrant_fetch_fn: Callable[[], set[int]] | None = None,
        neo4j_fetch_fn: Callable[[], set[int]] | None = None,
    ) -> int:
        """
        Recover checkpoint by querying existing data in databases.
        
        This is the main recovery method when checkpoint is lost. It queries
        Qdrant and/or Neo4j to find which articles have already been ingested.
        
        Args:
            qdrant_fetch_fn: Function to fetch article IDs from Qdrant
            neo4j_fetch_fn: Function to fetch article IDs from Neo4j
            
        Returns:
            Total number of IDs recovered
        """
        recovered_ids: set[int] = set()
        
        if qdrant_fetch_fn:
            try:
                logger.info("🔍 Scanning Qdrant for existing articles...")
                qdrant_ids = qdrant_fetch_fn()
                recovered_ids.update(qdrant_ids)
                logger.info(f"   Found {len(qdrant_ids):,} articles in Qdrant")
            except Exception as e:
                logger.warning(f"⚠️ Failed to scan Qdrant: {e}")
        
        if neo4j_fetch_fn:
            try:
                logger.info("🔍 Scanning Neo4j for existing articles...")
                neo4j_ids = neo4j_fetch_fn()
                recovered_ids.update(neo4j_ids)
                logger.info(f"   Found {len(neo4j_ids):,} articles in Neo4j")
            except Exception as e:
                logger.warning(f"⚠️ Failed to scan Neo4j: {e}")
        
        if recovered_ids:
            return self.sync_from_database(recovered_ids)
        
        return 0
