"""
Checkpoint manager for resumable ingestion.

Features:
- Atomic checkpoint saves
- Processed ID tracking for duplicate prevention
- Session statistics
- Emergency save on crash
"""

from __future__ import annotations

import atexit
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("ingestion.checkpoint")


class CheckpointManager:
    """
    Manages checkpoint state for resumable ingestion.

    Saves progress after each batch so ingestion can be resumed from
    the last successful batch if interrupted.

    IMPORTANT: processed_ids are ALWAYS preserved unless --clear-checkpoint
    is used. This ensures no duplicate articles are ever imported, even
    across multiple runs with different settings.
    """

    DEFAULT_CHECKPOINT_FILE = ".ingest_checkpoint.json"

    def __init__(
        self,
        checkpoint_file: str | None = None,
        collection: str = "wikipedia_hybrid",
        auto_save: bool = True,
        auto_save_interval: int = 100,  # Save every N batches instead of every batch
    ):
        self.checkpoint_file = Path(checkpoint_file or self.DEFAULT_CHECKPOINT_FILE)
        self.collection = collection
        self.auto_save = auto_save
        self.auto_save_interval = max(1, auto_save_interval)  # Minimum 1

        # Checkpoint state
        self.processed_ids: set[int] = set()
        self.last_article_id: int = 0
        self.articles_processed: int = 0
        self.chunks_created: int = 0
        self.session_start: float = time.time()
        self.total_elapsed: float = 0.0

        # Parts tracking
        self.processed_parts: set[int] = set()

        # Batch counter for throttled saves
        self._batch_counter: int = 0

        # Register atexit handler for emergency saves
        atexit.register(self._emergency_save)

    def _emergency_save(self) -> None:
        """Save checkpoint on unexpected exit."""
        if self.processed_ids:
            try:
                self.save()
                logger.info("💾 Emergency checkpoint saved")
            except Exception:
                pass

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

            logger.info(
                f"✅ Checkpoint loaded: {self.articles_processed:,} articles processed, "
                f"{len(self.processed_parts)} parts complete"
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
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "3.0",
        }

        # Write atomically using temp file
        # Use compact JSON (no indent) for faster serialization with large ID sets
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))  # Compact format, ~2-3x smaller

            # Atomic rename
            temp_file.replace(self.checkpoint_file)

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

    def record_batch(
        self,
        article_ids: list[int],
        chunk_count: int,
    ) -> None:
        """
        Record a successfully processed batch.

        Args:
            article_ids: List of processed article IDs
            chunk_count: Number of chunks created from this batch
        """
        for article_id in article_ids:
            self.processed_ids.add(article_id)
            self.last_article_id = max(self.last_article_id, article_id)

        self.articles_processed += len(article_ids)
        self.chunks_created += chunk_count

        # Throttled auto-save: only save every N batches to avoid I/O bottleneck
        # with large processed_ids sets (can be millions of IDs)
        if self.auto_save:
            self._batch_counter += 1
            if self._batch_counter >= self.auto_save_interval:
                self.save()
                self._batch_counter = 0

    def record_part_complete(self, part_index: int) -> None:
        """Record that a dump part has been fully processed."""
        self.processed_parts.add(part_index)
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

        logger.info("✅ Checkpoint completely cleared")

    def get_resume_info(self) -> str:
        """Get human-readable resume information."""
        if not self.processed_ids:
            return "Starting fresh ingestion"

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(self.total_elapsed))
        return (
            f"Resuming: {self.articles_processed:,} articles already processed, "
            f"{self.chunks_created:,} chunks created, "
            f"{len(self.processed_parts)} parts complete, "
            f"previous runtime: {elapsed_str}"
        )
