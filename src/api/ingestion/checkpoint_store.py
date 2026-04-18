"""Wave 3 Stream C — SQLite checkpoint store (Decision C).

Fine-grained resume points for the multi-hour corpus-ingestion run.
One row per ``(pass, last_committed_id)`` tracks the high-water mark
of successfully-committed records per pass, plus run-level state
(``running`` / ``paused`` / ``aborted`` / ``complete``).

Upsert semantics throughout — re-running the orchestrator picks up at
``last_committed_id + 1`` for each non-terminal pass without losing
data if a batch was partially written (the resume path verifies and
re-writes idempotently on mismatch).

File location defaults to ``/tmp/ohi-ingestion-state.db``;
overridable via the ``CORPUS_CHECKPOINT_PATH`` env var or the
``path`` constructor kwarg. SQLite chosen over a file-based JSON
journal because (a) crash-mid-write is atomic, (b) multi-process
orchestrator scenarios (Pass 1 in background while Pass 2 primes)
get serialisation for free.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

PassName = Literal["pass1", "pass1b", "pass2", "pass3"]
PassStatus = Literal["running", "paused", "aborted", "complete"]


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
    pass                TEXT PRIMARY KEY,
    last_committed_id   TEXT,
    started_at          TEXT,
    updated_at          TEXT,
    status              TEXT NOT NULL DEFAULT 'running',
    records_processed   INTEGER NOT NULL DEFAULT 0,
    dlq_size            INTEGER NOT NULL DEFAULT 0,
    notes               TEXT
);
"""


@dataclass(frozen=True)
class CheckpointRow:
    """Frozen projection of one checkpoint row."""

    pass_name: PassName
    last_committed_id: str | None
    started_at: str | None
    updated_at: str | None
    status: PassStatus
    records_processed: int
    dlq_size: int
    notes: str | None


class CheckpointStore:
    """Thin SQLite wrapper around ``ingestion_checkpoints``.

    Every method opens a short-lived connection so the store is safe
    to use from multiple threads / processes without a long-running
    connection pool. SQLite's default WAL mode handles concurrent
    readers; writers serialise via the single BEGIN IMMEDIATE lock.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get(
            "CORPUS_CHECKPOINT_PATH", "/tmp/ohi-ingestion-state.db"
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)
            # WAL mode for concurrent reads during long writes.
            conn.execute("PRAGMA journal_mode=WAL;")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, pass_name: PassName) -> CheckpointRow | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pass, last_committed_id, started_at, updated_at, status, "
                "records_processed, dlq_size, notes FROM ingestion_checkpoints "
                "WHERE pass = ?",
                (pass_name,),
            ).fetchone()
            if row is None:
                return None
            return CheckpointRow(
                pass_name=row[0],  # type: ignore[arg-type]
                last_committed_id=row[1],
                started_at=row[2],
                updated_at=row[3],
                status=row[4],  # type: ignore[arg-type]
                records_processed=int(row[5]),
                dlq_size=int(row[6]),
                notes=row[7],
            )

    def all_rows(self) -> list[CheckpointRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pass, last_committed_id, started_at, updated_at, status, "
                "records_processed, dlq_size, notes FROM ingestion_checkpoints"
            ).fetchall()
        return [
            CheckpointRow(
                pass_name=r[0],  # type: ignore[arg-type]
                last_committed_id=r[1],
                started_at=r[2],
                updated_at=r[3],
                status=r[4],  # type: ignore[arg-type]
                records_processed=int(r[5]),
                dlq_size=int(r[6]),
                notes=r[7],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def start_pass(self, pass_name: PassName) -> None:
        """Mark a pass as running. Idempotent — if a row already exists
        in ``paused`` state, flip to ``running`` without resetting
        counters."""
        now = _now_iso()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT status FROM ingestion_checkpoints WHERE pass = ?",
                (pass_name,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO ingestion_checkpoints "
                    "(pass, started_at, updated_at, status, records_processed, dlq_size) "
                    "VALUES (?, ?, ?, 'running', 0, 0)",
                    (pass_name, now, now),
                )
            else:
                conn.execute(
                    "UPDATE ingestion_checkpoints SET status='running', updated_at=? "
                    "WHERE pass = ?",
                    (now, pass_name),
                )

    def advance(
        self,
        pass_name: PassName,
        *,
        last_committed_id: str,
        records_in_batch: int,
    ) -> None:
        """Record a batch commit. Updates ``last_committed_id`` and
        increments ``records_processed``."""
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "UPDATE ingestion_checkpoints SET last_committed_id=?, updated_at=?, "
                "records_processed = records_processed + ? "
                "WHERE pass = ?",
                (last_committed_id, now, records_in_batch, pass_name),
            )

    def increment_dlq(self, pass_name: PassName, n: int = 1) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE ingestion_checkpoints SET dlq_size = dlq_size + ?, updated_at=? "
                "WHERE pass = ?",
                (n, _now_iso(), pass_name),
            )

    def set_status(
        self, pass_name: PassName, status: PassStatus, notes: str | None = None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE ingestion_checkpoints SET status=?, updated_at=?, notes=? "
                "WHERE pass = ?",
                (status, _now_iso(), notes, pass_name),
            )

    def clear(self, pass_name: PassName | None = None) -> None:
        """Drop checkpoint row(s). ``None`` clears all (used by
        ``--force-restart``)."""
        with self._conn() as conn:
            if pass_name is None:
                conn.execute("DELETE FROM ingestion_checkpoints")
            else:
                conn.execute(
                    "DELETE FROM ingestion_checkpoints WHERE pass = ?",
                    (pass_name,),
                )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


__all__ = ["CheckpointStore", "CheckpointRow", "PassName", "PassStatus"]
