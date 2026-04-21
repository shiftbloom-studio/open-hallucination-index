"""Wave 3 Stream C — lifecycle machinery (SIGUSR1 pause + DLQ + progress).

Shared infrastructure consumed by each ingestion pass:

* **SIGUSR1** installs a handler that flips an ``asyncio.Event`` the
  pass loop polls between batches. On the next natural batch boundary
  the loop flushes in-flight work, writes a ``paused`` checkpoint,
  and exits cleanly with code 0. SIGKILL (kill -9) loses the
  in-flight batch but the last committed checkpoint stays valid
  (resume re-writes idempotently).
* **Dead-letter queue** — JSON-lined file at ``/tmp/ohi-ingestion-dlq.jsonl``
  (overridable). Records that fail after ``CORPUS_MAX_RECORD_RETRIES``
  retries land here with full error context. ``--retry-dlq`` reruns
  only the DLQ; successful re-ingestion clears the entry.
* **Progress events** — every ``PROGRESS_EVERY`` records, emits a
  JSON line to stdout with ``{pass, processed, total_estimate,
  rate_per_sec, eta_hours, dlq_size}``. Greppable for external
  monitoring (Fabian's PC during long runs).

Unix-only for SIGUSR1 — on Windows the pause signal silently degrades
to a "poll the checkpoint store for status==paused" check every batch.
The orchestrator runs on Fabian's Windows machine for Stream I but
SIGUSR1 ↔ signal semantic is handled via a file-based flag
(``/tmp/ohi-ingestion.pause``) that cross-platform works; the POSIX
path uses signals, the Windows path checks file existence each batch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Defaults (tunable via env).
_TEMP_DIR = Path(tempfile.gettempdir())
_PAUSE_FLAG_FILE = Path(
    os.environ.get("CORPUS_PAUSE_FLAG_FILE", str(_TEMP_DIR / "ohi-ingestion.pause"))
)
_DLQ_FILE = Path(os.environ.get("CORPUS_DLQ_PATH", str(_TEMP_DIR / "ohi-ingestion-dlq.jsonl")))
_MAX_RECORD_RETRIES = int(os.environ.get("CORPUS_MAX_RECORD_RETRIES", "5"))
_PROGRESS_EVERY = int(os.environ.get("CORPUS_PROGRESS_EVERY", "10000"))


# ---------------------------------------------------------------------------
# Pause controller
# ---------------------------------------------------------------------------


@dataclass
class PauseController:
    """Cross-platform pause mechanism.

    On POSIX, SIGUSR1 flips the event. On any OS, the existence of
    ``_PAUSE_FLAG_FILE`` also flips it (so operators can
    ``touch /tmp/ohi-ingestion.pause`` from any shell).
    """

    event: asyncio.Event
    flag_file: Path = _PAUSE_FLAG_FILE

    def should_pause(self) -> bool:
        if self.event.is_set():
            return True
        if self.flag_file.exists():
            self.event.set()
            return True
        return False

    def clear(self) -> None:
        self.event.clear()
        with contextlib.suppress(FileNotFoundError):
            self.flag_file.unlink()


def install_pause_handler() -> PauseController:
    """Install SIGUSR1 handler (POSIX) + flag-file watcher.

    Returns a :class:`PauseController` the ingestion loop polls
    between batches via :meth:`should_pause`.
    """
    event = asyncio.Event()
    controller = PauseController(event=event)

    def _handler(signum, frame) -> None:  # noqa: ARG001 — standard signal handler sig
        logger.info("SIGUSR1 received — requesting graceful pause at next batch boundary")
        event.set()

    if hasattr(__import__("signal"), "SIGUSR1"):
        import signal  # noqa: PLC0415

        try:
            signal.signal(signal.SIGUSR1, _handler)
            logger.info("SIGUSR1 pause handler installed")
        except ValueError:
            # Raised when the current thread isn't the main thread (e.g.
            # under pytest). Not fatal — the flag-file path still works.
            logger.info("SIGUSR1 handler not installed (non-main thread); flag-file watcher active")
    else:
        logger.info("No SIGUSR1 on this platform; using flag-file watcher only")

    return controller


# ---------------------------------------------------------------------------
# Dead-letter queue
# ---------------------------------------------------------------------------


class DLQWriter:
    """Append-only JSON-lines DLQ writer."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else _DLQ_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        pass_name: str,
        record_id: str,
        error_class: str,
        error_detail: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "pass": pass_name,
            "record_id": record_id,
            "error_class": error_class,
            "error_detail": error_detail[:2000],  # cap detail size
            "attempted_at": _now_iso(),
        }
        if extra:
            payload.update(extra)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def iter_records(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return iter([])
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("DLQ: skipping malformed line: %s", line[:100])

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Emit JSON-lined progress events to stdout every N records."""

    def __init__(
        self, pass_name: str, total_estimate: int | None, every: int = _PROGRESS_EVERY
    ) -> None:
        self.pass_name = pass_name
        self.total_estimate = total_estimate
        self.every = every
        self._t_start = time.monotonic()
        self._processed = 0

    def tick(self, delta: int = 1, dlq_size: int = 0) -> None:
        self._processed += delta
        if self._processed % self.every == 0:
            elapsed = max(time.monotonic() - self._t_start, 1e-6)
            rate = self._processed / elapsed
            eta_hours: float | None = None
            if self.total_estimate and rate > 0:
                remaining = max(0, self.total_estimate - self._processed)
                eta_hours = remaining / rate / 3600.0
            event = {
                "ts": _now_iso(),
                "pass": self.pass_name,
                "processed": self._processed,
                "total_estimate": self.total_estimate,
                "rate_per_sec": round(rate, 2),
                "eta_hours": round(eta_hours, 3) if eta_hours is not None else None,
                "dlq_size": dlq_size,
            }
            sys.stdout.write(json.dumps(event) + "\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Retry helper (used by every pass for per-record write attempts)
# ---------------------------------------------------------------------------


async def retry_record(
    *,
    fn,
    record_id: str,
    max_retries: int = _MAX_RECORD_RETRIES,
    base_delay_s: float = 0.5,
) -> bool:
    """Run ``fn()`` up to ``max_retries`` times with exponential
    backoff. Returns True on success, False if all attempts exhausted.

    Kept as a plain function (not a decorator) so callers can pass
    arbitrary closures capturing the per-record payload.
    """
    for attempt in range(1, max_retries + 1):
        try:
            await fn()
            return True
        except Exception as exc:  # noqa: BLE001
            delay = base_delay_s * (2 ** (attempt - 1))
            logger.warning(
                "Record %s attempt %d/%d failed: %s (sleep %.2fs)",
                record_id,
                attempt,
                max_retries,
                exc,
                delay,
            )
            if attempt < max_retries:
                await asyncio.sleep(delay)
    return False


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


__all__ = [
    "DLQWriter",
    "PauseController",
    "ProgressReporter",
    "install_pause_handler",
    "retry_record",
]
