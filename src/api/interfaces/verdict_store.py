"""Per-request DocumentVerdict storage port.

The ``/verify`` route persists each verdict keyed by ``request_id`` so
callers can re-fetch via ``GET /verdict/{request_id}``. Phase 1 uses an
in-memory dict; Phase 4 Task 1.10 optionally swaps in a Redis-backed
implementation for multi-worker persistence.

Retention policy (spec §11):
  - ``raw_text`` is opt-in via the ``retain_text`` flag. Default False;
    the store receives only the text_hash plus the DocumentVerdict. The
    original input text is never retained by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from models.results import DocumentVerdict


@runtime_checkable
class VerdictStore(Protocol):
    """Store verdicts by request_id; retrieve by request_id."""

    async def put(
        self,
        request_id: UUID,
        verdict: DocumentVerdict,
        *,
        text_hash: str,
        text: str | None,
    ) -> None:
        """Persist a verdict. ``text`` is ``None`` when the retention
        middleware did NOT grant raw-text retention for this request."""

    async def get(self, request_id: UUID) -> DocumentVerdict | None: ...
