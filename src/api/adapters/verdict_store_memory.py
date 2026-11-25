"""Phase 1 in-memory VerdictStore implementation.

Single-process dict keyed by request_id. Good enough for Phase 1 local
dev + CI; swap in a Redis-backed adapter when multi-worker persistence
matters (tracked in Task 1.10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from interfaces.verdict_store import VerdictStore
from models.results import DocumentVerdict


@dataclass
class _Entry:
    verdict: DocumentVerdict
    text_hash: str
    text: str | None
    stored_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemoryVerdictStore(VerdictStore):
    """Process-local verdict cache.

    No TTL enforcement in Phase 1 — retention time is bounded by
    worker lifetime. Phase 4 swaps in a durable store with spec §11
    retention (30d for verdicts, 90d for feedback rationale).
    """

    def __init__(self) -> None:
        self._by_request: dict[UUID, _Entry] = {}

    async def put(
        self,
        request_id: UUID,
        verdict: DocumentVerdict,
        *,
        text_hash: str,
        text: str | None,
    ) -> None:
        self._by_request[request_id] = _Entry(
            verdict=verdict,
            text_hash=text_hash,
            text=text,
        )

    async def get(self, request_id: UUID) -> DocumentVerdict | None:
        entry = self._by_request.get(request_id)
        return entry.verdict if entry is not None else None

    # ------------------------------------------------------------------
    # Non-port extras (useful in tests / debugging)
    # ------------------------------------------------------------------

    async def __len__(self) -> int:
        return len(self._by_request)

    def _count(self) -> int:
        """Sync-context size helper for unit tests."""
        return len(self._by_request)
