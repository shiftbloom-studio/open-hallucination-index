"""L6 Feedback + calibration-set storage port. Spec §8 + §12."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from models.feedback import (
        CalibrationEntry,
        DisputedClaim,
        FeedbackId,
        FeedbackSubmission,
    )


@runtime_checkable
class FeedbackStore(Protocol):
    """Inbound feedback queue + calibration-set storage.

    Phase 1 uses an in-memory implementation; Phase 4 Task 4.1 swaps in
    the Postgres-backed adapter that implements the spec §12 SQL.
    """

    async def submit(self, submission: FeedbackSubmission) -> FeedbackId:
        """Write to ``feedback_pending``. Idempotent on
        ``(claim_id, labeler_id_hash, label)`` — returns the existing
        feedback_id when the tuple already exists.
        """

    async def promote_consensus(self) -> int:
        """Run the 15-min consensus-promotion job.

        Returns the number of new calibration_set rows written. Disputed
        claims (>=3 concordant labelers on more than one label) get
        routed to the disputed_claims_queue instead.
        """

    async def get_calibration_set(
        self,
        partition: str,
    ) -> list[CalibrationEntry]:
        """Read all active calibration entries for a given partition
        (e.g. ``"biomedical:quantitative"``)."""

    async def list_disputed_claims(self) -> list[DisputedClaim]: ...
