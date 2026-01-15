"""
KnowledgeTracker Port
=====================

Abstract interface for recording and retrieving knowledge provenance traces.
Traces capture the full decision path during claim verification, including
all MCP sources consulted, evidence found, and relationships discovered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from open_hallucination_index.domain.knowledge_track import (
        KnowledgeMesh,
        TraceData,
    )


class KnowledgeTracker(ABC):
    """
    Port for knowledge provenance tracking.

    Responsibilities:
    - Record verification traces during claim processing
    - Store traces with configurable TTL (default 12 hours)
    - Retrieve traces by claim ID for /knowledge-track endpoint
    - Build 3D mesh structures from stored trace data
    """

    @abstractmethod
    async def record_trace(
        self,
        trace: TraceData,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Record a verification trace for a claim.

        Called after claim verification to store the full provenance
        including all MCP calls, evidence, and source metadata.

        Args:
            trace: Complete trace data from verification.
            ttl_seconds: Time-to-live in seconds (default: 12 hours / 43200s).

        Returns:
            True if trace was successfully stored.
        """
        ...

    @abstractmethod
    async def get_trace(self, claim_id: UUID) -> TraceData | None:
        """
        Retrieve a stored trace by claim ID.

        Args:
            claim_id: UUID of the verified claim.

        Returns:
            Stored trace data or None if not found/expired.
        """
        ...

    @abstractmethod
    async def trace_exists(self, claim_id: UUID) -> bool:
        """
        Check if a trace exists for the given claim ID.

        Args:
            claim_id: UUID of the claim to check.

        Returns:
            True if trace exists and is not expired.
        """
        ...

    @abstractmethod
    async def delete_trace(self, claim_id: UUID) -> bool:
        """
        Delete a stored trace.

        Args:
            claim_id: UUID of the claim trace to delete.

        Returns:
            True if trace was found and deleted.
        """
        ...

    @abstractmethod
    async def get_traces_batch(
        self,
        claim_ids: list[UUID],
    ) -> dict[UUID, TraceData | None]:
        """
        Retrieve multiple traces by claim IDs.

        Args:
            claim_ids: List of claim UUIDs to retrieve.

        Returns:
            Dictionary mapping claim_id to trace data (or None if not found).
        """
        ...

    @abstractmethod
    async def record_traces_batch(
        self,
        traces: list[TraceData],
        ttl_seconds: int | None = None,
    ) -> int:
        """
        Record multiple traces in a batch operation.

        Args:
            traces: List of trace data to store.
            ttl_seconds: TTL for all traces (default: 12 hours).

        Returns:
            Number of traces successfully stored.
        """
        ...

    @abstractmethod
    async def build_mesh(
        self,
        claim_id: UUID,
        depth: int = 2,
    ) -> KnowledgeMesh | None:
        """
        Build a 3D knowledge mesh from stored trace data.

        Constructs a graph structure suitable for 3D visualization,
        including nodes for claims, evidence, sources, and entities,
        with edges representing relationships.

        Args:
            claim_id: UUID of the claim to build mesh for.
            depth: How many relationship hops to traverse (1-5, default 2).

        Returns:
            KnowledgeMesh ready for visualization, or None if trace not found.
        """
        ...


class KnowledgeTrackerError(Exception):
    """Exception raised when knowledge tracking operations fail."""

    pass
