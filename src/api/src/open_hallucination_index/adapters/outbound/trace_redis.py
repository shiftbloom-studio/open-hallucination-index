"""
Redis Trace Storage Adapter
===========================

Adapter for storing knowledge provenance traces in Redis.
Traces are stored with 12-hour TTL for retrieval via /knowledge-track endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import TYPE_CHECKING, Any
from uuid import UUID

import redis.asyncio as redis

from open_hallucination_index.domain.knowledge_track import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeMesh,
    KnowledgeNode,
    NodeType,
    TraceData,
)
from open_hallucination_index.ports.knowledge_tracker import (
    KnowledgeTracker,
    KnowledgeTrackerError,
)

if TYPE_CHECKING:
    from open_hallucination_index.infrastructure.config import RedisSettings

logger = logging.getLogger(__name__)

# Redis key prefixes
TRACE_PREFIX = "ohi:trace:"

# Default TTL: 12 hours in seconds
DEFAULT_TRACE_TTL = 43200


class RedisTraceAdapter(KnowledgeTracker):
    """
    Redis adapter for storing and retrieving verification traces.

    Stores complete trace data including MCP call logs, evidence chains,
    and relationship data for building 3D knowledge meshes.
    """

    def __init__(self, settings: RedisSettings) -> None:
        """
        Initialize the adapter with configuration.

        Args:
            settings: Redis connection settings.
        """
        self._settings = settings
        self._client: redis.Redis | None = None  # type: ignore[type-arg]
        self._default_ttl = DEFAULT_TRACE_TTL

    async def connect(self) -> None:
        """Establish connection to Redis with retries."""
        max_retries = 10
        retry_delay = 1.0
        last_error = None

        for attempt in range(max_retries):
            try:
                password = None
                if self._settings.password:
                    password = self._settings.password.get_secret_value()

                if self._settings.socket_path:
                    pool = redis.ConnectionPool(
                        connection_class=redis.UnixDomainSocketConnection,
                        path=self._settings.socket_path,
                        password=password,
                        db=self._settings.db,
                        max_connections=self._settings.max_connections,
                    )
                    if attempt == 0:
                        logger.info(
                            "Connecting to Redis (traces) via Unix socket: %s",
                            self._settings.socket_path,
                        )
                else:
                    pool = redis.ConnectionPool(
                        host=self._settings.host,
                        port=self._settings.port,
                        password=password,
                        db=self._settings.db,
                        max_connections=self._settings.max_connections,
                    )
                    pool.connection_class = type(
                        "IPv4Connection",
                        (pool.connection_class,),
                        {"socket_type": socket.AF_INET},
                    )
                    if attempt == 0:
                        logger.info(
                            "Connecting to Redis (traces) via TCP: %s:%s",
                            self._settings.host,
                            self._settings.port,
                        )

                self._client = redis.Redis(
                    connection_pool=pool,
                    decode_responses=False,
                )

                await self._client.ping()  # type: ignore[misc]

                if self._settings.socket_path:
                    logger.info(
                        "Connected to Redis (traces) via Unix socket: %s",
                        self._settings.socket_path,
                    )
                else:
                    logger.info(
                        "Connected to Redis (traces) at %s:%s",
                        self._settings.host,
                        self._settings.port,
                    )
                return

            except (redis.ConnectionError, FileNotFoundError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "Redis trace connection attempt %s failed: %s. Retrying...",
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(retry_delay)
                continue
            except Exception as e:
                logger.error(f"Unexpected Redis error: {e}")
                raise KnowledgeTrackerError(f"Connection failed: {e}") from e

        msg = f"Connection failed after {max_retries} attempts: {last_error}"
        logger.error(f"Redis trace connection failed: {msg}")
        raise KnowledgeTrackerError(msg)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis (traces)")

    async def health_check(self) -> bool:
        """Check if Redis is reachable."""
        if self._client is None:
            return False
        try:
            await self._client.ping()  # type: ignore[misc]
            return True
        except Exception as e:
            logger.warning(f"Redis trace health check failed: {e}")
            return False

    def _make_key(self, claim_id: UUID) -> str:
        """Create Redis key for trace data."""
        return f"{TRACE_PREFIX}{claim_id}"

    def _serialize_trace(self, trace: TraceData) -> bytes:
        """Serialize trace data to JSON bytes."""
        data = trace.model_dump(mode="json")
        return json.dumps(data).encode("utf-8")

    def _deserialize_trace(self, data: bytes) -> TraceData:
        """Deserialize trace data from JSON bytes."""
        json_data = json.loads(data)
        return TraceData.model_validate(json_data)

    async def record_trace(
        self,
        trace: TraceData,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Record a verification trace for a claim.

        Args:
            trace: Complete trace data from verification.
            ttl_seconds: TTL in seconds (default: 12 hours).

        Returns:
            True if trace was stored successfully.
        """
        if self._client is None:
            logger.warning("Redis client not connected for trace storage")
            return False

        try:
            key = self._make_key(trace.claim_id)
            ttl = ttl_seconds or self._default_ttl
            data = self._serialize_trace(trace)

            await self._client.setex(key, ttl, data)
            logger.debug(f"Recorded trace for claim {trace.claim_id} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Failed to record trace: {e}")
            return False

    async def get_trace(self, claim_id: UUID) -> TraceData | None:
        """
        Retrieve a stored trace by claim ID.

        Args:
            claim_id: UUID of the verified claim.

        Returns:
            Stored trace data or None if not found/expired.
        """
        if self._client is None:
            logger.warning("Redis client not connected")
            return None

        try:
            key = self._make_key(claim_id)
            data = await self._client.get(key)

            if data is None:
                return None

            return self._deserialize_trace(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize trace: {e}")
            await self.delete_trace(claim_id)
            return None
        except Exception as e:
            logger.error(f"Failed to get trace: {e}")
            return None

    async def trace_exists(self, claim_id: UUID) -> bool:
        """Check if a trace exists for the given claim ID."""
        if self._client is None:
            return False

        try:
            key = self._make_key(claim_id)
            return await self._client.exists(key) > 0  # type: ignore[misc]
        except Exception as e:
            logger.error(f"Failed to check trace existence: {e}")
            return False

    async def delete_trace(self, claim_id: UUID) -> bool:
        """Delete a stored trace."""
        if self._client is None:
            return False

        try:
            key = self._make_key(claim_id)
            deleted = await self._client.delete(key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete trace: {e}")
            return False

    async def get_traces_batch(
        self,
        claim_ids: list[UUID],
    ) -> dict[UUID, TraceData | None]:
        """Retrieve multiple traces by claim IDs."""
        if self._client is None or not claim_ids:
            return {claim_id: None for claim_id in claim_ids}

        try:
            keys = [self._make_key(cid) for cid in claim_ids]
            results = await self._client.mget(keys)

            output: dict[UUID, TraceData | None] = {}
            for claim_id, data in zip(claim_ids, results, strict=True):
                if data is not None:
                    try:
                        output[claim_id] = self._deserialize_trace(data)
                    except Exception:
                        output[claim_id] = None
                else:
                    output[claim_id] = None

            return output

        except Exception as e:
            logger.error(f"Failed to get traces batch: {e}")
            return {claim_id: None for claim_id in claim_ids}

    async def record_traces_batch(
        self,
        traces: list[TraceData],
        ttl_seconds: int | None = None,
    ) -> int:
        """Record multiple traces in a batch operation."""
        if self._client is None or not traces:
            return 0

        try:
            ttl = ttl_seconds or self._default_ttl
            pipe = self._client.pipeline()

            for trace in traces:
                key = self._make_key(trace.claim_id)
                data = self._serialize_trace(trace)
                pipe.setex(key, ttl, data)

            await pipe.execute()
            logger.debug(f"Recorded {len(traces)} traces in batch")
            return len(traces)

        except Exception as e:
            logger.error(f"Failed to record traces batch: {e}")
            return 0

    async def build_mesh(
        self,
        claim_id: UUID,
        depth: int = 2,
    ) -> KnowledgeMesh | None:
        """
        Build a 3D knowledge mesh from stored trace data.

        Constructs a graph structure suitable for 3D visualization.

        Args:
            claim_id: UUID of the claim to build mesh for.
            depth: Depth of the mesh to build.

        Returns:
            KnowledgeMesh or None if trace not found.
        """
        # Clamp depth to valid range
        depth = max(1, min(5, depth))

        trace = await self.get_trace(claim_id)
        if trace is None:
            return None

        nodes: list[KnowledgeNode] = []
        edges: list[KnowledgeEdge] = []
        node_ids: dict[str, UUID] = {}

        # Create central claim node
        claim_node = KnowledgeNode(
            id=claim_id,
            node_type=NodeType.CLAIM,
            label=self._truncate_label(trace.claim_text),
            source="claim",
            depth_level=0,
            weight=1.0,
            confidence=trace.verification_confidence,
            metadata={
                "status": trace.verification_status,
                "strategy": trace.verification_strategy,
            },
        )
        nodes.append(claim_node)
        node_ids["claim"] = claim_id

        # Add source nodes for all queried sources
        for source_name in trace.sources_queried:
            source_node = KnowledgeNode(
                node_type=NodeType.SOURCE,
                label=source_name.upper(),
                source=source_name,
                depth_level=1,
                weight=0.8 if source_name in trace.sources_contributed else 0.4,
                metadata={
                    "contributed": source_name in trace.sources_contributed,
                    "query_time_ms": trace.query_times_ms.get(source_name, 0),
                },
            )
            nodes.append(source_node)
            node_ids[f"source:{source_name}"] = source_node.id

            # Edge from source to claim
            edge_type = (
                EdgeType.VERIFIED_BY
                if source_name in trace.sources_contributed
                else EdgeType.RELATED_TO
            )
            edges.append(
                KnowledgeEdge(
                    source_node_id=source_node.id,
                    target_node_id=claim_id,
                    edge_type=edge_type,
                    weight=0.8 if source_name in trace.sources_contributed else 0.3,
                )
            )

        # Add evidence nodes
        self._add_evidence_nodes(
            trace.supporting_evidence,
            EdgeType.SUPPORTS,
            claim_id,
            nodes,
            edges,
            node_ids,
            depth,
        )
        self._add_evidence_nodes(
            trace.refuting_evidence,
            EdgeType.REFUTES,
            claim_id,
            nodes,
            edges,
            node_ids,
            depth,
        )

        # Add entity relationship nodes if depth > 1
        if depth > 1 and trace.entity_relationships:
            self._add_entity_nodes(
                trace.entity_relationships,
                nodes,
                edges,
                node_ids,
                depth,
            )

        # Add semantic neighbor nodes if depth > 1
        if depth > 1 and trace.semantic_neighbors:
            self._add_semantic_nodes(
                trace.semantic_neighbors,
                claim_id,
                nodes,
                edges,
                node_ids,
                depth,
            )

        return KnowledgeMesh(
            center_claim_id=claim_id,
            nodes=nodes,
            edges=edges,
            depth_levels=depth,
            total_sources_queried=len(trace.sources_queried),
            total_evidence_found=(len(trace.supporting_evidence) + len(trace.refuting_evidence)),
        )

    def _truncate_label(self, text: str, max_len: int = 50) -> str:
        """Truncate text for node labels."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    def _add_evidence_nodes(
        self,
        evidence_list: list[dict[str, Any]],
        edge_type: EdgeType,
        claim_id: UUID,
        nodes: list[KnowledgeNode],
        edges: list[KnowledgeEdge],
        node_ids: dict[str, UUID],
        depth: int,
    ) -> None:
        """Add evidence nodes and edges to the mesh."""
        for ev in evidence_list:
            content = ev.get("content", "")
            source = ev.get("source", "unknown")
            source_uri = ev.get("source_uri")
            ev_id = ev.get("id", str(len(nodes)))

            similarity = ev.get("similarity_score") or 0.7
            node = KnowledgeNode(
                node_type=NodeType.EVIDENCE,
                label=self._truncate_label(content),
                source=source,
                source_uri=source_uri,
                depth_level=2,
                weight=similarity,
                confidence=similarity,
                metadata={
                    "match_type": ev.get("match_type"),
                    "source_id": ev.get("source_id"),
                },
            )
            nodes.append(node)
            node_ids[f"evidence:{ev_id}"] = node.id

            # Edge from evidence to claim
            edges.append(
                KnowledgeEdge(
                    source_node_id=node.id,
                    target_node_id=claim_id,
                    edge_type=edge_type,
                    weight=similarity,
                    label=edge_type.value,
                )
            )

            # Connect evidence to its source node if exists
            source_key = f"source:{source}"
            if source_key in node_ids:
                edges.append(
                    KnowledgeEdge(
                        source_node_id=node_ids[source_key],
                        target_node_id=node.id,
                        edge_type=EdgeType.DERIVED_FROM,
                        weight=0.6,
                    )
                )

    def _add_entity_nodes(
        self,
        relationships: list[dict[str, Any]],
        nodes: list[KnowledgeNode],
        edges: list[KnowledgeEdge],
        node_ids: dict[str, UUID],
        depth: int,
    ) -> None:
        """Add entity relationship nodes from Neo4j data."""
        for rel in relationships[:20]:  # Limit to avoid overwhelming mesh
            source_entity = rel.get("source", "")
            target_entity = rel.get("target", "")
            relationship = rel.get("relationship", "RELATED_TO")

            # Add source entity if not exists
            source_key = f"entity:{source_entity}"
            if source_key not in node_ids and source_entity:
                entity_node = KnowledgeNode(
                    node_type=NodeType.ENTITY,
                    label=source_entity,
                    source="neo4j",
                    depth_level=min(3, depth),
                    weight=0.5,
                )
                nodes.append(entity_node)
                node_ids[source_key] = entity_node.id

            # Add target entity if not exists
            target_key = f"entity:{target_entity}"
            if target_key not in node_ids and target_entity:
                entity_node = KnowledgeNode(
                    node_type=NodeType.ENTITY,
                    label=target_entity,
                    source="neo4j",
                    depth_level=min(3, depth),
                    weight=0.5,
                )
                nodes.append(entity_node)
                node_ids[target_key] = entity_node.id

            # Add edge between entities
            if source_key in node_ids and target_key in node_ids:
                edges.append(
                    KnowledgeEdge(
                        source_node_id=node_ids[source_key],
                        target_node_id=node_ids[target_key],
                        edge_type=EdgeType.RELATED_TO,
                        weight=0.5,
                        label=relationship,
                    )
                )

    def _add_semantic_nodes(
        self,
        neighbors: list[dict[str, Any]],
        claim_id: UUID,
        nodes: list[KnowledgeNode],
        edges: list[KnowledgeEdge],
        node_ids: dict[str, UUID],
        depth: int,
    ) -> None:
        """Add semantic neighbor nodes from Qdrant data."""
        for neighbor in neighbors[:15]:  # Limit neighbors
            content = neighbor.get("content", "")
            score = neighbor.get("similarity_score", 0.5)
            source = neighbor.get("source", "qdrant")

            node = KnowledgeNode(
                node_type=NodeType.FACT,
                label=self._truncate_label(content),
                source=source,
                depth_level=min(3, depth),
                weight=score,
                confidence=score,
                metadata={"semantic_similarity": score},
            )
            nodes.append(node)

            # Connect to claim with semantic relationship
            edges.append(
                KnowledgeEdge(
                    source_node_id=node.id,
                    target_node_id=claim_id,
                    edge_type=EdgeType.RELATED_TO,
                    weight=score,
                    label="semantically similar",
                )
            )
