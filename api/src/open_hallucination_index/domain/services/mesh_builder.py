"""
Knowledge Mesh Builder Service
==============================

Service for building 3D-visualization-ready knowledge graphs from
verification traces, Neo4j relationships, and Qdrant semantic data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from open_hallucination_index.domain.knowledge_track import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeMesh,
    KnowledgeNode,
    MCP_SOURCE_DESCRIPTIONS,
    NodeType,
    SourceReference,
    TraceData,
)

if TYPE_CHECKING:
    from uuid import UUID

    from open_hallucination_index.ports.knowledge_store import (
        GraphKnowledgeStore,
        VectorKnowledgeStore,
    )
    from open_hallucination_index.ports.knowledge_tracker import KnowledgeTracker

logger = logging.getLogger(__name__)


class KnowledgeMeshBuilder:
    """
    Service for building enhanced knowledge meshes.

    Combines trace data from Redis with live queries to Neo4j and Qdrant
    to build rich 3D-visualization-ready knowledge graphs.
    """

    def __init__(
        self,
        trace_store: KnowledgeTracker,
        graph_store: GraphKnowledgeStore | None = None,
        vector_store: VectorKnowledgeStore | None = None,
    ) -> None:
        """
        Initialize the mesh builder.

        Args:
            trace_store: Redis trace storage for cached trace data.
            graph_store: Neo4j for entity relationships.
            vector_store: Qdrant for semantic neighbors.
        """
        self._trace_store = trace_store
        self._graph_store = graph_store
        self._vector_store = vector_store

    async def build_mesh(
        self,
        claim_id: UUID,
        depth: int = 2,
        include_semantic_neighbors: bool = True,
        max_evidence_nodes: int = 20,
        max_entity_nodes: int = 30,
    ) -> KnowledgeMesh | None:
        """
        Build a comprehensive knowledge mesh for a claim.

        Retrieves trace data from Redis and optionally enriches with
        live Neo4j/Qdrant queries for deeper relationship exploration.

        Args:
            claim_id: UUID of the claim to build mesh for.
            depth: Relationship depth (1-5, default 2).
            include_semantic_neighbors: Include Qdrant semantic matches.
            max_evidence_nodes: Maximum evidence nodes to include.
            max_entity_nodes: Maximum entity nodes to include.

        Returns:
            KnowledgeMesh ready for 3D visualization, or None if not found.
        """
        depth = max(1, min(5, depth))

        # Start with basic mesh from trace store
        mesh = await self._trace_store.build_mesh(claim_id, depth)
        if mesh is None:
            return None

        # If we have stores and depth > 2, enhance with additional data
        if depth > 2:
            mesh = await self._enhance_mesh(
                mesh,
                include_semantic_neighbors,
                max_evidence_nodes,
                max_entity_nodes,
            )

        return mesh

    async def _enhance_mesh(
        self,
        mesh: KnowledgeMesh,
        include_semantic: bool,
        max_evidence: int,
        max_entities: int,
    ) -> KnowledgeMesh:
        """Enhance mesh with additional Neo4j/Qdrant data."""
        nodes = list(mesh.nodes)
        edges = list(mesh.edges)

        # Get claim text from central node
        claim_node = next(
            (n for n in nodes if n.id == mesh.center_claim_id), None
        )
        if claim_node is None:
            return mesh

        # Enhanced entity relationships from Neo4j
        if self._graph_store is not None:
            entity_nodes = [n for n in nodes if n.node_type == NodeType.ENTITY]
            for entity in entity_nodes[:10]:
                try:
                    related = await self._graph_store.get_entity_properties(
                        entity.label
                    )
                    if related and len(nodes) < max_entities + len(mesh.nodes):
                        # Add related entities as deeper nodes
                        for prop_name, prop_value in list(related.items())[:5]:
                            if isinstance(prop_value, str) and len(prop_value) < 100:
                                related_node = KnowledgeNode(
                                    node_type=NodeType.ENTITY,
                                    label=f"{prop_name}: {prop_value}",
                                    source="neo4j",
                                    depth_level=4,
                                    weight=0.3,
                                )
                                nodes.append(related_node)
                                edges.append(
                                    KnowledgeEdge(
                                        source_node_id=entity.id,
                                        target_node_id=related_node.id,
                                        edge_type=EdgeType.PART_OF,
                                        weight=0.4,
                                        label=prop_name,
                                    )
                                )
                except Exception as e:
                    logger.debug(f"Failed to get entity properties: {e}")

        return KnowledgeMesh(
            center_claim_id=mesh.center_claim_id,
            nodes=nodes,
            edges=edges,
            depth_levels=mesh.depth_levels,
            total_sources_queried=mesh.total_sources_queried,
            total_evidence_found=mesh.total_evidence_found,
        )

    def build_source_references(
        self,
        trace: TraceData,
    ) -> list[SourceReference]:
        """
        Build source reference list from trace data.

        Creates SourceReference objects for all MCP sources consulted,
        including URLs, descriptions, and contribution status.

        Args:
            trace: Trace data containing MCP call logs.

        Returns:
            List of SourceReference objects for API response.
        """
        references: list[SourceReference] = []
        seen_sources: set[str] = set()

        # Add references from MCP call logs
        for call in trace.mcp_calls:
            source_name = call.get("source", call.get("tool", "unknown"))
            if source_name in seen_sources:
                continue
            seen_sources.add(source_name)

            url = call.get("url") or call.get("source_uri")
            snippet = call.get("content", call.get("response", ""))[:500]
            query_time = call.get("duration_ms", 0)

            references.append(
                SourceReference(
                    mcp_source=source_name,
                    source_description=MCP_SOURCE_DESCRIPTIONS.get(
                        source_name.lower(),
                        f"{source_name} - External knowledge source",
                    ),
                    url=url,
                    evidence_snippet=snippet if snippet else None,
                    confidence=call.get("confidence", 0.5),
                    contributed=source_name in trace.sources_contributed,
                    query_time_ms=query_time,
                )
            )

        # Add all remaining queried sources that weren't in MCP calls
        for source_name in trace.sources_queried:
            if source_name not in seen_sources:
                references.append(
                    SourceReference(
                        mcp_source=source_name,
                        source_description=MCP_SOURCE_DESCRIPTIONS.get(
                            source_name.lower(),
                            f"{source_name} - External knowledge source",
                        ),
                        url=None,
                        evidence_snippet=None,
                        confidence=0.0,
                        contributed=source_name in trace.sources_contributed,
                        query_time_ms=trace.query_times_ms.get(source_name),
                    )
                )

        # Add local sources (neo4j, qdrant, redis)
        local_sources = ["neo4j", "qdrant", "redis"]
        for local in local_sources:
            if local not in seen_sources:
                references.append(
                    SourceReference(
                        mcp_source=local,
                        source_description=MCP_SOURCE_DESCRIPTIONS.get(
                            local, f"{local} - Local knowledge store"
                        ),
                        url=None,
                        evidence_snippet=None,
                        confidence=0.5,
                        contributed=local in trace.sources_contributed,
                        query_time_ms=trace.query_times_ms.get(local),
                    )
                )

        # Sort by contribution status and confidence
        references.sort(key=lambda r: (not r.contributed, -r.confidence))

        return references

    @staticmethod
    def get_all_mcp_sources() -> list[SourceReference]:
        """
        Get list of all available MCP sources with descriptions.

        Returns:
            List of SourceReference for all known MCP sources.
        """
        return [
            SourceReference(
                mcp_source=source_name,
                source_description=description,
                url=None,
                evidence_snippet=None,
                confidence=0.0,
                contributed=False,
                query_time_ms=None,
            )
            for source_name, description in MCP_SOURCE_DESCRIPTIONS.items()
        ]
