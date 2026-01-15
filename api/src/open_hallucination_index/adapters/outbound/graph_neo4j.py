"""
Neo4j Graph Knowledge Store Adapter
===================================

Adapter for Neo4j graph database as a knowledge store.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from open_hallucination_index.domain.entities import Evidence, EvidenceSource
from open_hallucination_index.ports.knowledge_store import GraphKnowledgeStore

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from open_hallucination_index.domain.entities import Claim
    from open_hallucination_index.infrastructure.config import Neo4jSettings

logger = logging.getLogger(__name__)


class Neo4jError(Exception):
    """Exception raised when Neo4j operations fail."""

    pass


class Neo4jGraphAdapter(GraphKnowledgeStore):
    """
    Adapter for Neo4j as a graph-based knowledge store.

    Provides exact and inferred fact lookup via Cypher queries.
    """

    def __init__(self, settings: Neo4jSettings) -> None:
        """
        Initialize the adapter with configuration.

        Args:
            settings: Neo4j connection settings.
        """
        self._settings = settings
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.uri,
                auth=(
                    self._settings.username,
                    self._settings.password.get_secret_value(),
                ),
                max_connection_pool_size=self._settings.max_connection_pool_size,
            )

            # Neo4j can take a while to accept Bolt connections after container start.
            # With large Wikipedia imports, startup can take several minutes.
            max_attempts = 60  # Increased for large databases
            retry_interval = 30.0  # Fixed 30-second interval for less log spam
            for attempt in range(1, max_attempts + 1):
                try:
                    await self._driver.verify_connectivity()
                    logger.info(f"Connected to Neo4j at {self._settings.uri}")
                    return
                except ServiceUnavailable:
                    if attempt >= max_attempts:
                        raise
                    # Only log every 2 attempts to reduce noise
                    if attempt == 1 or attempt % 2 == 0:
                        logger.info(
                            "Neo4j not ready yet (attempt %s/%s). Retrying in %.0fs...",
                            attempt,
                            max_attempts,
                            retry_interval,
                        )
                    await asyncio.sleep(retry_interval)
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise Neo4jError(f"Authentication failed: {e}") from e
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise Neo4jError(f"Service unavailable: {e}") from e

    async def disconnect(self) -> None:
        """Close the Neo4j connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    async def health_check(self) -> bool:
        """Check if Neo4j is reachable."""
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")
            return False

    async def query_triplet(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[Evidence]:
        """
        Query for matching triplets in the graph.

        Uses flexible Cypher matching based on provided parameters.
        """
        if self._driver is None:
            raise Neo4jError("Not connected to Neo4j")

        if not any([subject, predicate, obj]):
            raise ValueError("At least one of subject, predicate, or obj must be provided")

        conditions = []
        params: dict[str, Any] = {}

        if subject:
            conditions.append("toLower(s.name) CONTAINS toLower($subject)")
            params["subject"] = subject

        if obj:
            conditions.append("toLower(o.name) CONTAINS toLower($obj)")
            params["obj"] = obj

        where_clause = " AND ".join(conditions) if conditions else "true"

        if predicate:
            query = f"""
                MATCH (s)-[r]->(o)
                WHERE type(r) = $predicate AND {where_clause}
                RETURN s.name AS subject, type(r) AS predicate, o.name AS object,
                       properties(s) AS s_props, properties(o) AS o_props
                LIMIT 10
            """
            params["predicate"] = predicate
        else:
            query = f"""
                MATCH (s)-[r]->(o)
                WHERE {where_clause}
                RETURN s.name AS subject, type(r) AS predicate, o.name AS object,
                       properties(s) AS s_props, properties(o) AS o_props
                LIMIT 10
            """

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, params)
                records = await result.data()

                return [
                    Evidence(
                        id=uuid4(),
                        source=EvidenceSource.GRAPH_EXACT,
                        content=f"{r['subject']} {r['predicate']} {r['object']}",
                        structured_data={
                            "subject": r["subject"],
                            "predicate": r["predicate"],
                            "object": r["object"],
                            "subject_properties": r.get("s_props", {}),
                            "object_properties": r.get("o_props", {}),
                        },
                        match_type="exact",
                        retrieved_at=datetime.now(UTC),
                    )
                    for r in records
                ]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            raise Neo4jError(f"Query failed: {e}") from e

    async def find_evidence_for_claim(
        self,
        claim: Claim,
        max_hops: int = 2,
    ) -> list[Evidence]:
        """
        Find graph evidence supporting or refuting a claim.

        Performs multi-hop traversal to find direct and indirect evidence.
        """
        if self._driver is None:
            raise Neo4jError("Not connected to Neo4j")

        evidence: list[Evidence] = []

        # Try exact triplet match if claim has structured form
        if claim.subject and claim.object:
            direct = await self.query_triplet(
                subject=claim.subject,
                predicate=claim.predicate,
                obj=claim.object,
            )
            evidence.extend(direct)

        # If no direct match, try entity-based search
        if not evidence:
            await self._entity_search(claim, evidence)

        # Multi-hop path search if subject and object are known
        if claim.subject and claim.object and max_hops > 1:
            path_evidence = await self._find_paths(claim.subject, claim.object, max_hops)
            evidence.extend(path_evidence)

        return evidence

    async def _entity_search(self, claim: Claim, evidence: list[Evidence]) -> None:
        """Entity search based on capitalized words in claim."""
        if self._driver is None:
            return

        words = claim.text.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 1]

        if not capitalized:
            return

        query = """
            MATCH (n)
            WHERE any(word IN $words WHERE toLower(n.name) CONTAINS toLower(word))
            WITH n
            LIMIT 5
            MATCH (n)-[r]-(m)
            RETURN n.name AS entity, type(r) AS relation, m.name AS related
            LIMIT 10
        """

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, {"words": capitalized})
                records = await result.data()

                for r in records:
                    evidence.append(
                        Evidence(
                            id=uuid4(),
                            source=EvidenceSource.GRAPH_INFERRED,
                            content=f"{r['entity']} {r['relation']} {r['related']}",
                            structured_data=r,
                            match_type="entity_search",
                            retrieved_at=datetime.now(UTC),
                        )
                    )
        except Exception as e:
            logger.debug(f"Entity search failed: {e}")

    async def _find_paths(self, subject: str, obj: str, max_hops: int) -> list[Evidence]:
        """Find paths between two entities up to max_hops."""
        if self._driver is None:
            return []

        query = f"""
            MATCH path = shortestPath(
                (s)-[*1..{max_hops}]-(o)
            )
            WHERE toLower(s.name) CONTAINS toLower($subject)
              AND toLower(o.name) CONTAINS toLower($obj)
            RETURN [n IN nodes(path) | n.name] AS nodes,
                   [r IN relationships(path) | type(r)] AS relations
            LIMIT 3
        """

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, {"subject": subject, "obj": obj})  # type: ignore[arg-type]
                records = await result.data()

                evidence = []
                for rec in records:
                    nodes = rec.get("nodes", [])
                    relations = rec.get("relations", [])
                    path_str = " -> ".join(
                        f"{nodes[i]} [{relations[i]}]" if i < len(relations) else nodes[i]
                        for i in range(len(nodes))
                    )
                    evidence.append(
                        Evidence(
                            id=uuid4(),
                            source=EvidenceSource.GRAPH_INFERRED,
                            content=path_str,
                            structured_data=rec,
                            match_type="path",
                            retrieved_at=datetime.now(UTC),
                        )
                    )
                return evidence
        except Exception as e:
            logger.debug(f"Path search failed: {e}")
            return []

    async def entity_exists(self, entity: str) -> bool:
        """Check if an entity exists in the knowledge graph."""
        if self._driver is None:
            raise Neo4jError("Not connected to Neo4j")

        query = """
            MATCH (n)
            WHERE toLower(n.name) = toLower($name)
            RETURN count(n) > 0 AS exists
        """

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, {"name": entity})
                record = await result.single()
                return record["exists"] if record else False
        except Exception as e:
            logger.error(f"Entity existence check failed: {e}")
            return False

    async def get_entity_properties(
        self,
        entity: str,
    ) -> dict[str, Any] | None:
        """Retrieve all properties of an entity."""
        if self._driver is None:
            raise Neo4jError("Not connected to Neo4j")

        query = """
            MATCH (n)
            WHERE toLower(n.name) = toLower($name)
            RETURN properties(n) AS props, labels(n) AS labels
            LIMIT 1
        """

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, {"name": entity})
                record = await result.single()
                if record:
                    return {
                        "properties": record["props"],
                        "labels": record["labels"],
                    }
                return None
        except Exception as e:
            logger.error(f"Entity properties retrieval failed: {e}")
            return None

    async def persist_external_evidence(self, evidence: Evidence) -> bool:
        """
        Persist evidence from external sources (MCP) to the graph.

        Creates an ExternalEvidence node with source metadata for future lookups.
        Deduplicates based on content hash.

        Args:
            evidence: Evidence object from MCP source.

        Returns:
            True if persisted successfully, False otherwise.
        """
        if self._driver is None:
            return False

        # Create content hash for deduplication
        import hashlib

        content_hash = hashlib.sha256(evidence.content.encode()).hexdigest()[:32]

        # Determine source type from EvidenceSource enum
        source_type = evidence.source.value if evidence.source else "unknown"

        query = """
            MERGE (e:ExternalEvidence {content_hash: $content_hash})
            ON CREATE SET
                e.content = $content,
                e.source_type = $source_type,
                e.source_id = $source_id,
                e.source_uri = $source_uri,
                e.created_at = datetime(),
                e.similarity_score = $similarity_score,
                e.match_type = $match_type
            ON MATCH SET
                e.last_accessed = datetime(),
                e.access_count = COALESCE(e.access_count, 0) + 1
            RETURN e.content_hash AS hash, e.created_at IS NOT NULL AS was_created
        """

        params = {
            "content_hash": content_hash,
            "content": evidence.content[:5000],  # Limit stored content
            "source_type": source_type,
            "source_id": evidence.source_id,
            "source_uri": evidence.source_uri,
            "similarity_score": evidence.similarity_score,
            "match_type": evidence.match_type,
        }

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, params)
                record = await result.single()
                if record:
                    logger.debug(f"Persisted external evidence: {content_hash[:8]}...")
                    return True
                return False
        except Exception as e:
            logger.warning(f"Failed to persist external evidence: {e}")
            return False

    async def find_persisted_evidence(self, query_text: str, limit: int = 5) -> list[Evidence]:
        """
        Find previously persisted external evidence matching a query.

        Args:
            query_text: Text to search for in persisted evidence.
            limit: Maximum results to return.

        Returns:
            List of matching Evidence objects.
        """
        if self._driver is None:
            return []

        # Simple full-text search on content
        words = [w.lower() for w in query_text.split()[:5] if len(w) > 3]
        if not words:
            return []

        query = """
            MATCH (e:ExternalEvidence)
            WHERE any(word IN $words WHERE toLower(e.content) CONTAINS word)
            RETURN e
            ORDER BY e.similarity_score DESC
            LIMIT $limit
        """

        evidences: list[Evidence] = []
        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, {"words": words, "limit": limit})
                async for record in result:
                    node = record["e"]
                    source_str = node.get("source_type", "cached")
                    try:
                        source = EvidenceSource(source_str)
                    except ValueError:
                        source = EvidenceSource.CACHED

                    evidences.append(
                        Evidence(
                            id=uuid4(),
                            source=source,
                            source_id=node.get("source_id"),
                            content=node.get("content", ""),
                            similarity_score=node.get("similarity_score"),
                            match_type="persisted",
                            retrieved_at=datetime.now(UTC),
                            source_uri=node.get("source_uri"),
                        )
                    )
        except Exception as e:
            logger.warning(f"Persisted evidence search failed: {e}")

        return evidences

    async def count_evidence_for_claim(
        self,
        claim: Claim,
    ) -> int:
        """
        Fast count of evidence without retrieving full content.

        Useful for sufficiency checks before committing to full retrieval.

        Args:
            claim: The claim to check evidence for.

        Returns:
            Approximate count of matching evidence.
        """
        if self._driver is None:
            return 0

        count = 0

        # Count direct triplet matches if claim has structured form
        if claim.subject and claim.object:
            count += await self._count_triplet_matches(claim.subject, claim.object)

        # Count entity matches
        count += await self._count_entity_matches(claim.text)

        return count

    async def _count_triplet_matches(self, subject: str, obj: str) -> int:
        """Count triplet matches without retrieving data."""
        if self._driver is None:
            return 0

        query = """
            MATCH (s)-[r]->(o)
            WHERE toLower(s.name) CONTAINS toLower($subject)
              AND toLower(o.name) CONTAINS toLower($obj)
            RETURN count(*) AS cnt
        """

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, {"subject": subject, "obj": obj})
                record = await result.single()
                return record["cnt"] if record else 0
        except Exception:
            return 0

    async def _count_entity_matches(self, text: str) -> int:
        """Count entity matches in text."""
        if self._driver is None:
            return 0

        # Extract capitalized words as potential entities
        words = text.split()
        entities = [w for w in words if w and w[0].isupper() and len(w) > 1]

        if not entities:
            return 0

        query = """
            MATCH (n)-[r]-(m)
            WHERE any(word IN $words WHERE toLower(n.name) CONTAINS toLower(word))
            RETURN count(*) AS cnt
        """

        try:
            async with self._driver.session(database=self._settings.database) as session:
                result = await session.run(query, {"words": entities[:5]})
                record = await result.single()
                return min(record["cnt"], 50) if record else 0  # Cap for performance
        except Exception:
            return 0

    async def ensure_indexes(self) -> None:
        """
        Ensure optimal indexes exist for common query patterns.

        Creates indexes for:
        - Node name lookups (case-insensitive)
        - ExternalEvidence content hash
        - ExternalEvidence source type
        """
        if self._driver is None:
            return

        indexes = [
            # Full-text index on common node name property
            """
            CREATE TEXT INDEX node_name_text IF NOT EXISTS
            FOR (n:Entity)
            ON (n.name)
            """,
            # Index for external evidence deduplication
            """
            CREATE INDEX external_evidence_hash IF NOT EXISTS
            FOR (e:ExternalEvidence)
            ON (e.content_hash)
            """,
            # Index for source-based lookups
            """
            CREATE INDEX external_evidence_source IF NOT EXISTS
            FOR (e:ExternalEvidence)
            ON (e.source_type)
            """,
            # Composite index for similarity filtering
            """
            CREATE INDEX external_evidence_similarity IF NOT EXISTS
            FOR (e:ExternalEvidence)
            ON (e.similarity_score)
            """,
        ]

        for index_query in indexes:
            try:
                async with self._driver.session(database=self._settings.database) as session:
                    await session.run(index_query)
            except Exception as e:
                # Index might already exist or syntax differs by Neo4j version
                logger.debug(f"Index creation note: {e}")

        logger.info("Neo4j indexes ensured")

    async def get_stats(self) -> dict[str, Any]:
        """
        Get graph statistics for monitoring.

        Returns:
            Dictionary with node/relationship counts and index status.
        """
        if self._driver is None:
            return {"connected": False}

        stats: dict[str, Any] = {"connected": True}

        try:
            async with self._driver.session(database=self._settings.database) as session:
                # Node count
                result = await session.run("MATCH (n) RETURN count(n) AS count")
                record = await result.single()
                stats["node_count"] = record["count"] if record else 0

                # Relationship count
                result = await session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                record = await result.single()
                stats["relationship_count"] = record["count"] if record else 0

                # External evidence count
                result = await session.run("MATCH (e:ExternalEvidence) RETURN count(e) AS count")
                record = await result.single()
                stats["external_evidence_count"] = record["count"] if record else 0

        except Exception as e:
            logger.warning(f"Stats collection failed: {e}")
            stats["error"] = str(e)

        return stats
