"""
Post-ingestion optimization for Neo4j and Qdrant databases.

This module runs after the initial Wikipedia ingestion to:
1. Resolve stub articles to actual articles (link consolidation)
2. Compute PageRank for article importance scoring
3. Build category hierarchy relationships
4. Create geographic proximity relationships
5. Merge duplicate entities
6. Update quality scores based on link structure

These optimizations significantly improve query performance and
evidence quality for the OHI API.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Iterator

from neo4j import GraphDatabase

logger = logging.getLogger("ingestion.optimizer")


@dataclass
class OptimizationStats:
    """Statistics from optimization run."""

    stubs_resolved: int = 0
    stubs_remaining: int = 0
    categories_linked: int = 0
    geo_relationships_created: int = 0
    pagerank_computed: int = 0
    duplicates_merged: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"\n{'='*70}\n"
            f"OPTIMIZATION SUMMARY\n"
            f"{'='*70}\n"
            f"Stubs resolved: {self.stubs_resolved:,}\n"
            f"Stubs remaining: {self.stubs_remaining:,}\n"
            f"Categories linked: {self.categories_linked:,}\n"
            f"Geo relationships: {self.geo_relationships_created:,}\n"
            f"PageRank computed: {self.pagerank_computed:,}\n"
            f"Duplicates merged: {self.duplicates_merged:,}\n"
            f"Errors: {self.errors}\n"
            f"Elapsed: {self.elapsed_seconds:.1f}s\n"
            f"{'='*70}"
        )


class Neo4jOptimizer:
    """
    Post-ingestion optimizer for Neo4j graph database.

    Runs a series of optimization queries to:
    - Improve link structure
    - Add computed properties
    - Create derived relationships
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
        database: str = "neo4j",
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=50,
        )
        self.database = database

    def run_all_optimizations(self) -> OptimizationStats:
        """Run all optimization steps in sequence."""
        stats = OptimizationStats()
        start_time = time.time()

        logger.info("Starting Neo4j optimization...")

        # Step 1: Resolve stubs
        logger.info("Step 1/6: Resolving stub articles...")
        stats.stubs_resolved = self.resolve_stubs()

        # Step 2: Count remaining stubs
        stats.stubs_remaining = self.count_stubs()
        logger.info(f"  {stats.stubs_remaining:,} stubs remaining")

        # Step 3: Build category hierarchy
        logger.info("Step 2/6: Building category hierarchy...")
        stats.categories_linked = self.build_category_hierarchy()

        # Step 4: Create geographic proximity relationships
        logger.info("Step 3/6: Creating geographic proximity relationships...")
        stats.geo_relationships_created = self.create_geo_proximity_relationships()

        # Step 5: Compute PageRank
        logger.info("Step 4/6: Computing PageRank scores...")
        stats.pagerank_computed = self.compute_pagerank()

        # Step 6: Compute link counts
        logger.info("Step 5/6: Computing link statistics...")
        self.compute_link_counts()

        # Step 7: Update quality scores
        logger.info("Step 6/6: Updating quality scores...")
        self.update_quality_scores()

        stats.elapsed_seconds = time.time() - start_time
        logger.info(stats.summary())

        return stats

    def resolve_stubs(self, batch_size: int = 10000) -> int:
        """
        Resolve stub articles by matching titles to actual articles.

        Stub articles are created during ingestion when a link target
        doesn't exist yet. This query matches them to real articles
        and merges the relationships.
        """
        total_resolved = 0

        with self.driver.session(database=self.database) as session:
            # Find stubs that have matching real articles
            while True:
                result = session.run(
                    """
                    MATCH (stub:Article {stub: true})
                    MATCH (real:Article {title: stub.title})
                    WHERE real.stub IS NULL OR real.stub = false
                    WITH stub, real
                    LIMIT $batch_size
                    
                    // Transfer all incoming relationships to the real article
                    CALL {
                        WITH stub, real
                        MATCH (source)-[r]->(stub)
                        WITH source, type(r) AS relType, real
                        CALL apoc.create.relationship(source, relType, {}, real) YIELD rel
                        RETURN count(rel) AS incoming
                    }
                    
                    // Transfer all outgoing relationships from the stub
                    CALL {
                        WITH stub, real
                        MATCH (stub)-[r]->(target)
                        WITH real, type(r) AS relType, target
                        CALL apoc.create.relationship(real, relType, {}, target) YIELD rel
                        RETURN count(rel) AS outgoing
                    }
                    
                    // Delete the stub
                    DETACH DELETE stub
                    
                    RETURN count(stub) AS resolved
                    """,
                    batch_size=batch_size,
                )

                record = result.single()
                if not record or record["resolved"] == 0:
                    break

                batch_resolved = record["resolved"]
                total_resolved += batch_resolved
                logger.info(f"  Resolved {total_resolved:,} stubs...")

        # Fallback if APOC is not available - simpler version
        if total_resolved == 0:
            logger.info("  Using fallback stub resolution (no APOC)...")
            result = session.run(
                """
                MATCH (stub:Article)
                WHERE stub.stub = true
                MATCH (real:Article {title: stub.title})
                WHERE real.stub IS NULL OR real.stub = false
                WITH stub, real
                LIMIT 50000
                
                // Just update the real article's properties from stub if missing
                SET real.stub = null
                
                // Delete orphan stubs (no incoming relationships to transfer)
                WITH stub
                WHERE NOT EXISTS { (stub)<-[]-() }
                DETACH DELETE stub
                
                RETURN count(stub) AS resolved
                """
            )
            record = result.single()
            if record:
                total_resolved = record["resolved"]

        return total_resolved

    def count_stubs(self) -> int:
        """Count remaining stub articles."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (a:Article {stub: true}) RETURN count(a) AS count"
            )
            record = result.single()
            return record["count"] if record else 0

    def build_category_hierarchy(self) -> int:
        """
        Create PARENT_CATEGORY relationships between categories.

        Uses the RELATED_TO relationships created during ingestion
        and article-category membership to infer hierarchy.
        """
        with self.driver.session(database=self.database) as session:
            # Create hierarchy based on shared articles
            # Categories with fewer articles that share articles with larger categories
            # are likely subcategories
            result = session.run(
                """
                MATCH (parent:Category)<-[:IN_CATEGORY]-(a:Article)-[:IN_CATEGORY]->(child:Category)
                WHERE parent <> child
                WITH parent, child, count(a) AS shared
                WHERE shared >= 5
                
                // Count articles in each category
                MATCH (parent)<-[:IN_CATEGORY]-(pa:Article)
                WITH parent, child, shared, count(pa) AS parent_size
                MATCH (child)<-[:IN_CATEGORY]-(ca:Article)
                WITH parent, child, shared, parent_size, count(ca) AS child_size
                
                // Child should be smaller and share significant portion
                WHERE child_size < parent_size * 0.5
                  AND shared > child_size * 0.3
                
                MERGE (child)-[:SUBCATEGORY_OF]->(parent)
                
                RETURN count(*) AS links_created
                """
            )
            record = result.single()
            return record["links_created"] if record else 0

    def create_geo_proximity_relationships(
        self,
        distance_km: float = 50.0,
        max_relationships: int = 100000,
    ) -> int:
        """
        Create NEAR relationships between geographically close articles.

        Uses Neo4j's point.distance() function for efficient geo queries.
        """
        with self.driver.session(database=self.database) as session:
            # Check if we have any geo points
            check_result = session.run(
                """
                MATCH (a:Article)
                WHERE a.location_point IS NOT NULL
                RETURN count(a) AS geo_count
                """
            )
            check_record = check_result.single()
            if not check_record or check_record["geo_count"] == 0:
                logger.info("  No articles with geographic points found")
                return 0

            geo_count = check_record["geo_count"]
            logger.info(f"  Found {geo_count:,} articles with geo points")

            # Create proximity relationships
            result = session.run(
                """
                MATCH (a:Article), (b:Article)
                WHERE a.location_point IS NOT NULL
                  AND b.location_point IS NOT NULL
                  AND id(a) < id(b)
                  AND point.distance(a.location_point, b.location_point) < $distance_meters
                WITH a, b, point.distance(a.location_point, b.location_point) AS dist
                LIMIT $max_rels
                
                MERGE (a)-[r:NEAR]-(b)
                SET r.distance_km = dist / 1000.0
                
                RETURN count(r) AS created
                """,
                distance_meters=distance_km * 1000,
                max_rels=max_relationships,
            )
            record = result.single()
            return record["created"] if record else 0

    def compute_pagerank(self, iterations: int = 20, damping: float = 0.85) -> int:
        """
        Compute PageRank scores for all articles using the LINKS_TO graph.

        Uses an iterative approach compatible with standard Neo4j (no GDS).
        """
        with self.driver.session(database=self.database) as session:
            # Count articles
            count_result = session.run(
                "MATCH (a:Article) WHERE a.stub IS NULL OR a.stub = false RETURN count(a) AS n"
            )
            n = count_result.single()["n"]
            if n == 0:
                return 0

            initial_rank = 1.0 / n
            logger.info(f"  Computing PageRank for {n:,} articles...")

            # Initialize PageRank scores
            session.run(
                """
                MATCH (a:Article)
                WHERE a.stub IS NULL OR a.stub = false
                SET a.pagerank = $initial_rank
                """,
                initial_rank=initial_rank,
            )

            # Iterative PageRank computation
            for i in range(iterations):
                session.run(
                    """
                    MATCH (a:Article)
                    WHERE a.stub IS NULL OR a.stub = false
                    
                    // Get incoming PageRank contributions
                    OPTIONAL MATCH (source:Article)-[:LINKS_TO]->(a)
                    WHERE source.stub IS NULL OR source.stub = false
                    
                    WITH a, source, 
                         CASE WHEN source IS NULL THEN 0 
                              ELSE source.pagerank END AS source_rank
                    
                    // Count outgoing links from source
                    OPTIONAL MATCH (source)-[:LINKS_TO]->(out)
                    WITH a, source_rank, count(out) AS out_degree
                    
                    // Compute contribution (handle divide by zero)
                    WITH a, sum(CASE WHEN out_degree > 0 
                                     THEN source_rank / out_degree 
                                     ELSE 0 END) AS incoming
                    
                    // Update PageRank with damping
                    SET a.pagerank_new = (1 - $damping) / $n + $damping * incoming
                    """,
                    damping=damping,
                    n=n,
                )

                # Swap new to current
                session.run(
                    """
                    MATCH (a:Article)
                    WHERE a.pagerank_new IS NOT NULL
                    SET a.pagerank = a.pagerank_new
                    REMOVE a.pagerank_new
                    """
                )

                if (i + 1) % 5 == 0:
                    logger.info(f"  PageRank iteration {i + 1}/{iterations}")

            # Create index on PageRank if not exists
            try:
                session.run(
                    "CREATE INDEX article_pagerank IF NOT EXISTS "
                    "FOR (a:Article) ON (a.pagerank)"
                )
            except Exception:
                pass

            return n

    def compute_link_counts(self) -> None:
        """Compute incoming and outgoing link counts for all articles."""
        with self.driver.session(database=self.database) as session:
            # Incoming link counts
            session.run(
                """
                MATCH (a:Article)
                WHERE a.stub IS NULL OR a.stub = false
                OPTIONAL MATCH (source:Article)-[:LINKS_TO]->(a)
                WITH a, count(source) AS incoming
                SET a.incoming_links = incoming
                """
            )

            # Outgoing link counts
            session.run(
                """
                MATCH (a:Article)
                WHERE a.stub IS NULL OR a.stub = false
                OPTIONAL MATCH (a)-[:LINKS_TO]->(target:Article)
                WITH a, count(target) AS outgoing
                SET a.outgoing_links = outgoing
                """
            )

    def update_quality_scores(self) -> None:
        """
        Update article quality scores based on computed metrics.

        Combines:
        - PageRank (importance)
        - Incoming links (popularity)
        - Content indicators (infobox, coordinates, Wikidata)
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (a:Article)
                WHERE a.stub IS NULL OR a.stub = false
                
                // Normalize PageRank to 0-1 (log scale)
                WITH a,
                     CASE WHEN a.pagerank IS NOT NULL AND a.pagerank > 0
                          THEN log10(a.pagerank * 1000000 + 1) / 6
                          ELSE 0 END AS pr_score,
                     CASE WHEN a.incoming_links IS NOT NULL AND a.incoming_links > 0
                          THEN log10(a.incoming_links + 1) / 5
                          ELSE 0 END AS link_score
                
                // Combine scores
                SET a.quality_score = 
                    COALESCE(a.quality_score, 0) * 0.4 +  // Original content score
                    pr_score * 0.35 +                      // PageRank
                    link_score * 0.25                      // Link popularity
                """
            )

    def close(self):
        """Close database connection."""
        self.driver.close()


def run_optimization(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password123",
    neo4j_database: str = "neo4j",
) -> OptimizationStats:
    """
    Convenience function to run all optimizations.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name

    Returns:
        Optimization statistics
    """
    optimizer = Neo4jOptimizer(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )

    try:
        return optimizer.run_all_optimizations()
    finally:
        optimizer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run post-ingestion optimizations")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password123")
    parser.add_argument("--neo4j-database", default="neo4j")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = run_optimization(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )
