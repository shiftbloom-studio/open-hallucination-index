"""
Neo4j graph store with rich relationship modeling.

Features:
- Optimized batch writes with UNWIND
- Rich relationship types for knowledge graph
- Async upload support with deadlock retry
- Connection pooling for throughput
- Full-text search indexes
"""

from __future__ import annotations

import logging
import random
import threading
import time
from queue import Empty, Queue

from neo4j import GraphDatabase
from neo4j.exceptions import TransientError

from ingestion.models import ProcessedArticle

logger = logging.getLogger("ingestion.neo4j")

# Retry settings for deadlock handling
MAX_RETRIES = 8
BASE_RETRY_DELAY = 0.2  # seconds
MAX_RETRY_DELAY = 4.0  # seconds


class Neo4jGraphStore:
    """
    Neo4j store with optimized batch writes and rich relationship modeling.

    Relationship Types Created (28 total):
    Core Article Relationships:
    - LINKS_TO: Article links to another article
    - IN_CATEGORY: Article belongs to a category  
    - MENTIONS: Article mentions an entity
    - SEE_ALSO: Explicit "See also" relationship
    - DISAMBIGUATES: Disambiguation page links
    - RELATED_TO: Category-category co-occurrence

    Person Relationships:
    - LOCATED_IN: Person/thing located in place
    - HAS_OCCUPATION: Person has occupation
    - HAS_NATIONALITY: Person has nationality
    - MARRIED_TO: Spouse relationship
    - PARENT_OF: Parent-child relationship
    - CHILD_OF: Child-parent relationship
    - EDUCATED_AT: Attended educational institution
    - EMPLOYED_BY: Works for organization
    - WON_AWARD: Received award

    Creative/Influence Relationships:
    - AUTHORED: Created a work
    - HAS_GENRE: Work has genre
    - INFLUENCED_BY: Was influenced by
    - INFLUENCED: Influenced others

    Organization Relationships:
    - FOUNDED_BY: Organization founded by person
    - HEADQUARTERED_IN: HQ location
    - IN_INDUSTRY: Operates in industry

    Geographic Relationships:
    - IN_COUNTRY: Located in country
    - PART_OF: Geographic hierarchy
    - CAPITAL_OF: Capital city of country/region

    Temporal Relationships:
    - PRECEDED_BY: Predecessor relationship
    - SUCCEEDED_BY: Successor relationship

    Classification:
    - INSTANCE_OF: Type classification

    Node Types:
    - Article: Main Wikipedia articles
    - Category: Wikipedia categories
    - Entity: Named entities (people, places, things)
    - Person: Individuals (spouses, children, founders)
    - Location: Geographic locations
    - Country: Countries
    - Occupation: Job/profession types
    - Nationality: Nationalities
    - EducationalInstitution: Schools, universities
    - Organization: Companies, employers
    - Award: Awards and honors
    - CreativeWork: Books, articles, works
    - Genre: Creative genres
    - Industry: Business industries
    - Type: Classification types

    Optimizations:
    - Large connection pool (100 connections)
    - Batched UNWIND queries
    - Async upload queue with worker threads
    - Separate queries to avoid cartesian products
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
        database: str = "neo4j",
        upload_workers: int = 4,
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=100,
            connection_acquisition_timeout=120,
        )
        self.database = database

        # Upload queue for async processing
        self._upload_queue: Queue[tuple[list[dict], threading.Event]] = Queue(
            maxsize=upload_workers * 2
        )
        self._shutdown = False

        # Start upload worker threads
        self._upload_threads = []
        for i in range(upload_workers):
            t = threading.Thread(
                target=self._upload_worker, daemon=True, name=f"neo4j_uploader_{i}"
            )
            t.start()
            self._upload_threads.append(t)

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Initialize constraints and indexes for optimal performance."""
        with self.driver.session(database=self.database) as session:
            # Constraints (also create indexes)
            constraints = [
                "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
                "CREATE CONSTRAINT occupation_name IF NOT EXISTS FOR (o:Occupation) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT nationality_name IF NOT EXISTS FOR (n:Nationality) REQUIRE n.name IS UNIQUE",
                # NEW: Constraints for new node types
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT edu_name IF NOT EXISTS FOR (e:EducationalInstitution) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT award_name IF NOT EXISTS FOR (a:Award) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT work_name IF NOT EXISTS FOR (w:CreativeWork) REQUIRE w.name IS UNIQUE",
                "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
                "CREATE CONSTRAINT industry_name IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
                "CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT type_name IF NOT EXISTS FOR (t:Type) REQUIRE t.name IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)  # type: ignore[arg-type]
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.debug(f"Constraint: {e}")

            # Full-text indexes for search
            fulltext_indexes = [
                "CREATE FULLTEXT INDEX article_search IF NOT EXISTS FOR (a:Article) ON EACH [a.title, a.first_paragraph]",
                "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
            ]

            for index in fulltext_indexes:
                try:
                    session.run(index)  # type: ignore[arg-type]
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.debug(f"Fulltext index: {e}")

            # Regular indexes for common lookups
            indexes = [
                "CREATE INDEX article_title IF NOT EXISTS FOR (a:Article) ON (a.title)",
                "CREATE INDEX article_infobox IF NOT EXISTS FOR (a:Article) ON (a.infobox_type)",
                "CREATE INDEX article_word_count IF NOT EXISTS FOR (a:Article) ON (a.word_count)",
            ]

            for index in indexes:
                try:
                    session.run(index)  # type: ignore[arg-type]
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.debug(f"Index: {e}")

            logger.info("✅ Neo4j schema initialized")

    def _upload_worker(self):
        """Worker thread that processes upload queue."""
        while not self._shutdown:
            try:
                batch_data, done_event = self._upload_queue.get(timeout=1.0)
                if batch_data is None:  # Shutdown signal
                    break
                self._do_upload(batch_data)
                done_event.set()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Neo4j upload worker error: {e}")

    def _prepare_batch_data(self, articles: list[ProcessedArticle]) -> list[dict]:
        """Prepare batch data for upload."""
        batch_data = []

        for pa in articles:
            article = pa.article
            batch_data.append(
                {
                    "id": article.id,
                    "title": article.title,
                    "url": article.url,
                    "word_count": article.word_count,
                    "first_paragraph": (
                        article.first_paragraph[:1000]
                        if article.first_paragraph
                        else ""
                    ),
                    "infobox_type": (
                        article.infobox.type if article.infobox else None
                    ),
                    "links": list(article.links)[:200],  # Top 200 links
                    "categories": list(article.categories)[:50],
                    "entities": list(article.entities)[:100],
                    "see_also": list(article.see_also_links)[:50],
                    "disambiguation": list(article.disambiguation_links)[:100],
                    "chunk_ids": [c.chunk_id for c in pa.chunks],
                    "chunk_count": len(pa.chunks),
                    # Structured data for rich relationships (existing)
                    "birth_date": article.birth_date,
                    "death_date": article.death_date,
                    "location": article.location,
                    "occupation": article.occupation,
                    "nationality": article.nationality,
                    # NEW: Additional relationship data
                    "spouse": article.spouse,
                    "children": list(article.children)[:20],
                    "parents": list(article.parents)[:10],
                    "education": list(article.education)[:10],
                    "employer": list(article.employer)[:10],
                    "awards": list(article.awards)[:20],
                    "author_of": list(article.author_of)[:20],
                    "genre": list(article.genre)[:10],
                    "influenced_by": list(article.influenced_by)[:20],
                    "influenced": list(article.influenced)[:20],
                    "founded_by": article.founded_by,
                    "founding_date": article.founding_date,
                    "headquarters": article.headquarters,
                    "industry": article.industry,
                    "country": article.country,
                    "capital_of": article.capital_of,
                    "part_of": article.part_of,
                    "predecessor": article.predecessor,
                    "successor": article.successor,
                    "instance_of": article.instance_of,
                }
            )

        return batch_data

    def _run_with_retry(self, session, query: str, batch: list[dict], operation_name: str) -> bool:
        """
        Execute a Neo4j query with exponential backoff retry on deadlock.
        
        Returns True if successful, False if all retries exhausted.
        """
        for attempt in range(MAX_RETRIES):
            try:
                session.run(query, batch=batch)
                return True
            except TransientError as e:
                # Deadlock detected - retry with exponential backoff + jitter
                if attempt < MAX_RETRIES - 1:
                    delay = min(
                        BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.1),
                        MAX_RETRY_DELAY
                    )
                    logger.debug(
                        f"Deadlock in {operation_name}, retry {attempt + 1}/{MAX_RETRIES} "
                        f"after {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(f"{operation_name} failed after {MAX_RETRIES} retries: {e}")
                    return False
            except Exception as e:
                logger.error(f"{operation_name} error: {e}")
                return False
        return False

    def _do_upload(self, batch_data: list[dict]):
        """Execute the actual upload queries with deadlock retry."""
        if not batch_data:
            return

        with self.driver.session(database=self.database) as session:
            # 1. Create/update Article nodes
            article_query = """
            UNWIND $batch AS data
            MERGE (a:Article {id: data.id})
            SET a.title = data.title,
                a.url = data.url,
                a.word_count = data.word_count,
                a.first_paragraph = data.first_paragraph,
                a.infobox_type = data.infobox_type,
                a.vector_chunk_ids = data.chunk_ids,
                a.chunk_count = data.chunk_count,
                a.birth_date = data.birth_date,
                a.death_date = data.death_date,
                a.location = data.location,
                a.occupation = data.occupation,
                a.nationality = data.nationality,
                a.instance_of = data.instance_of,
                a.country = data.country,
                a.industry = data.industry,
                a.headquarters = data.headquarters,
                a.founding_date = data.founding_date,
                a.spouse = data.spouse,
                a.founded_by = data.founded_by,
                a.part_of = data.part_of,
                a.capital_of = data.capital_of,
                a.predecessor = data.predecessor,
                a.successor = data.successor,
                a.last_updated = datetime()
            """
            self._run_with_retry(session, article_query, batch_data, "Article creation")

            # 2. Category relationships (most common)
            category_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.categories AS catName
            MERGE (c:Category {name: catName})
            MERGE (a)-[:IN_CATEGORY]->(c)
            """
            self._run_with_retry(session, category_query, batch_data, "Category relationships")

            # 3. Entity relationships
            entity_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.entities AS entityName
            MERGE (e:Entity {name: entityName})
            MERGE (a)-[:MENTIONS]->(e)
            """
            self._run_with_retry(session, entity_query, batch_data, "Entity relationships")

            # 4. Article-to-Article links (LINKS_TO)
            links_query = """
            UNWIND $batch AS data
            MATCH (source:Article {id: data.id})
            WITH source, data
            UNWIND data.links AS linkTitle
            MERGE (target:Article {title: linkTitle})
            ON CREATE SET target.stub = true
            MERGE (source)-[:LINKS_TO]->(target)
            """
            self._run_with_retry(session, links_query, batch_data, "Links relationships")

            # 5. See Also relationships (explicit related content)
            see_also_query = """
            UNWIND $batch AS data
            MATCH (source:Article {id: data.id})
            WITH source, data
            UNWIND data.see_also AS linkTitle
            MERGE (target:Article {title: linkTitle})
            ON CREATE SET target.stub = true
            MERGE (source)-[:SEE_ALSO]->(target)
            """
            self._run_with_retry(session, see_also_query, batch_data, "See also relationships")

            # 6. Disambiguation relationships
            disambig_query = """
            UNWIND $batch AS data
            MATCH (source:Article {id: data.id})
            WHERE size(data.disambiguation) > 0
            WITH source, data
            UNWIND data.disambiguation AS linkTitle
            MERGE (target:Article {title: linkTitle})
            ON CREATE SET target.stub = true
            MERGE (source)-[:DISAMBIGUATES]->(target)
            """
            self._run_with_retry(session, disambig_query, batch_data, "Disambiguation relationships")

            # 7. Location relationships
            location_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.location IS NOT NULL
            MERGE (l:Location {name: data.location})
            MERGE (a)-[:LOCATED_IN]->(l)
            """
            self._run_with_retry(session, location_query, batch_data, "Location relationships")

            # 8. Occupation relationships
            occupation_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.occupation IS NOT NULL
            MERGE (o:Occupation {name: data.occupation})
            MERGE (a)-[:HAS_OCCUPATION]->(o)
            """
            self._run_with_retry(session, occupation_query, batch_data, "Occupation relationships")

            # 9. Nationality relationships
            nationality_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.nationality IS NOT NULL
            MERGE (n:Nationality {name: data.nationality})
            MERGE (a)-[:HAS_NATIONALITY]->(n)
            """
            self._run_with_retry(session, nationality_query, batch_data, "Nationality relationships")

            # 10. Category co-occurrence (articles sharing categories are related)
            # This creates implicit relationships between categories
            category_cooccur_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE size(data.categories) >= 2
            WITH a, data.categories AS cats
            UNWIND range(0, size(cats)-2) AS i
            UNWIND range(i+1, size(cats)-1) AS j
            WITH cats[i] AS cat1, cats[j] AS cat2
            MERGE (c1:Category {name: cat1})
            MERGE (c2:Category {name: cat2})
            MERGE (c1)-[:RELATED_TO]-(c2)
            """
            # Only run for subset to avoid explosion
            subset = [d for d in batch_data if len(d.get("categories", [])) >= 2][:50]
            if subset:
                self._run_with_retry(session, category_cooccur_query, subset, "Category co-occurrence")

            # =================================================================
            # NEW RELATIONSHIPS (11-25): 15 additional relationship types
            # =================================================================

            # 11. Spouse relationships (MARRIED_TO)
            spouse_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.spouse IS NOT NULL
            MERGE (spouse:Person {name: data.spouse})
            MERGE (a)-[:MARRIED_TO]->(spouse)
            """
            self._run_with_retry(session, spouse_query, batch_data, "Spouse relationships")

            # 12. Children relationships (PARENT_OF)
            children_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.children AS childName
            MERGE (child:Person {name: childName})
            MERGE (a)-[:PARENT_OF]->(child)
            """
            self._run_with_retry(session, children_query, batch_data, "Children relationships")

            # 13. Parent relationships (CHILD_OF)
            parents_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.parents AS parentName
            MERGE (parent:Person {name: parentName})
            MERGE (a)-[:CHILD_OF]->(parent)
            """
            self._run_with_retry(session, parents_query, batch_data, "Parents relationships")

            # 14. Education relationships (EDUCATED_AT)
            education_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.education AS schoolName
            MERGE (school:EducationalInstitution {name: schoolName})
            MERGE (a)-[:EDUCATED_AT]->(school)
            """
            self._run_with_retry(session, education_query, batch_data, "Education relationships")

            # 15. Employer relationships (EMPLOYED_BY)
            employer_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.employer AS employerName
            MERGE (employer:Organization {name: employerName})
            MERGE (a)-[:EMPLOYED_BY]->(employer)
            """
            self._run_with_retry(session, employer_query, batch_data, "Employer relationships")

            # 16. Award relationships (WON_AWARD)
            awards_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.awards AS awardName
            MERGE (award:Award {name: awardName})
            MERGE (a)-[:WON_AWARD]->(award)
            """
            self._run_with_retry(session, awards_query, batch_data, "Awards relationships")

            # 17. Author relationships (AUTHORED)
            author_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.author_of AS workName
            MERGE (work:CreativeWork {name: workName})
            MERGE (a)-[:AUTHORED]->(work)
            """
            self._run_with_retry(session, author_query, batch_data, "Author relationships")

            # 18. Genre relationships (HAS_GENRE)
            genre_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.genre AS genreName
            MERGE (genre:Genre {name: genreName})
            MERGE (a)-[:HAS_GENRE]->(genre)
            """
            self._run_with_retry(session, genre_query, batch_data, "Genre relationships")

            # 19. Influenced by relationships (INFLUENCED_BY)
            influenced_by_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.influenced_by AS influencerName
            MERGE (influencer:Article {title: influencerName})
            ON CREATE SET influencer.stub = true
            MERGE (a)-[:INFLUENCED_BY]->(influencer)
            """
            self._run_with_retry(session, influenced_by_query, batch_data, "Influenced by relationships")

            # 20. Influenced relationships (INFLUENCED)
            influenced_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WITH a, data
            UNWIND data.influenced AS influencedName
            MERGE (influenced:Article {title: influencedName})
            ON CREATE SET influenced.stub = true
            MERGE (a)-[:INFLUENCED]->(influenced)
            """
            self._run_with_retry(session, influenced_query, batch_data, "Influenced relationships")

            # 21. Founded by relationships (FOUNDED_BY)
            founded_by_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.founded_by IS NOT NULL
            MERGE (founder:Person {name: data.founded_by})
            MERGE (a)-[:FOUNDED_BY]->(founder)
            """
            self._run_with_retry(session, founded_by_query, batch_data, "Founded by relationships")

            # 22. Headquarters relationships (HEADQUARTERED_IN)
            headquarters_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.headquarters IS NOT NULL
            MERGE (hq:Location {name: data.headquarters})
            MERGE (a)-[:HEADQUARTERED_IN]->(hq)
            """
            self._run_with_retry(session, headquarters_query, batch_data, "Headquarters relationships")

            # 23. Industry relationships (IN_INDUSTRY)
            industry_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.industry IS NOT NULL
            MERGE (ind:Industry {name: data.industry})
            MERGE (a)-[:IN_INDUSTRY]->(ind)
            """
            self._run_with_retry(session, industry_query, batch_data, "Industry relationships")

            # 24. Country relationships (IN_COUNTRY)
            country_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.country IS NOT NULL
            MERGE (country:Country {name: data.country})
            MERGE (a)-[:IN_COUNTRY]->(country)
            """
            self._run_with_retry(session, country_query, batch_data, "Country relationships")

            # 25. Part of relationships (PART_OF) - geographic hierarchy
            part_of_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.part_of IS NOT NULL
            MERGE (parent:Location {name: data.part_of})
            MERGE (a)-[:PART_OF]->(parent)
            """
            self._run_with_retry(session, part_of_query, batch_data, "Part of relationships")

            # 26. Predecessor relationships (PRECEDED_BY)
            predecessor_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.predecessor IS NOT NULL
            MERGE (pred:Article {title: data.predecessor})
            ON CREATE SET pred.stub = true
            MERGE (a)-[:PRECEDED_BY]->(pred)
            """
            self._run_with_retry(session, predecessor_query, batch_data, "Predecessor relationships")

            # 27. Successor relationships (SUCCEEDED_BY)
            successor_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.successor IS NOT NULL
            MERGE (succ:Article {title: data.successor})
            ON CREATE SET succ.stub = true
            MERGE (a)-[:SUCCEEDED_BY]->(succ)
            """
            self._run_with_retry(session, successor_query, batch_data, "Successor relationships")

            # 28. Instance of relationships (INSTANCE_OF) - type classification
            instance_of_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.instance_of IS NOT NULL
            MERGE (type:Type {name: data.instance_of})
            MERGE (a)-[:INSTANCE_OF]->(type)
            """
            self._run_with_retry(session, instance_of_query, batch_data, "Instance of relationships")

            # 29. Capital of relationships (CAPITAL_OF) - capital cities
            capital_of_query = """
            UNWIND $batch AS data
            MATCH (a:Article {id: data.id})
            WHERE data.capital_of IS NOT NULL
            MERGE (region:Location {name: data.capital_of})
            MERGE (a)-[:CAPITAL_OF]->(region)
            """
            self._run_with_retry(session, capital_of_query, batch_data, "Capital of relationships")

    def upload_batch_async(self, articles: list[ProcessedArticle]) -> threading.Event:
        """
        Upload a batch of articles asynchronously.

        Returns an Event that will be set when upload completes.
        """
        batch_data = self._prepare_batch_data(articles)
        done_event = threading.Event()

        if not batch_data:
            done_event.set()
            return done_event

        self._upload_queue.put((batch_data, done_event))
        return done_event

    def upload_batch_sync(self, articles: list[ProcessedArticle]) -> int:
        """
        Upload a batch of articles synchronously.

        Returns the number of articles uploaded.
        """
        batch_data = self._prepare_batch_data(articles)
        if not batch_data:
            return 0

        self._do_upload(batch_data)
        return len(batch_data)

    def flush(self):
        """Wait for all pending uploads to complete."""
        while not self._upload_queue.empty():
            try:
                batch_data, event = self._upload_queue.get_nowait()
                self._do_upload(batch_data)
                event.set()
            except Empty:
                break

    def close(self):
        """Close driver connection and cleanup."""
        self._shutdown = True

        # Signal workers to stop
        for _ in self._upload_threads:
            try:
                self._upload_queue.put((None, threading.Event()), timeout=1.0)  # type: ignore[arg-type]
            except Exception:
                pass

        # Wait for workers
        for t in self._upload_threads:
            t.join(timeout=5.0)

        self.driver.close()

    def get_stats(self) -> dict:
        """Get database statistics."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (a:Article) WITH count(a) as articles
                    MATCH (c:Category) WITH articles, count(c) as categories
                    MATCH (e:Entity) WITH articles, categories, count(e) as entities
                    MATCH ()-[r]->() WITH articles, categories, entities, count(r) as relationships
                    RETURN articles, categories, entities, relationships
                    """
                )
                record = result.single()
                if record:
                    return {
                        "articles": record["articles"],
                        "categories": record["categories"],
                        "entities": record["entities"],
                        "relationships": record["relationships"],
                    }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        return {}
