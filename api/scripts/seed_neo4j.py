#!/usr/bin/env python3
"""
Neo4j Knowledge Base Seeder
===========================

Populates Neo4j with common facts for verification.
Run this script after starting the containers.

Usage:
    python scripts/seed_neo4j.py
"""

import asyncio
import logging
from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection settings (match docker-compose.yml)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# Knowledge base: (Subject, Predicate, Object, Source)
FACTS = [
    # Famous Scientists
    ("Albert Einstein", "was_born_in", "Ulm, Germany", "wikipedia"),
    ("Albert Einstein", "birth_date", "1879-03-14", "wikipedia"),
    ("Albert Einstein", "profession", "physicist", "wikipedia"),
    ("Albert Einstein", "nationality", "German-American", "wikipedia"),
    ("Albert Einstein", "developed", "Theory of Relativity", "wikipedia"),
    ("Albert Einstein", "won", "Nobel Prize in Physics 1921", "wikipedia"),
    ("Albert Einstein", "worked_at", "Princeton University", "wikipedia"),
    ("Marie Curie", "was_born_in", "Warsaw, Poland", "wikipedia"),
    ("Marie Curie", "birth_date", "1867-11-07", "wikipedia"),
    ("Marie Curie", "profession", "physicist and chemist", "wikipedia"),
    ("Marie Curie", "discovered", "Polonium", "wikipedia"),
    ("Marie Curie", "discovered", "Radium", "wikipedia"),
    ("Marie Curie", "won", "Nobel Prize in Physics 1903", "wikipedia"),
    ("Marie Curie", "won", "Nobel Prize in Chemistry 1911", "wikipedia"),
    ("Isaac Newton", "was_born_in", "Woolsthorpe, England", "wikipedia"),
    ("Isaac Newton", "birth_date", "1643-01-04", "wikipedia"),
    ("Isaac Newton", "profession", "mathematician and physicist", "wikipedia"),
    ("Isaac Newton", "developed", "Laws of Motion", "wikipedia"),
    ("Isaac Newton", "developed", "Law of Universal Gravitation", "wikipedia"),
    ("Isaac Newton", "wrote", "Principia Mathematica", "wikipedia"),
    ("Charles Darwin", "was_born_in", "Shrewsbury, England", "wikipedia"),
    ("Charles Darwin", "profession", "naturalist", "wikipedia"),
    ("Charles Darwin", "developed", "Theory of Evolution", "wikipedia"),
    ("Charles Darwin", "wrote", "On the Origin of Species", "wikipedia"),
    ("Nikola Tesla", "was_born_in", "Smiljan, Croatia", "wikipedia"),
    ("Nikola Tesla", "profession", "inventor and engineer", "wikipedia"),
    ("Nikola Tesla", "invented", "AC Electrical System", "wikipedia"),
    ("Nikola Tesla", "invented", "Tesla Coil", "wikipedia"),
    # Programming Languages
    ("Python", "created_by", "Guido van Rossum", "wikipedia"),
    ("Python", "first_released", "1991", "wikipedia"),
    ("Python", "named_after", "Monty Python", "wikipedia"),
    ("Python", "is_a", "programming language", "wikipedia"),
    ("Python", "paradigm", "multi-paradigm", "wikipedia"),
    ("JavaScript", "created_by", "Brendan Eich", "wikipedia"),
    ("JavaScript", "first_released", "1995", "wikipedia"),
    ("JavaScript", "developed_at", "Netscape", "wikipedia"),
    ("JavaScript", "is_a", "programming language", "wikipedia"),
    ("Java", "created_by", "James Gosling", "wikipedia"),
    ("Java", "first_released", "1995", "wikipedia"),
    ("Java", "developed_at", "Sun Microsystems", "wikipedia"),
    ("Java", "is_a", "programming language", "wikipedia"),
    ("C++", "created_by", "Bjarne Stroustrup", "wikipedia"),
    ("C++", "first_released", "1985", "wikipedia"),
    ("C++", "is_a", "programming language", "wikipedia"),
    ("Rust", "created_by", "Graydon Hoare", "wikipedia"),
    ("Rust", "first_released", "2010", "wikipedia"),
    ("Rust", "developed_at", "Mozilla", "wikipedia"),
    ("Rust", "is_a", "programming language", "wikipedia"),
    # Frameworks
    ("React", "created_by", "Jordan Walke", "wikipedia"),
    ("React", "developed_at", "Meta", "wikipedia"),
    ("React", "first_released", "2013", "wikipedia"),
    ("React", "is_a", "JavaScript library", "wikipedia"),
    ("React", "used_for", "building user interfaces", "wikipedia"),
    ("FastAPI", "created_by", "Sebastián Ramírez", "wikipedia"),
    ("FastAPI", "first_released", "2018", "wikipedia"),
    ("FastAPI", "is_a", "Python web framework", "wikipedia"),
    ("FastAPI", "built_on", "Starlette", "wikipedia"),
    ("FastAPI", "built_on", "Pydantic", "wikipedia"),
    ("Django", "created_by", "Adrian Holovaty and Simon Willison", "wikipedia"),
    ("Django", "first_released", "2005", "wikipedia"),
    ("Django", "is_a", "Python web framework", "wikipedia"),
    ("Django", "named_after", "Django Reinhardt", "wikipedia"),
    ("Next.js", "created_by", "Guillermo Rauch", "wikipedia"),
    ("Next.js", "developed_at", "Vercel", "wikipedia"),
    ("Next.js", "first_released", "2016", "wikipedia"),
    ("Next.js", "is_a", "React framework", "wikipedia"),
    # Companies
    ("Microsoft", "founded_by", "Bill Gates and Paul Allen", "wikipedia"),
    ("Microsoft", "founded_in", "1975", "wikipedia"),
    ("Microsoft", "headquarters", "Redmond, Washington", "wikipedia"),
    ("Microsoft", "created", "Windows", "wikipedia"),
    ("Microsoft", "owns", "GitHub", "wikipedia"),
    ("Microsoft", "owns", "LinkedIn", "wikipedia"),
    ("Apple", "founded_by", "Steve Jobs, Steve Wozniak, and Ronald Wayne", "wikipedia"),
    ("Apple", "founded_in", "1976", "wikipedia"),
    ("Apple", "headquarters", "Cupertino, California", "wikipedia"),
    ("Apple", "created", "iPhone", "wikipedia"),
    ("Apple", "created", "MacBook", "wikipedia"),
    ("Google", "founded_by", "Larry Page and Sergey Brin", "wikipedia"),
    ("Google", "founded_in", "1998", "wikipedia"),
    ("Google", "headquarters", "Mountain View, California", "wikipedia"),
    ("Google", "parent_company", "Alphabet Inc.", "wikipedia"),
    ("Meta", "founded_by", "Mark Zuckerberg", "wikipedia"),
    ("Meta", "formerly_known_as", "Facebook", "wikipedia"),
    ("Meta", "founded_in", "2004", "wikipedia"),
    ("Meta", "headquarters", "Menlo Park, California", "wikipedia"),
    ("Meta", "owns", "Instagram", "wikipedia"),
    ("Meta", "owns", "WhatsApp", "wikipedia"),
    ("Amazon", "founded_by", "Jeff Bezos", "wikipedia"),
    ("Amazon", "founded_in", "1994", "wikipedia"),
    ("Amazon", "headquarters", "Seattle, Washington", "wikipedia"),
    ("Amazon", "owns", "AWS", "wikipedia"),
    # Geography
    ("Great Wall of China", "located_in", "China", "wikipedia"),
    ("Great Wall of China", "length", "21,196 kilometers", "wikipedia"),
    ("Great Wall of China", "is_a", "fortification", "wikipedia"),
    ("Eiffel Tower", "located_in", "Paris, France", "wikipedia"),
    ("Eiffel Tower", "height", "330 meters", "wikipedia"),
    ("Eiffel Tower", "built_in", "1889", "wikipedia"),
    ("Eiffel Tower", "designed_by", "Gustave Eiffel", "wikipedia"),
    ("Mount Everest", "located_in", "Nepal and Tibet", "wikipedia"),
    ("Mount Everest", "height", "8,849 meters", "wikipedia"),
    ("Mount Everest", "is_the", "highest mountain on Earth", "wikipedia"),
    ("Pacific Ocean", "is_the", "largest ocean on Earth", "wikipedia"),
    ("Pacific Ocean", "area", "165.25 million square kilometers", "wikipedia"),
    ("Amazon River", "located_in", "South America", "wikipedia"),
    ("Amazon River", "is_the", "largest river by volume", "wikipedia"),
    # Historical Events
    ("World War II", "started_in", "1939", "wikipedia"),
    ("World War II", "ended_in", "1945", "wikipedia"),
    ("Moon Landing", "occurred_on", "1969-07-20", "wikipedia"),
    ("Moon Landing", "mission", "Apollo 11", "wikipedia"),
    ("Moon Landing", "astronaut", "Neil Armstrong", "wikipedia"),
    ("Moon Landing", "astronaut", "Buzz Aldrin", "wikipedia"),
    ("Internet", "invented_at", "CERN and ARPANET", "wikipedia"),
    ("World Wide Web", "invented_by", "Tim Berners-Lee", "wikipedia"),
    ("World Wide Web", "invented_in", "1989", "wikipedia"),
]


async def seed_database():
    """Connect to Neo4j and insert all facts."""
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    async with driver.session() as session:
        # Create constraints for better performance
        logger.info("Creating constraints...")
        try:
            await session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
        except Exception as e:
            logger.warning(f"Constraint creation: {e}")

        # Clear existing data (optional)
        logger.info("Clearing existing data...")
        await session.run("MATCH (n) DETACH DELETE n")

        # Insert facts
        logger.info(f"Inserting {len(FACTS)} facts...")
        for subject, predicate, obj, source in FACTS:
            query = """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $object})
            MERGE (s)-[r:RELATION {type: $predicate}]->(o)
            SET r.source = $source
            RETURN s.name, type(r), o.name
            """
            await session.run(
                query,
                subject=subject,
                predicate=predicate,
                object=obj,
                source=source,
            )

        # Also create typed relationships for common predicates
        logger.info("Creating typed relationships...")
        typed_relations = [
            ("is_a", "IS_A"),
            ("created_by", "CREATED_BY"),
            ("was_born_in", "BORN_IN"),
            ("developed", "DEVELOPED"),
            ("won", "WON"),
            ("founded_by", "FOUNDED_BY"),
            ("located_in", "LOCATED_IN"),
            ("created", "CREATED"),
            ("owns", "OWNS"),
        ]

        for predicate, rel_type in typed_relations:
            query = f"""
            MATCH (s:Entity)-[r:RELATION {{type: $predicate}}]->(o:Entity)
            MERGE (s)-[:{rel_type}]->(o)
            """
            await session.run(query, predicate=predicate)

        # Count results
        result = await session.run("MATCH (n:Entity) RETURN count(n) as count")
        record = await result.single()
        entity_count = record["count"]

        result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
        record = await result.single()
        rel_count = record["count"]

        logger.info(f"✅ Seeded {entity_count} entities and {rel_count} relationships")

    await driver.close()


if __name__ == "__main__":
    asyncio.run(seed_database())
