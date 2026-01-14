#!/usr/bin/env python3
"""
Neo4j Wikipedia Graph Optimizer
===============================

Führt Aufräumarbeiten, Deduplizierung und Graph-Analysen durch,
um die Datenbank "schlauer" und performanter zu machen.

Features:
1. Redirect Resolution: Leitet Links direkt auf das Ziel um.
2. Stub Cleanup: Löscht leere Platzhalter, wenn echte Artikel existieren.
3. PageRank: Berechnet die Wichtigkeit von Artikeln.
4. Community Detection: Findet Themen-Cluster (Louvain Algorithmus).
"""

import os
import logging
import time
from neo4j import GraphDatabase

# Konfiguration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class GraphOptimizer:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    def run_query(self, query, params=None, desc="Query"):
        """Führt eine Cypher-Query aus und misst die Zeit."""
        start = time.time()
        with self.driver.session() as session:
            result = session.run(query, params or {})
            summary = result.consume()
            counters = summary.counters
            
            duration = time.time() - start
            logger.info(f"✅ {desc} fertig in {duration:.2f}s")
            return counters

    def resolve_redirects(self):
        """
        WICHTIGSTE FUNKTION:
        Wenn A -> B (Redirect) -> C (Echter Artikel),
        dann ändere alle Links die auf B zeigen, sodass sie direkt auf C zeigen.
        Lösche danach B.
        """
        logger.info("🚀 Starte Redirect-Auflösung (Das kann dauern)...")
        
        # 1. Links umbiegen (Redirects umgehen)
        # Wir nehmen alle Knoten, die auf einen Redirect zeigen, und verbinden sie direkt mit dem Ziel.
        query = """
            MATCH (source)-[r1:LINKS_TO]->(redirect:Article {is_redirect: true})-[r2:REDIRECTS_TO]->(target:Article)
            MERGE (source)-[r_new:LINKS_TO]->(target)
            DELETE r1
        """
        # Da das sehr viele Daten sein können, machen wir das in Batches (apoc.periodic.iterate wäre besser, 
        # aber wir bleiben hier bei reinem Cypher/Python für Kompatibilität).
        # Hier eine einfache iterative Schleife:
        
        while True:
            batch_query = f"""
                MATCH (source)-[r1:LINKS_TO]->(redirect:Article {{is_redirect: true}})-[r2:REDIRECTS_TO]->(target:Article)
                WITH source, r1, redirect, target
                LIMIT 10000
                MERGE (source)-[:LINKS_TO]->(target)
                DELETE r1
                RETURN count(*) as count
            """
            with self.driver.session() as session:
                res = session.run(batch_query).single()
                count = res["count"]
                logger.info(f"   Redirects umgebogen: {count} (laufend...)")
                if count == 0:
                    break

        # 2. Alte Redirect-Knoten löschen, die nun isoliert sind oder nur noch als Sprungbrett dienten
        logger.info("🧹 Lösche verarbeitete Redirect-Knoten...")
        self.run_query("""
            MATCH (r:Article {is_redirect: true})
            WHERE NOT (r)--() OR (r)-[:REDIRECTS_TO]->() 
            DETACH DELETE r
        """, desc="Redirect-Knoten Bereinigung")

    def prune_stubs(self):
        """
        Löscht Knoten, die nur als Platzhalter (Stubs) erstellt wurden, 
        aber nie mit Inhalt gefüllt wurden und weniger als 2 Verbindungen haben.
        """
        logger.info("✂️ Entferne irrelevante Stubs...")
        query = """
            MATCH (a:Article)
            WHERE a.text IS NULL 
            AND size((a)--()) <= 1
            DETACH DELETE a
        """
        # Auch hier in Batches, um Timeouts zu vermeiden
        while True:
            batch_q = """
                MATCH (a:Article)
                WHERE a.text IS NULL AND size((a)--()) <= 1
                WITH a LIMIT 10000
                DETACH DELETE a
                RETURN count(a) as count
            """
            with self.driver.session() as session:
                res = session.run(batch_q).single()
                count = res["count"]
                logger.info(f"   Stubs gelöscht: {count}")
                if count == 0:
                    break

    def check_gds_availability(self):
        """Prüft, ob das Graph Data Science (GDS) Plugin installiert ist."""
        try:
            with self.driver.session() as s:
                s.run("CALL gds.version()")
            return True
        except Exception:
            logger.warning("⚠️  Neo4j GDS Plugin nicht gefunden. PageRank & Community Detection werden übersprungen.")
            logger.warning("   (Installiere GDS Plugin in Neo4j Desktop/Docker für diese Features)")
            return False

    def calculate_importance(self):
        """
        Berechnet PageRank.
        Voraussetzung: Graph Data Science (GDS) Library ist installiert.
        """
        if not self.check_gds_availability(): return

        logger.info("🧠 Berechne PageRank (Wichtigkeit)...")
        
        # 1. Graph im Arbeitsspeicher projizieren
        graph_name = "wiki_graph"
        
        # Falls Graph schon existiert, droppen
        self.run_query(f"CALL gds.graph.drop('{graph_name}', false)", desc="GDS Graph Cleanup")

        # Projizieren (Wir betrachten Artikel und deren Links)
        self.run_query(f"""
            CALL gds.graph.project(
                '{graph_name}',
                'Article',
                'LINKS_TO'
            )
        """, desc="Graph Projektion")

        # 2. PageRank berechnen und in Datenbank schreiben
        self.run_query(f"""
            CALL gds.pageRank.write(
                '{graph_name}', 
                {{ 
                    maxIterations: 20, 
                    dampingFactor: 0.85, 
                    writeProperty: 'pagerank' 
                }}
            )
        """, desc="PageRank Berechnung")
        
        # 3. Graph aus Speicher entfernen
        self.run_query(f"CALL gds.graph.drop('{graph_name}', false)", desc="Graph Release")

    def detect_communities(self):
        """
        Findet Themen-Cluster mittels Louvain Algorithmus.
        """
        if not self.check_gds_availability(): return

        logger.info("🏘️ Erkenne Communities (Themen-Cluster)...")
        
        graph_name = "wiki_community"
        self.run_query(f"CALL gds.graph.drop('{graph_name}', false)", desc="GDS Graph Cleanup")

        # Projizieren (als ungerichteten Graphen für bessere Cluster)
        self.run_query(f"""
            CALL gds.graph.project(
                '{graph_name}',
                'Article',
                {{
                    LINKS_TO: {{ orientation: 'UNDIRECTED' }}
                }}
            )
        """, desc="Graph Projektion für Communities")

        # Louvain ausführen
        self.run_query(f"""
            CALL gds.louvain.write(
                '{graph_name}',
                {{ writeProperty: 'communityId' }}
            )
        """, desc="Louvain Community Detection")
        
        self.run_query(f"CALL gds.graph.drop('{graph_name}', false)", desc="Graph Release")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Neo4j Wikipedia Graph Optimizer")
    parser.add_argument("--neo4j-uri", default=None, help="Neo4j URI (Standard: bolt://localhost:7687 oder NEO4J_URI env)")
    parser.add_argument("--neo4j-user", default=None, help="Neo4j Username (Standard: neo4j oder NEO4J_USER env)")
    parser.add_argument("--neo4j-password", default=None, help="Neo4j Password (Standard: password oder NEO4J_PASSWORD env)")
    args = parser.parse_args()
    
    logger.info("Start Neo4j Optimization...")
    opt = GraphOptimizer(
        args.neo4j_uri or NEO4J_URI,
        args.neo4j_user or NEO4J_USER,
        args.neo4j_password or NEO4J_PASSWORD
    )
    
    try:
        # 1. Struktur bereinigen
        opt.resolve_redirects()
        opt.prune_stubs()
        
        # 2. Graph Analyse (Benötigt GDS Plugin)
        opt.calculate_importance()
        opt.detect_communities()
        
        logger.info("\n🎉 Optimierung abgeschlossen!")
        logger.info("Tipp: Führe jetzt in Neo4j Browser aus:")
        logger.info("MATCH (a:Article) RETURN a.title, a.pagerank ORDER BY a.pagerank DESC LIMIT 10")
        
    except Exception as e:
        logger.error(f"Fehler bei der Optimierung: {e}")
    finally:
        opt.close()

if __name__ == "__main__":
    main()