#!/usr/bin/env python3
"""
Wikipedia XML Dump to Neo4j Importer (Optimized & Robust)
=========================================================

Features:
- Natural Sorting: Handles 'articles2' before 'articles10' correctly.
- Robust Resume: Works even if files are deleted or moved between runs.
- Memory Efficient: Streaming XML parsing.
- Gap Handling: Automatically picks the earliest available file (e.g., starts at articles22 if 1-21 are missing).

Usage:
    python import_wikipedia_to_neo4j.py --directory /path/to/dumps
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from pathlib import Path
from threading import Event
from typing import Iterator, Any
from xml.etree.ElementTree import iterparse

# Global shutdown event
shutdown_event = Event()

# Third-party imports
try:
    import wikitextparser as wtp
except ImportError:
    print("Empfehlung: Installiere 'wikitextparser' für bessere Textqualität.")
    print("pip install wikitextparser")
    # Wir machen weiter, nutzen aber den Regex Fallback
    wtp = None

from neo4j import GraphDatabase

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wikipedia_import.log"),
    ],
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    logger.info("\n⚠️  Signal empfangen. Beende aktuellen Batch sauber...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

PROGRESS_FILE = "wikipedia_import_progress.json"

# --- Helper Functions ---

def natural_sort_key(s: str | Path) -> list:
    """
    Ermöglicht "natürliches" Sortieren (z.B. articles2 kommt vor articles10).
    Zerlegt den String in Text- und Zahlenblöcke.
    """
    s = str(s)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def strip_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1] if "}" in tag else tag
    return tag

# --- Data Structures ---

@dataclass
class WikiArticle:
    title: str
    page_id: int
    revision_id: int
    text: str
    categories: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    redirect_to: str | None = None

@dataclass
class ImportProgress:
    """
    Speichert den Fortschritt.
    WICHTIG: Speichert nur Dateinamen (Basenames), keine absoluten Pfade,
    damit das Skript auch nach Verschieben von Ordnern funktioniert.
    """
    processed_filenames: list[str] = field(default_factory=list)  # Nur Dateinamen!
    current_filename: str | None = None  # Nur Dateiname!
    last_page_id: int = 0
    total_articles: int = 0
    total_relationships: int = 0
    updated_at: str = ""

    def save(self, filepath: str = PROGRESS_FILE) -> None:
        self.updated_at = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, filepath: str = PROGRESS_FILE) -> "ImportProgress":
        if not os.path.exists(filepath):
            return cls()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Migration für alte Progress-Files (falls 'processed_files' statt 'processed_filenames')
            if 'processed_files' in data:
                data['processed_filenames'] = [os.path.basename(p) for p in data.pop('processed_files')]
            if 'current_file' in data:
                data['current_filename'] = os.path.basename(data.pop('current_file')) if data['current_file'] else None
                
            return cls(**data)
        except Exception as e:
            logger.warning(f"Konnte Progress-Datei nicht lesen ({e}). Starte neu.")
            return cls()

# --- Parsing & Cleaning Logic ---

def clean_wiki_text(text: str) -> str:
    """Bereinigt Wiki-Markup zu Plain Text."""
    if not text: return ""
    
    # 1. Parsing (mit wikitextparser wenn vorhanden, sonst Regex)
    if wtp:
        try:
            text = wtp.parse(text).plain_text()
        except Exception:
            pass # Fallback to regex below
    
    # Regex Fallback / Cleanup
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text) # Links
    text = re.sub(r"\{\{[^}]*\}\}", "", text) # Templates
    text = re.sub(r"'''?", "", text) # Bold/Italic
    text = unescape(text) # HTML Entities
    text = re.sub(r"<[^>]+>", "", text) # HTML Tags
    
    # Whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_metadata(text: str):
    """Extrahiert Kategorien, Links und Redirects."""
    cats = []
    links = []
    redirect = None
    
    # Redirect check
    redir_match = re.match(r"#REDIRECT\s*\[\[([^\]]+)\]\]", text, re.IGNORECASE)
    if redir_match:
        redirect = redir_match.group(1).strip()
        return cats, links, redirect

    # Kategorien
    for match in re.finditer(r"\[\[Category:([^|\]]+)(?:\|[^\]]*)?\]\]", text, re.IGNORECASE):
        cats.append(match.group(1).strip())

    # Interne Links (nur Main Namespace, keine Files/Talks)
    for match in re.finditer(r"\[\[([^|\]#]+)(?:[#|][^\]]*)?\]\]", text):
        link = match.group(1).strip()
        if link and ":" not in link:
            links.append(link)
            
    return cats, links[:50], redirect # Limit links to 50 for performance

def process_article_node(data: tuple) -> WikiArticle | None:
    """Worker Funktion für ThreadPool."""
    title, page_id, revision_id, raw_text = data
    
    try:
        cats, links, redirect = extract_metadata(raw_text)
        
        if redirect:
             return WikiArticle(title, page_id, revision_id, "", [], [], redirect)

        clean_text = clean_wiki_text(raw_text)
        
        # Ignoriere zu kurze Artikel (oft Stubs oder Fehler)
        if len(clean_text) < 150:
            return None

        return WikiArticle(
            title=title,
            page_id=page_id,
            revision_id=revision_id,
            text=clean_text[:10000], # Neo4j Text Limit beachten
            categories=cats,
            links=links
        )
    except Exception:
        return None

# --- File Handling ---

def find_dump_files(directory: str) -> list[Path]:
    """Findet XML Dateien und sortiert sie natürlich (1, 2, 10, 22)."""
    p = Path(directory)
    if not p.exists():
        return []
    
    files = list(p.glob("*.xml*"))
    # Filter files that look like dumps (contain 'pages-articles')
    files = [f for f in files if "pages-articles" in f.name and f.is_file()]
    
    # Natural Sort: Wichtig für Split-Files (articles1, articles2, articles10)
    files.sort(key=lambda x: natural_sort_key(x.name))
    
    return files

def iter_xml_articles(filepath: Path, start_after_id: int = 0) -> Iterator[tuple]:
    """Generator, der Artikel aus dem XML streamt."""
    # Compression support
    if str(filepath).endswith(".bz2"):
        import bz2; open_func = bz2.open
    elif str(filepath).endswith(".gz"):
        import gzip; open_func = gzip.open
    else:
        open_func = open

    context = None
    try:
        context = iterparse(open_func(filepath, "rb"), events=("start", "end"))
        current = {}
        in_page = False
        
        for event, elem in context:
            tag = strip_namespace(elem.tag)
            
            if event == "start" and tag == "page":
                in_page = True
                current = {}
            
            elif event == "end":
                if tag == "page":
                    in_page = False
                    ns = int(current.get("ns", 0))
                    pid = int(current.get("id", 0))
                    
                    # Logik: Nur Namespace 0 (Artikel), Resume-Logik beachten
                    if ns == 0 and pid > start_after_id:
                        yield (
                            current.get("title", ""),
                            pid,
                            int(current.get("revision", 0)),
                            current.get("text", "")
                        )
                    
                    current = {}
                    elem.clear() # Memory freigeben!
                    
                elif in_page:
                    if tag == "title": current["title"] = elem.text or ""
                    elif tag == "id" and "id" not in current: current["id"] = elem.text # Page ID
                    elif tag == "id": current["revision"] = elem.text # Rev ID
                    elif tag == "ns": current["ns"] = elem.text
                    elif tag == "text": current["text"] = elem.text or ""

    except Exception as e:
        logger.error(f"Fehler beim Lesen von {filepath.name}: {e}")
        raise
    finally:
        if context: del context

# --- Neo4j Class (Updated with MERGE optimization) ---

class Neo4jImporter:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.setup_schema()

    def setup_schema(self):
        with self.driver.session() as s:
            # Constraints sind essentiell für MERGE Performance
            cmds = [
                "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.page_id IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE INDEX article_title IF NOT EXISTS FOR (a:Article) ON (a.title)"
            ]
            for cmd in cmds:
                s.run(cmd)

    def import_batch(self, articles: list[WikiArticle]):
        if not articles: return 0, 0
        
        # Datensätze vorbereiten
        nodes_data = []
        rels_cat_data = []
        rels_link_data = []
        rels_redir_data = []
        
        for a in articles:
            # Base Node
            nodes_data.append({
                "pid": a.page_id, "tit": a.title, "txt": a.text, "redir": bool(a.redirect_to)
            })
            # Categories
            for c in a.categories:
                rels_cat_data.append({"pid": a.page_id, "cat": c})
            # Links (wenn kein Redirect)
            if not a.redirect_to:
                for l in a.links:
                    rels_link_data.append({"pid": a.page_id, "target": l})
            # Redirect
            if a.redirect_to:
                rels_redir_data.append({"pid": a.page_id, "target": a.redirect_to})

        with self.driver.session() as session:
            # 1. Artikel Knoten erstellen (MERGE)
            session.run("""
                UNWIND $batch AS row
                MERGE (a:Article {page_id: row.pid})
                ON CREATE SET a.title = row.tit, a.text = row.txt, a.is_redirect = row.redir
                ON MATCH SET a.title = row.tit, a.text = row.txt // Update content if re-importing
            """, batch=nodes_data)

            # 2. Kategorien verknüpfen
            if rels_cat_data:
                session.run("""
                    UNWIND $batch AS row
                    MATCH (a:Article {page_id: row.pid})
                    MERGE (c:Category {name: row.cat})
                    MERGE (a)-[:IN_CATEGORY]->(c)
                """, batch=rels_cat_data)
            
            # 3. Redirects
            if rels_redir_data:
                session.run("""
                    UNWIND $batch AS row
                    MATCH (a:Article {page_id: row.pid})
                    MERGE (t:Article {title: row.target}) // Target might trigger stub creation
                    MERGE (a)-[:REDIRECTS_TO]->(t)
                """, batch=rels_redir_data)
                
            # 4. Links (Achtung: Erzeugt Stubs für Artikel die noch nicht existieren)
            if rels_link_data:
                session.run("""
                    UNWIND $batch AS row
                    MATCH (a:Article {page_id: row.pid})
                    MERGE (t:Article {title: row.target})
                    MERGE (a)-[:LINKS_TO]->(t)
                """, batch=rels_link_data)
                
        return len(nodes_data), len(rels_cat_data) + len(rels_link_data)

    def close(self):
        self.driver.close()

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", default=".", help="Pfad zu den XML Dumps")
    parser.add_argument("--batch-size", "-b", type=int, default=500)
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--neo4j-uri", default=None, help="Neo4j URI (Standard: bolt://localhost:7687 oder NEO4J_URI env)")
    parser.add_argument("--neo4j-user", default=None, help="Neo4j Username (Standard: neo4j oder NEO4J_USER env)")
    parser.add_argument("--neo4j-password", default=None, help="Neo4j Password (Standard: password oder NEO4J_PASSWORD env)")
    args = parser.parse_args()

    # 1. Progress laden
    progress = ImportProgress.load()
    
    # 2. Dateien suchen & sortieren
    files = find_dump_files(args.directory)
    if not files:
        logger.error("Keine Wikipedia XML Dateien gefunden!")
        return

    logger.info(f"Gefunden: {len(files)} Dateien.")
    
    # 3. Dateien filtern (bereits erledigte überspringen)
    # Wir vergleichen nur Dateinamen, um gegen Ordner-Verschiebungen robust zu sein
    files_to_process = [f for f in files if f.name not in progress.processed_filenames]
    
    if not files_to_process:
        logger.info("Alle Dateien wurden bereits verarbeitet!")
        return

    # Check: War die letzte Datei eine, die wir noch bearbeiten müssen?
    # Falls die Datei, bei der wir unterbrochen haben, gelöscht wurde, 
    # wird sie hier nicht mehr gefunden und wir starten automatisch mit der nächsten.
    files_map = {f.name: f for f in files_to_process}
    
    # Neo4j Verbindung
    neo4j = Neo4jImporter(
        args.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        args.neo4j_user or os.getenv("NEO4J_USER", "neo4j"),
        args.neo4j_password or os.getenv("NEO4J_PASSWORD", "password123")
    )

    try:
        # Über die verbleibenden Dateien iterieren
        for dump_file in files_to_process:
            filename = dump_file.name
            
            # Resume Check: Ist das die Datei, bei der wir unterbrochen haben?
            start_page_id = 0
            if progress.current_filename == filename:
                start_page_id = progress.last_page_id
                logger.info(f"RESUME: {filename} ab Page ID {start_page_id}")
            else:
                logger.info(f"START: {filename}")
                # Reset page ID für neue Datei
                progress.current_filename = filename
                progress.last_page_id = 0
                progress.save()

            batch = []
            
            # Worker Pool
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Streaming starten
                iterator = iter_xml_articles(dump_file, start_page_id)
                
                while not shutdown_event.is_set():
                    # Batch füllen
                    try:
                        chunk = []
                        for _ in range(args.batch_size):
                            chunk.append(next(iterator))
                    except StopIteration:
                        # Datei zu Ende
                        if chunk: # Letzten Rest verarbeiten
                            pass 
                        else:
                            break
                    
                    if not chunk: break # Sollte oben gefangen werden, aber sicher ist sicher

                    batch = chunk
                    
                    # Parallel Processing (Text Cleaning)
                    futures = [executor.submit(process_article_node, item) for item in batch]
                    processed_articles = []
                    
                    current_max_id = 0
                    for f in as_completed(futures):
                        res = f.result()
                        if res:
                            processed_articles.append(res)
                            current_max_id = max(current_max_id, res.page_id)

                    # Import in DB
                    if processed_articles:
                        neo4j.import_batch(processed_articles)
                        
                        # Progress Update
                        progress.last_page_id = current_max_id
                        progress.total_articles += len(processed_articles)
                        progress.save()
                        
                        logger.info(f"Importiert: {len(processed_articles)} Artikel (Letzte ID: {current_max_id})")

                    if shutdown_event.is_set():
                        break

            if shutdown_event.is_set():
                logger.warning("Abbruch durch User.")
                break
            
            # Datei erfolgreich abgeschlossen
            progress.processed_filenames.append(filename)
            progress.current_filename = None # "Reset" für die nächste Datei
            progress.last_page_id = 0
            progress.save()
            logger.info(f"FERTIG: {filename}")

    except KeyboardInterrupt:
        logger.info("Abbruch durch KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        neo4j.close()
        progress.save()
        logger.info("Verbindung geschlossen. Status gespeichert.")

if __name__ == "__main__":
    main()