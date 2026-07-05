CREATE TABLE IF NOT EXISTS corpus_ingestion_runs (
  run_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  mode TEXT NOT NULL,
  strategy TEXT NOT NULL,
  status TEXT NOT NULL,
  cursor TEXT,
  total_seen INTEGER NOT NULL DEFAULT 0,
  total_indexed INTEGER NOT NULL DEFAULT 0,
  total_chunks INTEGER NOT NULL DEFAULT 0,
  total_errors INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  config_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  completed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_corpus_ingestion_runs_status ON corpus_ingestion_runs(status);
CREATE INDEX IF NOT EXISTS idx_corpus_ingestion_runs_created_at ON corpus_ingestion_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS corpus_documents (
  doc_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT,
  lang TEXT NOT NULL DEFAULT 'en',
  revision TEXT,
  content_hash TEXT NOT NULL,
  license TEXT,
  indexed_at REAL NOT NULL,
  metadata_json TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_documents_source_id ON corpus_documents(source, source_id);
CREATE INDEX IF NOT EXISTS idx_corpus_documents_title ON corpus_documents(title);

CREATE TABLE IF NOT EXISTS corpus_chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  char_count INTEGER NOT NULL,
  embedding_model TEXT NOT NULL,
  vector_id TEXT NOT NULL,
  indexed_at REAL NOT NULL,
  metadata_json TEXT,
  FOREIGN KEY (doc_id) REFERENCES corpus_documents(doc_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_chunks_doc_index ON corpus_chunks(doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_corpus_chunks_vector_id ON corpus_chunks(vector_id);

CREATE TABLE IF NOT EXISTS wikidata_entities (
  qid TEXT PRIMARY KEY,
  label TEXT,
  description TEXT,
  aliases_json TEXT,
  claims_json TEXT,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS document_entities (
  doc_id TEXT NOT NULL,
  qid TEXT NOT NULL,
  relation TEXT NOT NULL DEFAULT 'mentions',
  score REAL NOT NULL DEFAULT 1,
  metadata_json TEXT,
  created_at REAL NOT NULL,
  PRIMARY KEY (doc_id, qid, relation)
);

CREATE INDEX IF NOT EXISTS idx_document_entities_qid ON document_entities(qid);

CREATE TABLE IF NOT EXISTS corpus_graph_edges (
  edge_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  edge_type TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1,
  metadata_json TEXT,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corpus_graph_edges_source ON corpus_graph_edges(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_corpus_graph_edges_target ON corpus_graph_edges(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_corpus_graph_edges_type ON corpus_graph_edges(edge_type);
