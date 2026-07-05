CREATE TABLE IF NOT EXISTS corpus_ingestion_batches (
  run_id TEXT NOT NULL,
  batch INTEGER NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  seen INTEGER NOT NULL DEFAULT 0,
  indexed INTEGER NOT NULL DEFAULT 0,
  chunks INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  started_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  completed_at REAL,
  PRIMARY KEY (run_id, batch),
  FOREIGN KEY (run_id) REFERENCES corpus_ingestion_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_corpus_ingestion_batches_status ON corpus_ingestion_batches(status);
