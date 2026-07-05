ALTER TABLE corpus_ingestion_runs ADD COLUMN batches_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE corpus_ingestion_runs ADD COLUMN batches_completed INTEGER NOT NULL DEFAULT 0;
