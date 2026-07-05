CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  phase TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  completed_at REAL,
  text_hash TEXT NOT NULL,
  result_json TEXT,
  error TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  claim_id TEXT NOT NULL,
  label TEXT NOT NULL,
  labeler_json TEXT NOT NULL,
  rationale TEXT,
  evidence_corrections_json TEXT,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback(request_id);

CREATE TABLE IF NOT EXISTS evidence_cache (
  evidence_id TEXT PRIMARY KEY,
  source_uri TEXT,
  title TEXT,
  content TEXT NOT NULL,
  retrieved_at TEXT NOT NULL,
  metadata_json TEXT
);
