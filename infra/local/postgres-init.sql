-- OHI v2 — local dev schema. Mirrors spec §12
-- (docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md).

CREATE TABLE IF NOT EXISTS verifications (
    id              UUID PRIMARY KEY,
    text_hash       CHAR(64) NOT NULL,
    request_id      UUID NOT NULL UNIQUE,
    document_verdict_jsonb JSONB NOT NULL,
    model_versions_jsonb JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_verifications_text_hash ON verifications(text_hash);
CREATE INDEX IF NOT EXISTS idx_verifications_created_at ON verifications(created_at);

CREATE TABLE IF NOT EXISTS claim_verdicts (
    id              UUID PRIMARY KEY,
    verification_id UUID NOT NULL REFERENCES verifications(id) ON DELETE CASCADE,
    claim_id        UUID NOT NULL,
    claim_jsonb     JSONB NOT NULL,
    calibrated_verdict_jsonb JSONB NOT NULL,
    information_gain DOUBLE PRECISION NOT NULL,
    queued_for_review BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_claim_verdicts_claim_id ON claim_verdicts(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_verdicts_verification_id ON claim_verdicts(verification_id);

CREATE TABLE IF NOT EXISTS feedback_pending (
    id              UUID PRIMARY KEY,
    claim_id        UUID NOT NULL,
    label           TEXT NOT NULL CHECK (label IN ('true', 'false', 'unverifiable', 'abstain')),
    labeler_kind    TEXT NOT NULL CHECK (labeler_kind IN ('user', 'expert', 'adjudicator')),
    labeler_id_hash CHAR(64) NOT NULL,
    rationale       TEXT,
    evidence_corrections_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb,
    ip_hash         CHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (claim_id, labeler_id_hash, label)
);
CREATE INDEX IF NOT EXISTS idx_feedback_pending_claim_id ON feedback_pending(claim_id);
CREATE INDEX IF NOT EXISTS idx_feedback_pending_created_at ON feedback_pending(created_at);

CREATE TABLE IF NOT EXISTS calibration_set (
    id              UUID PRIMARY KEY,
    claim_id        UUID NOT NULL,
    true_label      TEXT NOT NULL CHECK (true_label IN ('true', 'false', 'unverifiable')),
    source_tier     TEXT NOT NULL CHECK (source_tier IN ('consensus', 'trusted', 'adjudicator')),
    n_concordant    INT NOT NULL,
    adjudicated_by  TEXT,
    calibration_set_partition TEXT NOT NULL,  -- "domain:claim_type"
    posterior_at_label_time DOUBLE PRECISION NOT NULL,
    model_versions_at_label_time JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    retired_at      TIMESTAMPTZ,
    UNIQUE (claim_id)
);
CREATE INDEX IF NOT EXISTS idx_calibration_partition ON calibration_set(calibration_set_partition)
    WHERE retired_at IS NULL;

CREATE TABLE IF NOT EXISTS disputed_claims_queue (
    claim_id        UUID PRIMARY KEY,
    first_disputed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ,
    resolved_by     TEXT
);

CREATE TABLE IF NOT EXISTS retraining_runs (
    id              UUID PRIMARY KEY,
    layer           TEXT NOT NULL,  -- 'L3.nli' | 'L5.conformal' | 'L1.source_cred'
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed', 'rolled_back')),
    metrics_jsonb   JSONB,
    artifact_s3_uri TEXT,
    deployed_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_retraining_runs_started_at ON retraining_runs(started_at);
