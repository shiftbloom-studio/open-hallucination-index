-- OHI v2 production schema (algorithm spec §12)
-- Runs once at first postgres container start; docker-entrypoint-initdb.d picks it up.

BEGIN;

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Verifications — one row per /verify call
CREATE TABLE IF NOT EXISTS verifications (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash               char(64) NOT NULL,          -- sha256 of input text + options
    request_id              text NOT NULL,
    document_verdict_jsonb  jsonb NOT NULL,
    model_versions_jsonb    jsonb NOT NULL,
    created_at              timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_verifications_text_hash ON verifications(text_hash);
CREATE INDEX IF NOT EXISTS idx_verifications_request_id ON verifications(request_id);
CREATE INDEX IF NOT EXISTS idx_verifications_created_at ON verifications(created_at);

-- Claim verdicts
CREATE TABLE IF NOT EXISTS claim_verdicts (
    id                          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id             uuid NOT NULL REFERENCES verifications(id) ON DELETE CASCADE,
    claim_jsonb                 jsonb NOT NULL,
    calibrated_verdict_jsonb    jsonb NOT NULL,
    information_gain            real,
    queued_for_review           boolean NOT NULL DEFAULT false,
    created_at                  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_claim_verdicts_verification_id ON claim_verdicts(verification_id);

-- Feedback (untrusted + trusted intake)
CREATE TABLE IF NOT EXISTS feedback_pending (
    id                          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id                    uuid NOT NULL REFERENCES claim_verdicts(id) ON DELETE CASCADE,
    label                       text NOT NULL,
    labeler_kind                text NOT NULL CHECK (labeler_kind IN ('user','expert','adjudicator')),
    labeler_id_hash             char(64) NOT NULL,
    rationale                   text,
    evidence_corrections_jsonb  jsonb,
    ip_hash                     char(64),
    created_at                  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_pending_claim_id ON feedback_pending(claim_id);

-- Calibration set (promoted ground truth)
CREATE TABLE IF NOT EXISTS calibration_set (
    id                              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id                        uuid NOT NULL,
    true_label                      text NOT NULL,
    source_tier                     text NOT NULL CHECK (source_tier IN ('consensus','trusted','adjudicator')),
    n_concordant                    int NOT NULL DEFAULT 1,
    adjudicated_by                  text,
    calibration_set_partition       text,
    posterior_at_label_time         real,
    model_versions_at_label_time    jsonb,
    created_at                      timestamptz NOT NULL DEFAULT now(),
    retired_at                      timestamptz
);

-- Retraining runs
CREATE TABLE IF NOT EXISTS retraining_runs (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    layer               text NOT NULL CHECK (layer IN ('L3.nli','L5.conformal','L1.source_cred')),
    started_at          timestamptz NOT NULL DEFAULT now(),
    completed_at        timestamptz,
    status              text NOT NULL CHECK (status IN ('running','ok','failed')),
    metrics_jsonb       jsonb,
    artifact_s3_uri     text,
    deployed_at         timestamptz
);

-- Disputed claims queue (referenced by algorithm §12 consensus SQL)
CREATE TABLE IF NOT EXISTS disputed_claims_queue (
    claim_id    uuid PRIMARY KEY,
    queued_at   timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);

COMMIT;
