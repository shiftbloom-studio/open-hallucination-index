# OHI v2 Algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current OHI verification engine with the v2 algorithm specified in [docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md](../specs/2026-04-16-ohi-v2-algorithm-design.md): a seven-layer pipeline with a Probabilistic Claim Graph (TRW-BP + Gibbs validation), per-domain split-conformal calibration, online active learning, and an open-access HTTP API with rate-limit + cost-ceiling traffic protection.

**Architecture:** Layered pipeline (L1 decompose+retrieve → L2 domain route → L3 NLI cross-encoder → L4 Probabilistic Claim Graph → L5 conformal → L6 active-learning hook → L7 output assembly). v1 (`oracle.py`, `scorer.py`, `/api/v1/*`) is fully removed. Each layer has a port + adapter pattern; rigor tier (`fast`/`balanced`/`maximum`) is the API knob that selects work envelope.

**Tech Stack:** Python 3.14, FastAPI, Pydantic v2, asyncio, Hugging Face Transformers (DeBERTa-v3-large MNLI + RoBERTa-large MNLI), sentence-transformers, NumPy + SciPy (TRW-BP, Gibbs, copula CDF), Postgres (RDS in prod, SQLite local), Redis (cache), S3 (or MinIO local) for artifacts, Neo4j + Qdrant (existing), MCP sources (existing).

---

## Scope notice (read first)

This is a **monolithic plan** covering all 5 phases of the v2 algorithm rewrite. Phases 0 and 1 (the immediately-actionable work) are spelled out in full TDD bite-size steps. Phases 2–4 use a focused task format (Files / Interface / Implementation outline / Tests / Acceptance / Commit) without re-spelling every "run pytest" step — the pattern is established in Phase 0 / 1 and the per-phase first task in each later phase serves as a worked example. When a phase begins execution, its tasks may be expanded with additional TDD detail in a follow-up.

**Out-of-scope** (covered by separate sub-projects):
- AWS infrastructure with Terraform — sub-project 2.
- Next.js frontend rewrite — sub-project 3.
- Token-level streaming verification, `/rewrite` auto-correction, cross-language support — explicit non-goals per spec §1.

**Phase ordering is enforced by acceptance gates.** Do not begin Phase N+1 until Phase N's acceptance test passes (see "Acceptance gates" at the end of this plan). The acceptance gate is a CI job, not a vibe check.

---

## Cross-cutting conventions

These apply to every task in this plan unless explicitly overridden.

- **TDD discipline.** Test first, watch it fail, write minimal code to pass, watch it pass, refactor if needed, commit. See @superpowers:test-driven-development.
- **Verification before completion.** Never mark a task done until the full test suite (unit + integration + relevant benchmark) passes. See @superpowers:verification-before-completion.
- **Frequent commits.** One commit per task minimum; one commit per logical step within a task is welcome. Conventional Commits format: `feat(L3): ...`, `test(L4.pcg): ...`, `chore(infra): ...`, `docs(spec): ...`.
- **Type discipline.** Strict mypy (already configured in `src/api/pyproject.toml`). Every new function has type hints. No `Any` without an explanatory comment.
- **Lint discipline.** `ruff format .` then `ruff check .` before every commit. Pre-commit hook (already in `.pre-commit-config.yaml`) enforces.
- **No emojis in code or docs.**
- **Module boundaries.** New modules live under `src/api/pipeline/` per the spec module layout (§2). Each module exports its port from `__init__.py`; concrete adapters live in subdirectories.
- **Imports.** Absolute imports only (`from pipeline.nli.cross_encoder import ...`).
- **Async by default.** Every IO-bound operation is `async`; all pipeline orchestration uses `asyncio`.
- **Logging.** `logging.getLogger(__name__)` per module; INFO for normal operation, DEBUG for hot paths, WARNING for graceful degradation, ERROR for failures.
- **No `print` statements** outside dedicated CLI scripts.
- **Reading before editing.** Read the file with the Read tool before editing it. Read the relevant spec section before implementing it.
- **Directory:** all commands assumed run from repo root (`/c/Users/Fabia/Documents/shiftbloom/git/open-hallucination-index`) with the API venv active (`source .venv/Scripts/activate` on Git Bash for Windows, or `pip install -e "src/api[dev]"` already done).

---

## File structure (full plan)

This is the target tree after all 5 phases. Files marked **NEW** are created by this plan; **MODIFIED** are existing files we touch; **DELETED** is what goes away when v1 is removed.

```
src/api/
├── pipeline/
│   ├── __init__.py                                NEW (exports orchestrator + ports)
│   ├── pipeline.py                                NEW (L1→L7 orchestrator)
│   │
│   ├── decomposer.py                              MODIFIED (chunked decomp)
│   ├── retrieval/                                 NEW package
│   │   ├── __init__.py                            NEW
│   │   ├── router.py                              MOVED from pipeline/router.py
│   │   ├── collector.py                           MOVED from pipeline/collector.py
│   │   ├── selector.py                            MOVED from pipeline/selector.py
│   │   ├── source_credibility.py                  NEW
│   │   └── mesh.py                                MOVED from pipeline/mesh.py
│   │
│   ├── domain/                                    NEW (Phase 3)
│   │   ├── __init__.py                            NEW
│   │   ├── router.py                              NEW
│   │   ├── registry.py                            NEW
│   │   └── adapters/
│   │       ├── __init__.py                        NEW
│   │       ├── base.py                            NEW (DomainAdapter ABC)
│   │       ├── general.py                         NEW
│   │       ├── biomedical.py                      NEW
│   │       ├── legal.py                           NEW
│   │       ├── code.py                            NEW
│   │       └── social.py                          NEW
│   │
│   ├── nli/                                       NEW (Phase 2)
│   │   ├── __init__.py                            NEW
│   │   ├── cross_encoder.py                       NEW
│   │   ├── self_consistency.py                    NEW
│   │   ├── ensemble.py                            NEW (Phase 3 maximum-tier)
│   │   ├── batching.py                            NEW
│   │   ├── bi_encoder_filter.py                   NEW
│   │   └── paraphrase.py                          NEW (T5 paraphrase cache)
│   │
│   ├── pcg/                                       NEW (Phase 2)
│   │   ├── __init__.py                            NEW
│   │   ├── graph.py                               NEW (Ising construction)
│   │   ├── potentials.py                          NEW (unary α_c, edge J_ij)
│   │   ├── trw_bp.py                              NEW (TRW-BP inference)
│   │   ├── lbp.py                                 NEW (loopy BP fallback)
│   │   ├── gibbs.py                               NEW (MCMC sanity check)
│   │   └── refinement.py                          NEW (iterative refinement loop)
│   │
│   ├── conformal/                                 NEW (Phase 1 stub, Phase 2 fill, Phase 3 per-domain)
│   │   ├── __init__.py                            NEW
│   │   ├── split_conformal.py                     NEW
│   │   ├── mondrian.py                            NEW (Phase 3)
│   │   ├── mixture.py                             NEW (Phase 3, weighted exchangeability)
│   │   └── calibration_store.py                   NEW
│   │
│   ├── active_learning/                           NEW (Phase 4)
│   │   ├── __init__.py                            NEW
│   │   ├── information_gain.py                    NEW
│   │   ├── review_queue.py                        NEW
│   │   ├── feedback_store.py                      NEW
│   │   ├── consensus.py                           NEW (15-min promotion job)
│   │   └── retrainer.py                           NEW (nightly DAG entrypoint)
│   │
│   ├── assembly/                                  NEW (Phase 1)
│   │   ├── __init__.py                            NEW
│   │   ├── claim_verdict.py                       NEW (L7 ClaimVerdict builder)
│   │   ├── document_verdict.py                    NEW (L7 DocumentVerdict builder)
│   │   └── copula.py                              NEW (Gaussian copula joint prob)
│   │
│   ├── oracle.py                                  DELETED (v1)
│   └── scorer.py                                  DELETED (v1)
│
├── interfaces/
│   ├── decomposition.py                           MODIFIED (refined port)
│   ├── verification.py                            DELETED (v1 oracle port)
│   ├── scoring.py                                 DELETED (v1 scorer port)
│   ├── nli.py                                     NEW
│   ├── domain.py                                  NEW
│   ├── pcg.py                                     NEW
│   ├── conformal.py                               NEW
│   └── feedback.py                                NEW
│
├── models/
│   ├── entities.py                                MODIFIED (Claim normalization fields)
│   ├── results.py                                 REWRITTEN (new ClaimVerdict, DocumentVerdict)
│   ├── nli.py                                     NEW (NLIDistribution dataclass)
│   ├── pcg.py                                     NEW (PosteriorBelief, ClaimEdge)
│   └── feedback.py                                NEW (FeedbackSubmission, CalibrationEntry)
│
├── server/
│   ├── app.py                                     MODIFIED (mount /api/v2, drop /api/v1)
│   ├── routes/
│   │   ├── verify.py                              REWRITTEN (POST /verify, GET /verdict/{id})
│   │   ├── stream.py                              NEW (SSE /verify/stream)
│   │   ├── feedback.py                            NEW (POST /feedback)
│   │   ├── calibration.py                         NEW (GET /calibration/report)
│   │   ├── health.py                              MODIFIED (add /deep)
│   │   └── (any v1 routes)                        DELETED
│   ├── middleware/
│   │   ├── rate_limit.py                          NEW (per-IP token bucket)
│   │   ├── cost_ceiling.py                        NEW (daily $ ceiling)
│   │   ├── retention.py                           NEW (raw-text-not-persisted enforcement)
│   │   └── internal_auth.py                       NEW (bearer token for trusted callers)
│   └── schemas/
│       ├── verify.py                              REWRITTEN
│       ├── feedback.py                            NEW
│       ├── calibration.py                         NEW
│       └── (any v1 schemas)                       DELETED
│
├── adapters/
│   ├── (existing adapters)                        UNCHANGED (Neo4j, Qdrant, MCP, Redis, OpenAI, embeddings)
│   ├── nli_huggingface.py                         NEW (HF Transformers NLI impl)
│   ├── postgres_feedback.py                       NEW
│   ├── s3_artifacts.py                            NEW
│   └── domain_router_distilbert.py                NEW
│
├── config/
│   ├── settings.py                                MODIFIED (add v2 settings)
│   └── dependencies.py                            MODIFIED (DI wiring for v2)
│
└── pyproject.toml                                 MODIFIED (new deps)

scripts/                                           NEW top-level
├── benchmark/
│   ├── run_baseline.py                            NEW (Phase 0 v1 baseline capture)
│   ├── run_v2_benchmarks.py                       NEW (per-phase F1 measurement)
│   └── compare.py                                 NEW (vs baseline comparison)
├── retraining/
│   ├── nightly_dag.py                             NEW (Phase 4 retraining entrypoint)
│   └── rescore_calibration.py                     NEW (Stage 3 of nightly DAG)
└── ops/
    ├── ohi_models.py                              NEW (CLI: rollback, status)
    └── disputed_claims.py                         NEW (CLI: list, adjudicate)

infra/local/                                       NEW (Phase 0 local dev)
├── docker-compose.dev.yml                         NEW (Postgres + MinIO + existing)
├── postgres-init.sql                              NEW (schemas from spec §12)
└── minio-init.sh                                  NEW (S3 buckets)

docs/
├── algorithm/
│   ├── v2-overview.md                             NEW (Phase 1 deliverable)
│   └── v2-calibration.md                          NEW (Phase 4 deliverable)
├── api/
│   └── v2-reference.md                            NEW (regenerated from FastAPI; Phase 1)
└── operations/
    └── active-learning-runbook.md                 NEW (Phase 4)
```

---

## Phase 0 — Research foundation

**Goal:** Stand up the testing + measurement infrastructure that everything else depends on. Capture v1 baselines as the bar all phases must clear. Nothing user-facing ships in Phase 0.

**Acceptance gate:** `benchmark_results/v1_baseline_<date>.jsonl` is committed and contains scored runs for FActScore, TruthfulQA, HaluEval, and at least one domain benchmark per planned vertical.

### Task 0.1: Local Postgres + MinIO via docker-compose

**Files:**
- Create: `infra/local/docker-compose.dev.yml`
- Create: `infra/local/postgres-init.sql`
- Create: `infra/local/minio-init.sh`
- Create: `infra/local/README.md`
- Modify: `.env.example` (add `POSTGRES_*`, `S3_*` vars)
- Modify: `.gitignore` (ignore `infra/local/data/`)

**Worked example — full TDD bite-size steps for this task:**

- [ ] **Step 1: Write the failing test**

Create `tests/infra/test_local_stack.py`:

```python
"""Smoke tests for the local Phase 0 dev stack (Postgres + MinIO)."""
from __future__ import annotations

import os
import psycopg
import pytest
from minio import Minio


pytestmark = pytest.mark.infra  # opt-in marker; CI gates this on docker availability


def test_postgres_reachable_with_required_schemas():
    """Postgres is reachable and the spec §12 tables exist after init."""
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "ohi"),
        user=os.environ.get("POSTGRES_USER", "ohi"),
        password=os.environ.get("POSTGRES_PASSWORD", "ohi-local-dev"),
    )
    with conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name;"
        )
        tables = {row[0] for row in cur.fetchall()}
    expected = {
        "verifications",
        "claim_verdicts",
        "feedback_pending",
        "calibration_set",
        "disputed_claims_queue",
        "retraining_runs",
    }
    missing = expected - tables
    assert not missing, f"Missing tables: {missing}"


def test_minio_reachable_with_required_buckets():
    """MinIO is reachable and the artifact bucket exists."""
    client = Minio(
        os.environ.get("S3_ENDPOINT", "localhost:9000"),
        access_key=os.environ.get("S3_ACCESS_KEY", "ohi-local"),
        secret_key=os.environ.get("S3_SECRET_KEY", "ohi-local-dev-key"),
        secure=False,
    )
    assert client.bucket_exists("ohi-artifacts"), "ohi-artifacts bucket missing"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/infra/test_local_stack.py -v -m infra
```
Expected: FAIL with connection refused (Postgres not running yet).

- [ ] **Step 3: Write `infra/local/docker-compose.dev.yml`**

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16-alpine
    container_name: ohi-postgres-dev
    environment:
      POSTGRES_USER: ohi
      POSTGRES_PASSWORD: ohi-local-dev
      POSTGRES_DB: ohi
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./postgres-init.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ohi -d ohi"]
      interval: 5s
      timeout: 3s
      retries: 10

  minio:
    image: minio/minio:latest
    container_name: ohi-minio-dev
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ohi-local
      MINIO_ROOT_PASSWORD: ohi-local-dev-key
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./data/minio:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 3s
      retries: 10

  minio-init:
    image: minio/mc:latest
    container_name: ohi-minio-init
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: ["/bin/sh", "/init/minio-init.sh"]
    volumes:
      - ./minio-init.sh:/init/minio-init.sh:ro
```

- [ ] **Step 4: Write `infra/local/postgres-init.sql`**

This schema mirrors spec §12. Field names and types align with `models/feedback.py` (Phase 4) so Postgres is the source of truth from day one.

```sql
-- OHI v2 — local dev schema. Mirrors spec §12.

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
```

- [ ] **Step 5: Write `infra/local/minio-init.sh`**

```sh
#!/bin/sh
set -eu
mc alias set local http://minio:9000 ohi-local ohi-local-dev-key
mc mb --ignore-existing local/ohi-artifacts
mc mb --ignore-existing local/ohi-artifacts/nli-heads
mc mb --ignore-existing local/ohi-artifacts/calibration
mc mb --ignore-existing local/ohi-artifacts/source-cred
mc mb --ignore-existing local/ohi-artifacts/retraining-reports
mc mb --ignore-existing local/ohi-artifacts/eval-snapshots
echo "MinIO buckets initialised."
```

- [ ] **Step 6: Add envs to `.env.example` and ignore data dir**

Append to `.env.example`:

```
# Phase 0 local dev stack
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ohi
POSTGRES_USER=ohi
POSTGRES_PASSWORD=ohi-local-dev
S3_ENDPOINT=localhost:9000
S3_ACCESS_KEY=ohi-local
S3_SECRET_KEY=ohi-local-dev-key
S3_BUCKET=ohi-artifacts
```

Append to `.gitignore`:

```
infra/local/data/
```

- [ ] **Step 7: Add `psycopg` and `minio` to dev deps**

Edit `src/api/pyproject.toml` `[project.optional-dependencies] dev` list, add:

```
"psycopg[binary]>=3.2",
"minio>=7.2",
```

Then `pip install -e "src/api[dev]"`.

- [ ] **Step 8: Bring stack up and re-run test**

```
docker compose -f infra/local/docker-compose.dev.yml up -d
# wait ~10s for healthchecks
pytest tests/infra/test_local_stack.py -v -m infra
```
Expected: PASS (both tests).

- [ ] **Step 9: Write `infra/local/README.md`**

Short doc: how to bring up, tear down (`docker compose -f infra/local/docker-compose.dev.yml down`, optional `-v` to drop volumes), connection details, links to spec §12.

- [ ] **Step 10: Commit**

```bash
git add infra/local/ tests/infra/ src/api/pyproject.toml .env.example .gitignore
git commit -m "feat(infra): local Postgres + MinIO dev stack with spec §12 schemas

Brings up Postgres 16 + MinIO via docker-compose for Phase 0 development.
Postgres is initialised with the v2 tables (verifications, claim_verdicts,
feedback_pending, calibration_set, disputed_claims_queue, retraining_runs).
MinIO has the ohi-artifacts bucket pre-created.

Smoke test in tests/infra/test_local_stack.py asserts both services are
reachable and schemas/buckets exist."
```

---

### Task 0.2: Benchmark harness skeleton

**Files:**
- Create: `scripts/benchmark/__init__.py`
- Create: `scripts/benchmark/datasets.py` (dataset loaders for FActScore, TruthfulQA, HaluEval, PubMedQA, LegalBench-Entailment, LIAR)
- Create: `scripts/benchmark/runner.py` (engine-agnostic benchmark runner)
- Create: `scripts/benchmark/metrics.py` (F1, accuracy, ECE, calibration coverage)
- Create: `scripts/benchmark/run_baseline.py` (Phase 0 entrypoint: runs v1 against all benchmarks)
- Create: `tests/scripts/benchmark/test_runner.py`
- Create: `tests/scripts/benchmark/test_metrics.py`
- Create: `benchmark_results/.gitkeep`

**Interface:**

```python
# scripts/benchmark/runner.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    text: str
    expected_label: str   # benchmark-specific encoding
    metadata: dict

@dataclass(frozen=True)
class BenchmarkResult:
    example_id: str
    predicted_label: str
    p_true: float | None        # for calibration metrics
    interval: tuple[float, float] | None
    raw_response: dict          # full DocumentVerdict for v2; legacy shape for v1
    latency_ms: float
    error: str | None

class VerificationEngine(Protocol):
    name: str                   # "v1" | "v2-phase1" | etc.
    async def verify(self, example: BenchmarkExample) -> BenchmarkResult: ...

async def run_benchmark(
    engine: VerificationEngine,
    examples: list[BenchmarkExample],
    *,
    concurrency: int = 4,
    output_path: Path,
) -> Path:
    """Run engine over examples, write JSONL results to output_path. Returns the path."""
```

**Implementation outline:**

- `datasets.py` defines a `load_<name>()` async generator per benchmark, yielding `BenchmarkExample`. Datasets fetched from Hugging Face (`datasets` library) where possible, cached locally under `~/.cache/ohi-benchmarks/`. For benchmarks where atomic-claim ground truth is needed (FActScore), use the official atomic-fact split.
- `runner.py` runs `engine.verify` over examples with bounded concurrency (`asyncio.Semaphore`), tracks per-example latency, writes one JSON object per line to `output_path`. Atomic write (write to `.tmp`, rename).
- `metrics.py`: `compute_f1(results, get_predicted_label, get_true_label)`, `compute_ece(results, n_bins=10)`, `compute_calibration_coverage(results, target=0.90)`, `aggregate_per_domain(results, get_domain)`.
- `run_baseline.py` wires the existing v1 engine through a `V1EngineAdapter` and runs all configured datasets, writing `benchmark_results/v1_baseline_YYYY-MM-DD.jsonl` and a per-benchmark summary `benchmark_results/v1_baseline_YYYY-MM-DD.summary.json`.

**Tests:** unit tests on `compute_f1`, `compute_ece`, `compute_calibration_coverage` with known inputs/expected outputs. Integration test for `run_benchmark` with a mock engine and 10 synthetic examples (asserts JSONL round-trip + aggregate metrics).

**Acceptance:** `pytest tests/scripts/benchmark/ -v` passes. `python -m scripts.benchmark.run_baseline --datasets factscore --limit 50` produces a non-empty JSONL file in `benchmark_results/`.

**Commit:** `feat(benchmark): scaffold engine-agnostic benchmark harness with metrics`

---

### Task 0.3: Capture v1 baseline numbers

**Files:**
- Create: `scripts/benchmark/v1_engine_adapter.py` (wraps current `oracle.py` + `scorer.py` so the runner can call v1 through the engine protocol)
- Create: `benchmark_results/v1_baseline_<date>.jsonl` (output)
- Create: `benchmark_results/v1_baseline_<date>.summary.json` (output)
- Create: `docs/algorithm/v1-baseline.md` (human-readable summary, embedded in repo)

**Implementation outline:**

- `V1EngineAdapter` constructs the v1 verification stack via existing `dependencies.get_verify_service()`, wraps `verify_text(...)` in the `VerificationEngine.verify` protocol, maps the v1 `VerificationResult` into a `BenchmarkResult` (legacy `overall` becomes `p_true`; no interval).
- Run all six benchmarks (FActScore-atomic, TruthfulQA, HaluEval, PubMedQA, LegalBench-Entailment, LIAR) with `--limit 1000` each (sufficient for stable F1 estimates; full dataset in CI later).
- Generate `summary.json` with per-benchmark macro-F1 + mean latency + ECE (where labels permit).
- Generate `docs/algorithm/v1-baseline.md` with a markdown table of results, capture date, hardware, model versions of v1's LLM provider.

**Tests:** there is no unit test for "the baseline numbers are correct"; this task is a measurement task. Sanity check: F1 on FActScore should be > 0 (catches engine wiring bugs). Summary file must exist and be valid JSON.

**Acceptance:** `python -m scripts.benchmark.run_baseline --all --limit 1000` exits 0; both `.jsonl` and `.summary.json` exist and are committed; markdown summary committed.

**Commit:** `chore(baseline): capture v1 baseline numbers across 6 benchmarks (n=1000 each)`

---

### Task 0.4: Phase 0 acceptance gate in CI

**Files:**
- Modify: `.github/workflows/ci.yml` (add `benchmark-baseline-presence` job)
- Create: `scripts/ci/check_baseline_present.py` (asserts a baseline file is committed under `benchmark_results/v1_baseline_*.summary.json`)

**Implementation outline:**

- New CI job `benchmark-baseline-presence`: runs the check script. Fails the build if no v1 baseline summary file exists. This prevents Phase 1 from "passing" CI without the comparator we need.
- The script picks the latest baseline file by filename date, asserts it has entries for all six benchmarks, and prints the table.

**Tests:** `pytest tests/scripts/ci/test_check_baseline_present.py` covers happy path + missing-file path.

**Acceptance:** CI green on a PR that includes the baseline; CI red on a PR that deletes the baseline. Manually tested.

**Commit:** `chore(ci): require v1 baseline presence as Phase 0 gate`

---

## Phase 1 — Foundation (replaces v1)

**Goal:** Ship the new pipeline shell — chunked decomposer (L1), output assembly (L7), naive single-quantile L5 emitting `coverage_target=null`, new `/api/v2/*` routes, rate limiting + cost ceiling middleware, SSE streaming endpoint, retention enforcement. Delete v1 entirely. End state: API responds with v2-shape verdicts that are at parity-or-better with v1 on FActScore + TruthfulQA.

**Acceptance gate:** `python -m scripts.benchmark.run_v2_benchmarks --engine v2-phase1 --datasets factscore truthfulqa` produces F1 ≥ baseline F1 on both benchmarks (within ±1pt is acceptable; the headline contributions land in Phase 2). All v1 routes return 404. All v1 schemas removed. All Phase 1 tests pass.

### Task 1.1: New result models (worked example for Phase 1)

**Files:**
- Modify: `src/api/models/results.py` (rewrite)
- Create: `src/api/models/nli.py`
- Create: `src/api/models/pcg.py`
- Create: `tests/api/models/test_results_v2.py`

**Worked example — full TDD bite-size steps:**

- [ ] **Step 1: Write the failing tests**

Create `tests/api/models/test_results_v2.py`:

```python
"""Tests for the v2 result models (ClaimVerdict, DocumentVerdict)."""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from models.entities import Claim, ClaimType
from models.results import (
    ClaimEdge,
    ClaimVerdict,
    DocumentVerdict,
    EdgeType,
)


def _claim() -> Claim:
    return Claim(
        id=uuid4(),
        text="Einstein was born in 1879.",
        claim_type=ClaimType.TEMPORAL,
        confidence=0.9,
    )


def test_claim_verdict_accepts_null_coverage_target_when_fallback():
    cv = ClaimVerdict(
        claim=_claim(),
        p_true=0.96,
        interval=(0.91, 0.99),
        coverage_target=None,
        domain="general",
        domain_assignment_weights={"general": 1.0},
        supporting_evidence=[],
        refuting_evidence=[],
        pcg_neighbors=[],
        nli_self_consistency_variance=0.012,
        bp_validated=None,
        information_gain=0.04,
        queued_for_review=False,
        calibration_set_id=None,
        calibration_n=0,
        fallback_used="general",
    )
    assert cv.coverage_target is None
    assert cv.fallback_used == "general"


def test_claim_verdict_rejects_invalid_p_true():
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim=_claim(),
            p_true=1.5,  # invalid
            interval=(0.0, 1.0),
            coverage_target=None,
            domain="general",
            domain_assignment_weights={"general": 1.0},
            supporting_evidence=[],
            refuting_evidence=[],
            pcg_neighbors=[],
            nli_self_consistency_variance=0.0,
            bp_validated=None,
            information_gain=0.0,
            queued_for_review=False,
            calibration_set_id=None,
            calibration_n=0,
            fallback_used="general",
        )


def test_claim_verdict_rejects_inverted_interval():
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim=_claim(),
            p_true=0.5,
            interval=(0.7, 0.3),  # upper < lower
            coverage_target=0.9,
            domain="general",
            domain_assignment_weights={"general": 1.0},
            supporting_evidence=[],
            refuting_evidence=[],
            pcg_neighbors=[],
            nli_self_consistency_variance=0.0,
            bp_validated=True,
            information_gain=0.0,
            queued_for_review=False,
            calibration_set_id="c1",
            calibration_n=200,
            fallback_used=None,
        )


def test_claim_edge_serialization_round_trip():
    edge = ClaimEdge(
        neighbor_claim_id=uuid4(),
        edge_type=EdgeType.ENTAIL,
        edge_strength=0.81,
    )
    payload = edge.model_dump_json()
    parsed = ClaimEdge.model_validate_json(payload)
    assert parsed == edge


def test_document_verdict_min_required_fields():
    dv = DocumentVerdict(
        document_score=0.74,
        document_interval=(0.61, 0.84),
        internal_consistency=0.83,
        claims=[],
        decomposition_coverage=0.0,
        processing_time_ms=87341.0,
        rigor="balanced",
        refinement_passes_executed=0,
        model_versions={"decomposer": "v1", "router": "n/a"},
        request_id=uuid4(),
    )
    assert dv.pipeline_version == "ohi-v2.0"
    assert dv.rigor == "balanced"


def test_document_verdict_freeze_immutable():
    dv = DocumentVerdict(
        document_score=0.5,
        document_interval=(0.0, 1.0),
        internal_consistency=0.0,
        claims=[],
        decomposition_coverage=0.0,
        processing_time_ms=0.0,
        rigor="fast",
        refinement_passes_executed=0,
        model_versions={},
        request_id=uuid4(),
    )
    with pytest.raises(ValidationError):
        dv.document_score = 0.6  # frozen
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/api/models/test_results_v2.py -v
```
Expected: FAIL on import (no `ClaimVerdict`, `DocumentVerdict`, `ClaimEdge`, `EdgeType` in `models/results.py` yet).

- [ ] **Step 3: Rewrite `src/api/models/results.py`**

Replace its contents entirely (this is the v2 contract):

```python
"""
v2 Result Models
================

Frozen, immutable Pydantic models for the v2 verification API. Replaces the
v1 VerificationResult / TrustScore / ClaimVerification trio.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from models.entities import Claim, Evidence


# ---------------------------------------------------------------------------
# Edge types in the Probabilistic Claim Graph (Phase 2 fills these in;
# Phase 1 emits empty pcg_neighbors lists).
# ---------------------------------------------------------------------------

class EdgeType(StrEnum):
    ENTAIL = auto()
    CONTRADICT = auto()


class ClaimEdge(BaseModel):
    neighbor_claim_id: UUID
    edge_type: EdgeType
    edge_strength: float = Field(..., ge=0.0, le=1.0)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# ClaimVerdict — the per-claim public output. See spec §9.
# ---------------------------------------------------------------------------

class ClaimVerdict(BaseModel):
    claim: Claim

    # Calibrated probability + conformal interval
    p_true: float = Field(..., ge=0.0, le=1.0)
    interval: tuple[float, float]
    coverage_target: float | None = Field(default=None, ge=0.0, le=1.0)

    # Provenance & explainability
    domain: str
    domain_assignment_weights: dict[str, float]
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    refuting_evidence: list[Evidence] = Field(default_factory=list)
    pcg_neighbors: list[ClaimEdge] = Field(default_factory=list)
    nli_self_consistency_variance: float = Field(..., ge=0.0)
    bp_validated: bool | None = None  # None when Gibbs skipped (benign graph)

    # Active learning
    information_gain: float = Field(..., ge=0.0)
    queued_for_review: bool = False

    # Calibration metadata
    calibration_set_id: str | None = None
    calibration_n: int = Field(..., ge=0)
    fallback_used: Literal["domain", "general", "non_converged"] | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_interval(self) -> "ClaimVerdict":
        lo, hi = self.interval
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"interval must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
        return self


# ---------------------------------------------------------------------------
# DocumentVerdict — top-level public output. See spec §9.
# ---------------------------------------------------------------------------

class DocumentVerdict(BaseModel):
    document_score: float = Field(..., ge=0.0, le=1.0)
    document_interval: tuple[float, float]
    internal_consistency: float = Field(..., ge=0.0, le=1.0)
    claims: list[ClaimVerdict] = Field(default_factory=list)
    decomposition_coverage: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float = Field(..., ge=0.0)
    rigor: Literal["fast", "balanced", "maximum"]
    refinement_passes_executed: int = Field(..., ge=0)
    pipeline_version: str = Field(default="ohi-v2.0")
    model_versions: dict[str, str] = Field(default_factory=dict)
    request_id: UUID

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_doc_interval(self) -> "DocumentVerdict":
        lo, hi = self.document_interval
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"document_interval must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
        return self
```

- [ ] **Step 4: Create `src/api/models/nli.py`**

```python
"""Frozen value object for L3 NLI distributions. See spec §5."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class NLIDistribution(BaseModel):
    entail: float = Field(..., ge=0.0, le=1.0)
    contradict: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    variance: float = Field(..., ge=0.0)
    nli_model_id: str

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _sums_to_one(self) -> "NLIDistribution":
        s = self.entail + self.contradict + self.neutral
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"entail+contradict+neutral must be ~1, got {s}")
        return self
```

- [ ] **Step 5: Create `src/api/models/pcg.py`**

```python
"""Frozen value objects for L4 Probabilistic Claim Graph. See spec §6."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PosteriorBelief(BaseModel):
    p_true: float = Field(..., ge=0.0, le=1.0)
    p_false: float = Field(..., ge=0.0, le=1.0)
    converged: bool
    algorithm: Literal["TRW-BP", "LBP-fallback", "LBP-nonconvergent"]
    iterations: int = Field(..., ge=0)
    edge_count: int = Field(..., ge=0)
    log_partition_bound: float | None = None  # Only meaningful for TRW-BP

    model_config = {"frozen": True}
```

- [ ] **Step 6: Run tests to verify they pass**

```
pytest tests/api/models/test_results_v2.py -v
ruff format src/api/models/ tests/api/models/
ruff check src/api/models/ tests/api/models/
mypy src/api/models/
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/api/models/results.py src/api/models/nli.py src/api/models/pcg.py tests/api/models/test_results_v2.py
git commit -m "feat(models): v2 result schemas — ClaimVerdict, DocumentVerdict, NLI, PCG

Replaces the v1 VerificationResult / TrustScore / ClaimVerification trio
with the v2 contract from spec §9. Frozen Pydantic models with field
validators that enforce interval ordering and probability sums.

NLIDistribution and PosteriorBelief are new value objects used by L3 / L4
in subsequent tasks."
```

---

### Task 1.2: New port interfaces (NLI, Domain, PCG, Conformal, Feedback)

**Files:**
- Create: `src/api/interfaces/nli.py`
- Create: `src/api/interfaces/domain.py`
- Create: `src/api/interfaces/pcg.py`
- Create: `src/api/interfaces/conformal.py`
- Create: `src/api/interfaces/feedback.py`
- Modify: `src/api/interfaces/decomposition.py` (refine signature; add `chunked` flag)
- Delete: `src/api/interfaces/verification.py` (v1 oracle port)
- Delete: `src/api/interfaces/scoring.py` (v1 scorer port)
- Create: `tests/api/interfaces/test_ports_importable.py`

**Interface (key signatures):**

```python
# interfaces/nli.py
class NLIService(Protocol):
    async def claim_evidence(
        self, claim: Claim, evidence: list[Evidence], adapter: DomainAdapter
    ) -> list[NLIDistribution]: ...
    async def claim_claim(
        self, claims: list[Claim], adapter: DomainAdapter
    ) -> dict[tuple[ClaimId, ClaimId], NLIDistribution]: ...
    async def health_check(self) -> bool: ...

# interfaces/domain.py
class DomainAdapter(Protocol):
    domain: str
    def nli_model_id(self) -> str: ...
    def source_credibility(self) -> dict[str, float]: ...
    def calibration_set_id(self) -> str: ...
    def decomposition_hints(self) -> str | None: ...
    def claim_pair_relatedness_threshold(self) -> float: ...

class DomainRouter(Protocol):
    async def route(self, claim: Claim) -> "DomainAssignment": ...

# interfaces/pcg.py
class PCGInferenceService(Protocol):
    async def infer(
        self,
        claims: list[Claim],
        evidence_per_claim: dict[ClaimId, list[Evidence]],
        nli_ce: list[list[NLIDistribution]],     # claim×evidence
        nli_cc: dict[tuple[ClaimId, ClaimId], NLIDistribution],
        adapter_per_claim: dict[ClaimId, DomainAdapter],
        rigor: Literal["fast", "balanced", "maximum"],
    ) -> dict[ClaimId, PosteriorBelief]: ...

# interfaces/conformal.py
class ConformalCalibrator(Protocol):
    async def calibrate(
        self,
        claim: Claim,
        belief: PosteriorBelief,
        domain: str,
        stratum: str,
    ) -> "CalibratedVerdict": ...

# interfaces/feedback.py
class FeedbackStore(Protocol):
    async def submit(self, submission: "FeedbackSubmission") -> "FeedbackId": ...
    async def promote_consensus(self) -> int: ...   # returns # promoted
    async def get_calibration_set(self, partition: str) -> list["CalibrationEntry"]: ...
```

**Implementation outline:** these are Protocol classes (PEP 544 structural typing). No concrete implementations in this task — those land in the per-layer tasks. Move shared frozen value objects (`DomainAssignment`, `CalibratedVerdict`, `FeedbackSubmission`, `CalibrationEntry`) into `models/feedback.py` and `models/domain.py`; created in this task because the protocols reference them.

**Tests:** `test_ports_importable.py` imports every protocol and asserts it has the expected `__protocol_attrs__`. This is a regression-protection test against accidental signature drift.

**Acceptance:** `pytest tests/api/interfaces/ -v` passes; `mypy src/api/interfaces/` passes.

**Commit:** `feat(interfaces): v2 ports for NLI, Domain, PCG, Conformal, Feedback; remove v1 oracle/scoring ports`

---

### Task 1.3: L1 — chunked decomposition refactor

**Files:**
- Modify: `src/api/pipeline/decomposer.py` (add chunking, coverage re-prompt, normalization)
- Create: `src/api/pipeline/decomposer_chunking.py` (paragraph splitter with co-ref overlap)
- Create: `src/api/pipeline/decomposer_normalization.py` (date ISO, number SI, entity QID lookup hook)
- Modify: `src/api/models/entities.py` (add `Claim.normalized_form`, `Claim.entity_qids`)
- Create: `tests/api/pipeline/test_decomposer_chunking.py`
- Create: `tests/api/pipeline/test_decomposer_normalization.py`
- Create: `tests/api/pipeline/test_decomposer_chunked_integration.py`

**Interface:**

```python
class LLMClaimDecomposer(DecompositionService):
    async def decompose(
        self,
        text: str,
        *,
        context: str | None = None,
        max_claims: int = 50,
    ) -> list[Claim]:
        chunks = chunk_text(text, max_tokens=1800, overlap_tokens=200)
        per_chunk_claims: list[list[Claim]] = await asyncio.gather(
            *[self._decompose_chunk(c, context) for c in chunks]
        )
        merged = dedupe_and_merge(per_chunk_claims, sim_threshold=0.92)
        if self._coverage_below_threshold(merged, text):
            additional = await self._reprompt_for_missed(text, merged, context)
            merged = dedupe_and_merge([merged, additional], sim_threshold=0.92)
        return [normalize_claim(c) for c in merged[:max_claims]]
```

**Implementation outline:**

- `chunk_text` splits at paragraph boundaries (double-newline) and falls back to sentence boundaries within long paragraphs. Tracks original character offsets so `Claim.source_span` survives chunking. Uses `tiktoken` (or HF tokenizer) for accurate token counting.
- `dedupe_and_merge` uses `sentence-transformers/all-MiniLM-L6-v2` (already used in retrieval) for cosine similarity; falls back to normalized-form exact match if the encoder isn't available.
- `coverage_below_threshold`: True iff `len(claims) / sentence_count(text) < 0.4` AND POS-tag heuristic indicates fact-density. Use `nltk.pos_tag` (cached) — fact-dense = >40% nouns + numbers + proper nouns.
- `_reprompt_for_missed` issues one follow-up LLM call with prior claims listed; capped at one re-prompt.
- `normalize_claim` runs in this order: date normalization (regex → `dateutil.parser` → ISO 8601 string), number normalization (regex for "X million", "X%" → SI base unit), entity QID lookup (best-effort via Wikidata MCP if available; skipped silently if not).

**Tests:**
- `test_decomposer_chunking.py`: synthetic long text (10 paragraphs), assert chunks ≤ 1800 tokens, assert overlap windows present, assert `source_span` continuity preserved.
- `test_decomposer_normalization.py`: parametrized cases for "January 5, 1879" → "1879-01-05", "5 million" → "5000000", "5%" → "0.05".
- `test_decomposer_chunked_integration.py`: 5-paragraph fact-dense text; assert ≥ 8 claims extracted; assert no duplicates; assert all `Claim.normalized_form` populated.

**Acceptance:** `pytest tests/api/pipeline/test_decomposer*.py -v` passes; integration test against the local LLM (or skipped with marker if unavailable) extracts at least 8 claims from a fixture text.

**Commit:** `feat(L1): chunked decomposition with coverage re-prompt and entity normalization`

---

### Task 1.4: L1 — split retrieval into its own package

**Files:**
- Create: `src/api/pipeline/retrieval/__init__.py`
- Move: `src/api/pipeline/router.py` → `src/api/pipeline/retrieval/router.py`
- Move: `src/api/pipeline/collector.py` → `src/api/pipeline/retrieval/collector.py`
- Move: `src/api/pipeline/selector.py` → `src/api/pipeline/retrieval/selector.py`
- Move: `src/api/pipeline/mesh.py` → `src/api/pipeline/retrieval/mesh.py`
- Create: `src/api/pipeline/retrieval/source_credibility.py`
- Modify: `src/api/models/entities.py` (add `Evidence.source_credibility`, `Evidence.temporal_decay_factor`, `Evidence.fingerprint`)
- Modify: every importer of the moved modules (use `git grep "from pipeline\\.\\(router\\|collector\\|selector\\|mesh\\)"` to find callsites)
- Create: `tests/api/pipeline/retrieval/test_source_credibility.py`

**Interface:**

```python
# pipeline/retrieval/source_credibility.py

DEFAULT_PRIORS: dict[str, float] = {
    "peer_reviewed_journal": 0.95,
    "official_gov_docs": 0.92,
    "wikipedia_featured_article": 0.88,
    "mcp_curated": 0.80,
    "wikipedia_general": 0.78,
    "news_high_repute": 0.75,
    "qdrant_general": 0.70,
    "news_general": 0.65,
    "graph_inferred": 0.60,
}

def credibility_for(source: str, *, domain_overrides: dict[str, float] | None = None) -> float:
    if domain_overrides and source in domain_overrides:
        return domain_overrides[source]
    return DEFAULT_PRIORS.get(source, 0.5)


def temporal_decay(evidence_age_days: int, *, half_life_days: int = 365) -> float:
    """Half-life decay; returns 1.0 at age 0, 0.5 at half-life, 0.25 at 2x half-life."""
    return 0.5 ** (evidence_age_days / half_life_days)


def fingerprint(source_uri: str, content: str) -> str:
    """SHA-256 over normalized URI + normalized content for cross-path dedup."""
    normalized = (source_uri.strip().lower() + "\n" + " ".join(content.lower().split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
```

**Implementation outline:**

- The move is mechanical (`git mv`), but every callsite needs updating. Use `git grep` to find them; modify imports.
- `Evidence` model gains three optional fields with sensible defaults (so v1 callers continue to work during the transition); `Evidence` is otherwise unchanged.
- The `AdaptiveEvidenceCollector` is updated to populate `source_credibility`, `temporal_decay_factor`, and `fingerprint` on every emitted `Evidence`. Source credibility comes from `credibility_for(ev.source.value, domain_overrides=adapter.source_credibility() if adapter else None)` — adapter is None in Phase 1; per-domain overrides arrive in Phase 3.
- Deduplication in the collector now uses `fingerprint` instead of the ad-hoc string key.

**Tests:**
- `test_source_credibility.py`: parametrized `credibility_for` cases; `temporal_decay` half-life assertion; `fingerprint` determinism + collision-resistance smoke test (10k random inputs, no collisions).
- All existing pipeline tests must continue to pass.

**Acceptance:** `pytest tests/api/pipeline/ -v` passes; `mypy src/api/pipeline/` passes; `git status` shows the renames as renames (not delete+add).

**Commit:** `refactor(L1): split retrieval into pipeline/retrieval/ package, add source_credibility module`

---

### Task 1.5: L7 — output assembly with naive Phase 1 wiring

**Files:**
- Create: `src/api/pipeline/assembly/__init__.py`
- Create: `src/api/pipeline/assembly/claim_verdict.py`
- Create: `src/api/pipeline/assembly/document_verdict.py`
- Create: `src/api/pipeline/assembly/copula.py` (full implementation; used in Phase 1 with naive R = identity, fully populated in Phase 2)
- Create: `tests/api/pipeline/assembly/test_claim_verdict.py`
- Create: `tests/api/pipeline/assembly/test_document_verdict.py`
- Create: `tests/api/pipeline/assembly/test_copula.py`

**Interface:**

```python
# pipeline/assembly/claim_verdict.py

def assemble_claim_verdict(
    claim: Claim,
    *,
    p_true: float,
    interval: tuple[float, float],
    coverage_target: float | None,
    domain: str,
    domain_assignment_weights: dict[str, float],
    supporting_evidence: list[Evidence],
    refuting_evidence: list[Evidence],
    pcg_neighbors: list[ClaimEdge],
    nli_self_consistency_variance: float,
    bp_validated: bool | None,
    information_gain: float,
    queued_for_review: bool,
    calibration_set_id: str | None,
    calibration_n: int,
    fallback_used: str | None,
) -> ClaimVerdict: ...


# pipeline/assembly/document_verdict.py

def assemble_document_verdict(
    claim_verdicts: list[ClaimVerdict],
    *,
    correlation_matrix_R: np.ndarray | None,
    internal_consistency: float,
    decomposition_coverage: float,
    processing_time_ms: float,
    rigor: Literal["fast", "balanced", "maximum"],
    refinement_passes_executed: int,
    model_versions: dict[str, str],
    request_id: UUID,
) -> DocumentVerdict:
    """
    Build the document verdict. If correlation_matrix_R is None (Phase 1),
    use the identity matrix (claims treated as independent).
    """


# pipeline/assembly/copula.py

def gaussian_copula_joint(
    p_per_claim: np.ndarray,        # shape (N,), each ∈ (0, 1)
    correlation_matrix_R: np.ndarray,  # shape (N, N), PSD with unit diagonal
    *,
    n_mc_samples: int = 10_000,
    seed: int | None = None,
) -> float:
    """
    Compute Φ_R(Φ⁻¹(p_1), ..., Φ⁻¹(p_N)) via SciPy's exact MVN CDF when N ≤ 10
    and Monte Carlo otherwise.
    """


def nearest_psd(matrix: np.ndarray, *, eps: float = 1e-4) -> tuple[np.ndarray, float]:
    """
    Project to nearest PSD matrix in Frobenius norm via eigenvalue clipping.
    Returns (projected_matrix, frobenius_distance).
    """
```

**Implementation outline:** straight implementation of spec §9. The copula module uses `scipy.stats.multivariate_normal.cdf` for N ≤ 10 (Genz algorithm) and direct MC sampling otherwise. `nearest_psd` uses `numpy.linalg.eigh`, clips eigenvalues `< eps` to `eps`, and re-normalizes the diagonal to 1.

**Tests:**
- `test_copula.py`:
  - Single-claim degenerate case: `gaussian_copula_joint([0.7], np.array([[1.0]]))` → 0.7 (within MC noise).
  - Independent claims (R = I): joint = product of marginals (within MC noise).
  - Perfectly correlated (R = J): joint = min of marginals (within MC noise).
  - PSD projection: feed a non-PSD matrix, assert smallest eigenvalue ≥ eps after projection, assert diagonal is 1.
- `test_document_verdict.py`: golden test on a hand-built `ClaimVerdict[]` of 3 items; assert exact JSON shape via `model_dump_json()`.
- `test_claim_verdict.py`: round-trip serialization for one example with full provenance.

**Acceptance:** all tests pass; copula MC noise floor < 0.005 on the golden tests.

**Commit:** `feat(L7): output assembly with Gaussian copula joint probability`

---

### Task 1.6: L5 — naive single-quantile conformal stub

**Files:**
- Create: `src/api/pipeline/conformal/__init__.py`
- Create: `src/api/pipeline/conformal/split_conformal.py` (Phase 1: single global quantile; Phase 2 expansion)
- Create: `src/api/pipeline/conformal/calibration_store.py` (Phase 1: in-memory + file-backed; Postgres in Phase 4)
- Create: `tests/api/pipeline/conformal/test_split_conformal.py`
- Create: `tests/api/pipeline/conformal/test_calibration_store.py`

**Interface:**

```python
class SplitConformalCalibrator(ConformalCalibrator):
    def __init__(self, store: CalibrationStore) -> None: ...

    async def calibrate(
        self,
        claim: Claim,
        belief: PosteriorBelief,
        domain: str,
        stratum: str,
    ) -> CalibratedVerdict:
        # Phase 1: always emit fallback_used="general", coverage_target=None,
        #          interval = (0.0, 1.0). Phase 3 fills per-domain quantiles.
        ...
```

**Implementation outline:**

- Phase 1 implementation of `calibrate` is intentionally trivial: emit `p_true = belief.p_true`, `interval = (0.0, 1.0)`, `coverage_target = None`, `fallback_used = "general"`, `calibration_set_id = None`, `calibration_n = 0`. This is the *honest* stub that lets the whole pipeline run end-to-end without a real calibration set yet.
- `CalibrationStore` supports `get_quantile(partition)`, `get_n(partition)`, `add_entry(partition, posterior, true_label)`, `snapshot_to_s3(...)`. Phase 1 implementation is a JSON file at `infra/local/calibration/<partition>.json`; Phase 4 swaps in the Postgres-backed implementation.
- Spec §7 algorithm is fully implemented behind the `if n >= 50:` branch, even though Phase 1 will not hit it. This shrinks the diff for Phase 3.

**Tests:**
- `test_split_conformal.py`:
  - Phase 1 happy path: any input → `fallback_used="general"`, `interval=(0,1)`, `coverage_target=None`.
  - Algorithm correctness with a synthetic calibration set of 100 entries: empirical coverage on a held-out 100 entries should be 0.85 ≤ coverage ≤ 0.95.
- `test_calibration_store.py`: round-trip add/get for the JSON-file backend.

**Acceptance:** tests pass; algorithm test verifies coverage within [0.85, 0.95] on synthetic data.

**Commit:** `feat(L5): split-conformal calibrator with honest Phase 1 fallback stub`

---

### Task 1.7: Pipeline orchestrator

**Files:**
- Create: `src/api/pipeline/pipeline.py`
- Create: `tests/api/pipeline/test_pipeline_orchestrator.py`

**Interface:**

```python
class Pipeline:
    def __init__(
        self,
        decomposer: DecompositionService,
        retrieval: RetrievalService,
        domain_router: DomainRouter | None,        # None in Phase 1 (always-general fallback)
        nli: NLIService | None,                    # None in Phase 1; Phase 2 wires this
        pcg: PCGInferenceService | None,           # None in Phase 1; Phase 2 wires this
        conformal: ConformalCalibrator,
        information_gain_hook: InformationGainHook | None,  # None in Phase 1; Phase 4 wires this
    ) -> None: ...

    async def verify(
        self,
        text: str,
        *,
        context: str | None,
        domain_hint: str | None,
        rigor: Literal["fast", "balanced", "maximum"],
        retrieval_tier: Literal["local", "default", "max"],
        max_claims: int,
        request_id: UUID,
    ) -> DocumentVerdict: ...

    def stream(
        self, ...
    ) -> AsyncIterator[StreamEvent]: ...
```

**Implementation outline:**

- The orchestrator is the single place where the L1→L7 ordering is encoded. Each layer is invoked through its port; Phase 1 supplies `None` for layers not yet implemented, in which case the orchestrator uses a neutral placeholder (e.g., NLI=None → use the existing v1-style heuristic classification temporarily; PCG=None → posterior = mean of evidence support score).
- The placeholder behaviors are clearly marked `# PLACEHOLDER for Phase N` so they're easy to remove.
- `verify` returns a `DocumentVerdict` directly. `stream` yields `StreamEvent` instances corresponding to the spec §10 SSE events; the synchronous endpoint is `verify`.

**Tests:**
- End-to-end test with all-mocked layers asserts the full pipeline produces a valid `DocumentVerdict`.
- Test for Phase 1 placeholder behavior (PCG=None, NLI=None) on a 3-claim fixture.

**Acceptance:** orchestrator produces valid output for the fixture; mypy clean; lint clean.

**Commit:** `feat(pipeline): L1→L7 orchestrator with Phase 1 placeholder layers`

---

### Task 1.8: API routes — `/api/v2/verify` and `/api/v2/verdict/{id}`

**Files:**
- Modify: `src/api/server/app.py` (mount v2 router; drop v1 router)
- Create: `src/api/server/routes/verify.py`
- Create: `src/api/server/schemas/verify.py` (request schema)
- Delete: any existing v1 verify route + schema files
- Create: `tests/api/server/routes/test_verify_v2.py`

**Interface:**

```python
@router.post("/verify", response_model=DocumentVerdict)
async def verify(
    request: VerifyRequest,
    pipeline: Pipeline = Depends(get_pipeline),
) -> DocumentVerdict: ...

@router.get("/verdict/{request_id}", response_model=DocumentVerdict)
async def get_verdict(
    request_id: UUID,
    store: VerificationStore = Depends(get_verification_store),
) -> DocumentVerdict: ...
```

**Implementation outline:**

- `VerifyRequest` mirrors spec §10. Validation: `text` length ≤ 50000, `rigor` enum, `tier` enum, `max_claims` 1–100, `coverage_target` 0.5–0.99 or None.
- `verify` calls `pipeline.verify(...)`; persists the result via `VerificationStore.put(request_id, verdict)`; returns the verdict.
- `VerificationStore` is a port; Phase 1 implementation is in-memory + Redis-backed for cross-worker durability; Postgres-backed implementation already exists in Task 0.1's schema.
- `get_verdict` is a simple lookup; 404 if not found.
- Error handling: `ValidationError` → 400; layer failures (raised by orchestrator) → 503 with `degraded_layers` body.

**Tests:**
- `test_verify_v2.py`:
  - Happy path: small text → 200 + `DocumentVerdict`.
  - Empty text → 200 + `DocumentVerdict(claims=[], document_score=1.0)`.
  - Oversized text → 400.
  - Verdict round-trip: POST /verify → GET /verdict/{id} returns the same body.

**Acceptance:** route tests pass; OpenAPI docs at `/openapi.json` include the new schemas.

**Commit:** `feat(api): /api/v2/verify and /api/v2/verdict/{id} routes`

---

### Task 1.9: API routes — `/api/v2/verify/stream` (SSE)

**Files:**
- Create: `src/api/server/routes/stream.py`
- Modify: `src/api/server/app.py` (mount stream route)
- Create: `tests/api/server/routes/test_stream.py`

**Interface:** `POST /api/v2/verify/stream` returns `text/event-stream`. Events per spec §10.

**Implementation outline:**

- Use FastAPI's `StreamingResponse` with `media_type="text/event-stream"`.
- The body is an async generator that consumes `pipeline.stream(...)` events and serializes each as `event: <name>\ndata: <json>\n\n`.
- Each event is flushed immediately (no buffering at the framework level).
- On client disconnect, the orchestrator is cancelled via `asyncio.CancelledError` propagation.

**Tests:**
- `test_stream.py`: use `httpx.AsyncClient` with `stream` to consume events; assert event order matches spec §10 (decomposition_complete → claim_routed (×N) → nli_complete → pcg_propagation_complete → claim_verdict (×N) → document_verdict). Phase 1 may emit some events with placeholder data (e.g., `nli_complete: {claim_evidence_pairs_scored: 0}`).

**Acceptance:** tests pass; manual `curl -N` against a running dev server shows events streaming correctly.

**Commit:** `feat(api): /api/v2/verify/stream SSE endpoint with progressive layer events`

---

### Task 1.10: Rate limiting middleware

**Files:**
- Create: `src/api/server/middleware/rate_limit.py`
- Create: `src/api/server/middleware/cost_ceiling.py`
- Create: `src/api/server/middleware/internal_auth.py`
- Modify: `src/api/server/app.py` (register middleware in correct order)
- Modify: `src/api/config/settings.py` (add rate-limit + cost-ceiling settings)
- Create: `tests/api/server/middleware/test_rate_limit.py`
- Create: `tests/api/server/middleware/test_cost_ceiling.py`
- Create: `tests/api/server/middleware/test_internal_auth.py`

**Interface:**

```python
class PerIPTokenBucketMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: RateLimitSettings) -> None: ...
    async def dispatch(self, request, call_next) -> Response: ...

class CostCeilingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, daily_budget_usd: float, cost_estimator: CostEstimator) -> None: ...
    async def dispatch(self, request, call_next) -> Response: ...

class InternalAuthMiddleware(BaseHTTPMiddleware):
    """Applied after rate limit; sets request.state.is_internal=True if bearer matches."""
```

**Implementation outline:**

- Token-bucket per IP (per route family) using Redis as the shared store (so the limit applies across workers). Per-IP limits per spec §11. Internal-auth middleware sets `is_internal=True` if `Authorization: Bearer <token>` matches `INTERNAL_BEARER_TOKEN` env, and the rate-limit middleware skips internal requests.
- Cost ceiling: each `verify` request increments a Redis counter `cost:YYYY-MM-DD` by the estimated cost; if the counter exceeds `DAILY_BUDGET_USD`, return 503 with the spec-compliant body. `CostEstimator` for Phase 1 returns a flat estimate per request; Phase 2+ refines based on rigor + N.
- Reset behavior: cost counters expire at end-of-day UTC.

**Tests:**
- Hit `/verify` 11 times in 60s from same IP → 11th returns 429.
- Hit `/health/*` 100 times → all 200 (unlimited).
- Internal bearer bypass: 100 `/verify` calls with valid bearer → all proceed (200 unless cost ceiling hits).
- Cost ceiling: set `DAILY_BUDGET_USD=0.01`, single `/verify` → 503 budget-exhausted.

**Acceptance:** all middleware tests pass; integration test against the running app confirms 429s.

**Commit:** `feat(api): rate-limit + cost-ceiling + internal-auth middleware`

---

### Task 1.11: Retention middleware (no raw-text persistence)

**Files:**
- Create: `src/api/server/middleware/retention.py`
- Modify: `src/api/server/app.py` (register middleware)
- Create: `tests/api/server/middleware/test_retention.py`

**Implementation outline:**

- The middleware does nothing on the request path; it tags the request state with a `retain_text: bool` derived from the `?retain=true` query parameter (default False).
- Downstream, `VerificationStore.put` reads the tag — if False, it stores `text_hash = sha256(text)` only and `text = None`. If True, both are stored.
- This is a minimal middleware because the actual enforcement happens at the storage boundary; the middleware's role is to surface the policy explicitly.

**Tests:** post a request without `?retain`, fetch by request_id, assert `text` field is None / hash present.

**Acceptance:** tests pass; the default for unauthenticated public traffic is "no raw text".

**Commit:** `feat(api): retention policy enforcement (no raw text persisted by default)`

---

### Task 1.12: Health endpoint with `/deep`

**Files:**
- Modify: `src/api/server/routes/health.py` (add `/deep`)
- Create: `tests/api/server/routes/test_health_deep.py`

**Implementation outline:** `/health/deep` runs a synthetic `verify` payload through every layer with a tight timeout per layer; returns the per-layer status + latency dict per spec §10.

**Tests:** mock each layer to succeed quickly; assert response shape matches spec §10. Mock one layer to fail; assert that layer is marked `down` and overall status is `degraded`.

**Acceptance:** test passes; manual `curl /api/v2/health/deep` against running app returns the expected shape.

**Commit:** `feat(api): /health/deep with per-layer status + latency`

---

### Task 1.13: Delete v1 — files, routes, schemas, tests

**Files (DELETE):**
- `src/api/pipeline/oracle.py`
- `src/api/pipeline/scorer.py`
- `src/api/interfaces/verification.py`
- `src/api/interfaces/scoring.py`
- All v1-only routes under `src/api/server/routes/`
- All v1-only schemas under `src/api/server/schemas/`
- All v1-only tests
- v1 entries in `src/api/config/dependencies.py`

**Implementation outline:**

- Use `git grep "from pipeline.oracle"` and `git grep "from pipeline.scorer"` to find every callsite. Replace with the v2 pipeline imports where appropriate; delete dead callsites.
- Update `dependencies.py` to wire the v2 `Pipeline` instead of v1's `verify_service`.
- Run the full test suite; expect failures only in v1-specific tests, which are deleted as part of this task.

**Tests:** no new tests; existing tests must still pass after the deletes (other than the deleted v1 tests).

**Acceptance:** `pytest src/api/tests/ -v` is green; `git grep "oracle\\|scorer"` in `src/api/` returns no hits outside docs.

**Commit:** `refactor!: delete v1 oracle, scorer, routes, schemas, and tests`

---

### Task 1.14: Phase 1 acceptance — benchmark v2-phase1 vs v1 baseline

**Files:**
- Create: `scripts/benchmark/run_v2_benchmarks.py`
- Create: `scripts/benchmark/v2_engine_adapter.py`
- Create: `benchmark_results/v2_phase1_<date>.jsonl`
- Create: `benchmark_results/v2_phase1_<date>.summary.json`
- Modify: `.github/workflows/ci.yml` (add `phase1-acceptance` job)

**Implementation outline:**

- `V2EngineAdapter` wraps the new `Pipeline` and maps `DocumentVerdict` into `BenchmarkResult` (taking `document_score` as the truthful predicted probability).
- `run_v2_benchmarks` runs FActScore + TruthfulQA at `--limit 1000` each.
- `phase1-acceptance` CI job runs both benchmarks and compares F1 against the latest `v1_baseline_*.summary.json`. Fails the build if v2-phase1 F1 < (v1 F1 - 1pt) on either benchmark (within ±1pt is acceptable).

**Tests:** dry-run on small `--limit 50` subset locally before merging.

**Acceptance:** `phase1-acceptance` CI job is green.

**Commit:** `chore(benchmark): Phase 1 acceptance gate — v2-phase1 vs v1 baseline within ±1pt F1`

---

## Phase 2 — Reasoning core (the publishable contribution)

**Goal:** Replace Phase 1's placeholder NLI / PCG layers with the real implementations: NLI cross-encoder with self-consistency (K=10 default), Probabilistic Claim Graph with Ising-style log-linear potentials, TRW-BP primary inference, Gibbs MCMC sanity check on flagged graphs, iterative refinement loop, and conformal recalibration against the new posteriors.

**Acceptance gate:** `phase2-acceptance` CI job: F1 on FActScore + TruthfulQA + HaluEval improves by ≥ 3 points over Phase 1; conformal coverage on a held-out FEVER+ANLI test split is in [0.88, 0.92].

### Task 2.1: NLI cross-encoder service (worked example for Phase 2)

**Files:**
- Create: `src/api/pipeline/nli/__init__.py`
- Create: `src/api/pipeline/nli/cross_encoder.py`
- Create: `src/api/pipeline/nli/batching.py`
- Create: `src/api/adapters/nli_huggingface.py`
- Modify: `src/api/pyproject.toml` (add `transformers`, `torch`, `accelerate`, `safetensors`)
- Create: `tests/api/pipeline/nli/test_cross_encoder.py`
- Create: `tests/api/pipeline/nli/test_batching.py`

**Worked example — full TDD bite-size steps:**

- [ ] **Step 1: Write failing tests**

```python
# tests/api/pipeline/nli/test_cross_encoder.py
import pytest
from pipeline.nli.cross_encoder import HuggingFaceNLIService
from models.nli import NLIDistribution

@pytest.mark.gpu  # gated; CI uses small CPU model in fast lane
async def test_classify_known_entailment():
    svc = HuggingFaceNLIService(
        model_id="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dists = await svc._classify_pairs([
        ("A man is eating food.", "A man eats."),
    ])
    assert dists[0].entail > 0.7

async def test_classify_known_contradiction():
    svc = ...
    dists = await svc._classify_pairs([
        ("A man is eating food.", "A man is sleeping."),
    ])
    assert dists[0].contradict > 0.7

def test_distribution_sums_to_one():
    d = NLIDistribution(entail=0.7, contradict=0.2, neutral=0.1, variance=0.0, nli_model_id="x")
    assert abs((d.entail + d.contradict + d.neutral) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/api/pipeline/nli/ -v -k "test_distribution"
```
Expected: FAIL on import.

- [ ] **Step 3: Implement `cross_encoder.py`**

```python
"""HuggingFace NLI cross-encoder service. See spec §5."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interfaces.nli import NLIService
from interfaces.domain import DomainAdapter
from models.entities import Claim, Evidence
from models.nli import NLIDistribution
from pipeline.nli.batching import batch_pairs

logger = logging.getLogger(__name__)


class HuggingFaceNLIService(NLIService):
    def __init__(
        self,
        *,
        model_id: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        # MNLI label order: contradiction, neutral, entailment
        self._label_index = {"contradiction": 0, "neutral": 1, "entailment": 2}

    async def claim_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence],
        adapter: DomainAdapter,
    ) -> list[NLIDistribution]:
        # Premise = evidence content; hypothesis = claim text (standard MNLI orientation)
        pairs = [(ev.content, claim.text) for ev in evidence]
        return await self._classify_pairs(pairs)

    async def claim_claim(
        self,
        claims: list[Claim],
        adapter: DomainAdapter,
    ) -> dict[tuple, NLIDistribution]:
        # Build unordered pair list
        pairs_with_ids: list[tuple] = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                pairs_with_ids.append((claims[i].id, claims[j].id, claims[i].text, claims[j].text))
        if not pairs_with_ids:
            return {}
        text_pairs = [(p[2], p[3]) for p in pairs_with_ids]
        dists = await self._classify_pairs(text_pairs)
        return {(p[0], p[1]): d for p, d in zip(pairs_with_ids, dists)}

    async def _classify_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIDistribution]:
        out: list[NLIDistribution] = []
        for batch in batch_pairs(pairs, self._batch_size):
            inputs = self._tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                truncation=True,
                padding=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                out.append(NLIDistribution(
                    contradict=float(row[self._label_index["contradiction"]]),
                    neutral=float(row[self._label_index["neutral"]]),
                    entail=float(row[self._label_index["entailment"]]),
                    variance=0.0,  # set by self_consistency wrapper
                    nli_model_id=self._model_id,
                ))
        return out

    async def health_check(self) -> bool:
        try:
            await self._classify_pairs([("test premise", "test hypothesis")])
            return True
        except Exception as e:
            logger.warning("NLI health check failed: %s", e)
            return False
```

- [ ] **Step 4: Implement `batching.py`**

```python
"""Pad-aware batching for NLI cross-encoder inputs."""
from __future__ import annotations
from typing import Iterator

def batch_pairs(pairs: list[tuple[str, str]], batch_size: int) -> Iterator[list[tuple[str, str]]]:
    for i in range(0, len(pairs), batch_size):
        yield pairs[i : i + batch_size]
```

- [ ] **Step 5: Add deps to `pyproject.toml`**

```
"transformers>=4.45",
"torch>=2.4",
"accelerate>=0.34",
"safetensors>=0.4",
```

Then `pip install -e "src/api[dev]"`.

- [ ] **Step 6: Run tests**

```
pytest tests/api/pipeline/nli/ -v
```
Expected: PASS for `test_distribution_sums_to_one`; GPU tests skipped on CPU CI lane.

- [ ] **Step 7: Commit**

```bash
git add src/api/pipeline/nli/ src/api/adapters/nli_huggingface.py src/api/pyproject.toml tests/api/pipeline/nli/
git commit -m "feat(L3): NLI cross-encoder service with HuggingFace DeBERTa-v3-large-MNLI"
```

---

### Task 2.2: Self-consistency sampler (K configurable)

**Files:**
- Create: `src/api/pipeline/nli/self_consistency.py`
- Create: `src/api/pipeline/nli/paraphrase.py`
- Create: `tests/api/pipeline/nli/test_self_consistency.py`

**Interface:**

```python
class SelfConsistencyNLI(NLIService):
    def __init__(
        self,
        *,
        base_nli: NLIService,
        paraphraser: ParaphraseService,
        rigor_to_k: dict[str, int] = {"fast": 3, "balanced": 10, "maximum": 30},
    ) -> None: ...

    async def claim_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence],
        adapter: DomainAdapter,
        *,
        rigor: str = "balanced",
    ) -> list[NLIDistribution]: ...
```

**Implementation outline:**

- For each (claim, evidence) pair, draw K perturbations:
  1. Lexical paraphrase of claim (T5 paraphrase model, cached). Filter to bidirectional NLI entailment ≥ 0.9 vs original.
  2. Evidence sentence-window slide ±1, ±2 sentences (up to 5 windows).
  3. Premise/hypothesis swap (claim-as-premise vs claim-as-hypothesis).
- Run base NLI on each perturbation; average softmax probabilities, compute sample variance.
- Cache T5 paraphrases per claim (keyed by `claim.normalized_form`) in process LRU.

**Tests:**
- Variance > 0 when perturbations disagree, ≈ 0 when they agree.
- Mean of K=10 passes is more stable than K=3 (compare standard deviation across 10 separate K-runs).

**Acceptance:** tests pass; manual: known FEVER pair gives expected distribution within 0.05 of base NLI.

**Commit:** `feat(L3): self-consistency wrapper with input perturbations`

---

### Task 2.3: Bi-encoder claim-pair pre-filter

**Files:**
- Create: `src/api/pipeline/nli/bi_encoder_filter.py`
- Create: `tests/api/pipeline/nli/test_bi_encoder_filter.py`

**Interface:**

```python
class BiEncoderClaimPairFilter:
    def __init__(self, model_id: str = "sentence-transformers/all-mpnet-base-v2") -> None: ...

    def relatedness_matrix(self, claims: list[Claim]) -> np.ndarray:
        """Returns symmetric NxN cosine similarity matrix."""

    def select_pairs_to_score(
        self, claims: list[Claim], *, threshold: float, top_k_cap: int | None
    ) -> list[tuple[ClaimId, ClaimId]]: ...
```

**Implementation outline:** encode each claim once; compute pairwise cosine similarity via matrix product; in `fast`/`balanced` apply `threshold + top_k_cap`; in `maximum` (full quadratic) skip both.

**Tests:** synthetic claims with known relatedness; assert thresholding works.

**Acceptance:** tests pass.

**Commit:** `feat(L3): bi-encoder claim-pair pre-filter for O(N²) → O(N) pruning`

---

### Task 2.4: PCG construction — Ising potentials

**Files:**
- Create: `src/api/pipeline/pcg/__init__.py`
- Create: `src/api/pipeline/pcg/potentials.py`
- Create: `src/api/pipeline/pcg/graph.py`
- Create: `tests/api/pipeline/pcg/test_potentials.py`
- Create: `tests/api/pipeline/pcg/test_graph.py`

**Interface (from spec §6):**

```python
def compute_unary_alpha(
    nli_per_evidence: list[NLIDistribution],
    evidence_weights: np.ndarray,  # w_e from spec §6
) -> float:
    """Returns α_c per spec §6. Real-valued log-odds."""

def compute_edge_J(
    nli_pair: NLIDistribution,
    *,
    gamma: float = 1.0,
) -> float:
    """Returns J_ij per spec §6. Sign encodes entail/contradict."""

@dataclass(frozen=True)
class PCGGraph:
    node_alphas: np.ndarray            # (N,)
    edges: list[tuple[int, int, float]]  # (i, j, J_ij) with i < j
    has_frustrated_cycle: bool
    max_abs_J: float
```

**Implementation outline:** straight implementation of spec §6 unary + edge formulas. `has_frustrated_cycle` detected by DFS over the graph treating contradict edges as sign-flips.

**Tests:**
- Single-evidence cases: assert α_c sign matches the dominant entail/contradict mass.
- Edge: pure-entail NLI → positive J; pure-contradict → negative J; neutral → ~0 J.
- Frustrated triangle: 3 nodes, 3 contradict edges → `has_frustrated_cycle=True`.

**Acceptance:** all tests pass.

**Commit:** `feat(L4): Ising-style PCG potentials (unary α_c, edge J_ij)`

---

### Task 2.5: TRW-BP inference

**Files:**
- Create: `src/api/pipeline/pcg/trw_bp.py`
- Create: `src/api/pipeline/pcg/lbp.py`
- Create: `tests/api/pipeline/pcg/test_trw_bp.py`
- Create: `tests/api/pipeline/pcg/test_lbp.py`

**Interface:**

```python
def trw_bp_infer(
    graph: PCGGraph,
    *,
    max_iters: int = 8,
    damping: float = 0.5,
    tolerance: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (marginals, diagnostics). marginals[c] = P(T_c=+1)."""

def lbp_infer(
    graph: PCGGraph,
    *,
    max_iters: int = 8,
    damping: float = 0.3,
    tolerance: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Damped LBP fallback. Same signature."""
```

**Implementation outline:**

- Compute spanning-tree weights ρ via Kirchhoff's matrix-tree theorem (`numpy.linalg.det` of Laplacian minor).
- Implement TRW-BP message updates in log-space per spec §6 update rule.
- Track convergence (max |Δm|), iteration count, log-partition bound.
- LBP is the same scaffolding without ρ-reweighting.

**Tests:**
- Chain (3 nodes): exact marginals computable; assert TRW-BP matches within 0.01.
- V-structure: assert correct posteriors.
- Triangle: TRW-BP converges; marginals match small enumeration.
- Frustrated cycle: assert TRW-BP either converges or LBP fallback is invoked.

**Acceptance:** all tests pass; convergence rate on 30-claim graphs > 95%.

**Commit:** `feat(L4): TRW-BP primary inference + damped LBP fallback`

---

### Task 2.6: Gibbs sampling sanity check

**Files:**
- Create: `src/api/pipeline/pcg/gibbs.py`
- Create: `tests/api/pipeline/pcg/test_gibbs.py`

**Interface:**

```python
def gibbs_sample(
    graph: PCGGraph,
    *,
    n_burn_in: int = 2000,
    n_samples: int = 8000,
    n_chains: int = 1,
    init_state: np.ndarray | None = None,   # warm-start from TRW-BP marginals
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (marginals, diagnostics including R̂)."""

def should_invoke_gibbs(graph: PCGGraph) -> bool:
    """Per spec §6: True iff max|J| > 0.6 OR has_frustrated_cycle."""

def validate_trw_against_gibbs(
    trw_marginals: np.ndarray,
    gibbs_marginals: np.ndarray,
    *,
    threshold: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (chosen_marginals, per-node bp_validated dict)."""
```

**Implementation outline:**

- Systematic-scan Gibbs sampler in log-space; for each node sample T_c ~ P(T_c | T_{-c}, graph).
- Multi-chain supported via `n_chains > 1`; compute Gelman-Rubin R̂ across chain means.
- Seed derivation: spec §6 — `sha256(request_id + "gibbs")[:8]` as uint64.

**Tests:**
- Independent nodes (no edges): Gibbs marginals match `sigmoid(α_c)` within 0.02.
- Strong-edge chain: Gibbs marginals match enumeration within 0.05.
- Multi-chain R̂ < 1.05 on benign graphs after burn-in.

**Acceptance:** tests pass.

**Commit:** `feat(L4): Gibbs MCMC sanity check with multi-chain Gelman-Rubin`

---

### Task 2.7: Iterative refinement loop

**Files:**
- Create: `src/api/pipeline/pcg/refinement.py`
- Create: `tests/api/pipeline/pcg/test_refinement.py`

**Interface:**

```python
def needs_refinement(b_c: float, S_t: float, S_f: float) -> bool:
    """Spec §6 trigger predicate (bidirectional)."""

async def refinement_pass(
    flagged_claims: list[Claim],
    pcg: PCGInferenceService,
    nli: NLIService,
    retrieval: RetrievalService,
    adapter_per_claim: dict[ClaimId, DomainAdapter],
    *,
    pcg_neighbors: dict[ClaimId, list[ClaimId]],
) -> "RefinementResult": ...
```

**Implementation outline:**

- For each flagged claim, call `retrieval.collect(claim, tier="max")` plus query-expanded retrieval using top-3 contradicting neighbors as additional query terms.
- Re-run NLI on the new evidence and re-affected edges; re-run PCG; return new beliefs + diagnostics.
- Caller (orchestrator) decides whether to invoke another pass via no-progress detector (max |Δb| < 0.02).

**Tests:**
- Trigger predicate: parametrized cases for `(b_c, S_t, S_f)` triples.
- Mock retrieval + NLI + PCG; assert refinement_pass invoked once produces a `RefinementResult` with the new beliefs.

**Acceptance:** tests pass.

**Commit:** `feat(L4): iterative refinement loop with deterministic trigger predicate`

---

### Task 2.8: Wire L3 + L4 into the orchestrator

**Files:**
- Modify: `src/api/pipeline/pipeline.py` (replace placeholders)
- Modify: `src/api/config/dependencies.py` (DI for NLI service, PCG service, refinement)
- Create: `tests/api/pipeline/test_pipeline_phase2.py`

**Implementation outline:** replace the `# PLACEHOLDER for Phase 2` blocks in `pipeline.py` with real calls to `NLIService.claim_evidence`, `NLIService.claim_claim`, `PCGInferenceService.infer`, and (if rigor ≥ balanced) `refinement_pass`. Output of L4 (posterior beliefs) is now the input to L5.

**Tests:** end-to-end pipeline test on a 3-claim fixture with known expected posteriors (mocked NLI returning canned distributions); assert document score reflects joint inference (not just product of marginals).

**Acceptance:** tests pass; the pipeline produces non-trivial PCG neighbors in the output.

**Commit:** `feat(pipeline): wire L3 NLI + L4 PCG into the orchestrator`

---

### Task 2.9: Refit conformal against L4 posteriors

**Files:**
- Modify: `src/api/pipeline/conformal/split_conformal.py` (Phase 2 expansion: real quantile when calibration set has ≥ 50 entries)
- Create: `scripts/calibration/synthesize_phase2_calibration.py` (uses FEVER + ANLI dev splits to produce a synthetic global calibration set)
- Modify: `tests/api/pipeline/conformal/test_split_conformal.py`

**Implementation outline:**

- Run `synthesize_phase2_calibration.py` once; produces `infra/local/calibration/general:any.json` with ≥ 200 entries (claim posterior + true label).
- L5 now branches: if `n >= 50` → real conformal interval; else → Phase 1 fallback. Phase 2 lands the real branch (only `general:any` partition; per-domain in Phase 3).

**Tests:** add coverage assertion test on a held-out synthetic split.

**Acceptance:** coverage on the held-out split is in [0.88, 0.92].

**Commit:** `feat(L5): real conformal calibration with synthesized Phase 2 calibration set`

---

### Task 2.10: Phase 2 acceptance benchmark

**Files:**
- Modify: `scripts/benchmark/run_v2_benchmarks.py` (add `--engine v2-phase2`)
- Modify: `.github/workflows/ci.yml` (add `phase2-acceptance` job)
- Create: `benchmark_results/v2_phase2_<date>.summary.json`

**Implementation outline:** new CI job runs FActScore + TruthfulQA + HaluEval at `--limit 1000` against the v2-phase2 engine; compares to `v2_phase1_*.summary.json`; fails if F1 not improved by ≥ 3pt on at least 2 of 3 benchmarks; also asserts coverage ∈ [0.88, 0.92] on a held-out FEVER+ANLI split.

**Tests:** dry-run on `--limit 50` locally.

**Acceptance:** CI green.

**Commit:** `chore(benchmark): Phase 2 acceptance gate — ≥3pt F1 over Phase 1, coverage ∈ [0.88, 0.92]`

---

## Phase 3 — Domain awareness

**Goal:** Bring up the 5 verticals (general, biomedical, legal, code, social). Per-claim soft routing through `DomainRouter`. Per-domain NLI heads (cold-started by fine-tuning the Phase 2 head per domain). Per-domain conformal calibration sets (synthesized + 200-claim adjudicator labeling sprint at launch). Mondrian stratification and mixture conformal for soft assignments.

**Acceptance gate:** F1 improvement ≥ 5 points on at least 4 of the 5 domain benchmarks vs Phase 2 general-only baseline. All 5 domain calibration sets exist with `n ≥ 200`. Mondrian strata covered for each `(domain, claim_type)` pair where `n ≥ 50`; `general` fallback for the rest, marked correctly in `fallback_used`.

### Task 3.1: Domain enum + Domain port + 5 adapter shells

**Files:**
- Create: `src/api/pipeline/domain/__init__.py`
- Create: `src/api/pipeline/domain/registry.py`
- Create: `src/api/pipeline/domain/adapters/__init__.py`
- Create: `src/api/pipeline/domain/adapters/base.py` (DomainAdapter ABC)
- Create: `src/api/pipeline/domain/adapters/general.py`
- Create: `src/api/pipeline/domain/adapters/biomedical.py`
- Create: `src/api/pipeline/domain/adapters/legal.py`
- Create: `src/api/pipeline/domain/adapters/code.py`
- Create: `src/api/pipeline/domain/adapters/social.py`
- Modify: `src/api/models/results.py` (replace `domain: str` with `domain: Domain` enum once Domain is defined)
- Create: `tests/api/pipeline/domain/test_adapters.py`

**Worked example — full TDD bite-size steps for this task:**

- [ ] **Step 1: Write failing tests**

```python
# tests/api/pipeline/domain/test_adapters.py
import pytest
from pipeline.domain import DomainRegistry
from pipeline.domain.adapters import (
    GeneralAdapter, BiomedicalAdapter, LegalAdapter, CodeAdapter, SocialAdapter
)

def test_registry_lists_all_five_domains():
    reg = DomainRegistry.default()
    assert {a.domain for a in reg.adapters()} == {
        "general", "biomedical", "legal", "code", "social"
    }

def test_each_adapter_returns_required_fields():
    for cls in [GeneralAdapter, BiomedicalAdapter, LegalAdapter, CodeAdapter, SocialAdapter]:
        a = cls()
        assert isinstance(a.nli_model_id(), str)
        assert isinstance(a.source_credibility(), dict)
        assert isinstance(a.calibration_set_id(), str)
        assert 0.0 < a.claim_pair_relatedness_threshold() < 1.0

def test_social_and_legal_use_lower_relatedness_threshold():
    assert SocialAdapter().claim_pair_relatedness_threshold() <= 0.30
    assert LegalAdapter().claim_pair_relatedness_threshold() <= 0.30

def test_biomedical_adapter_overrides_pubmed_credibility():
    bio = BiomedicalAdapter()
    overrides = bio.source_credibility()
    assert overrides.get("pubmed", 0.0) >= 0.90
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/api/pipeline/domain/ -v
```
Expected: ImportError on `pipeline.domain`.

- [ ] **Step 3: Write `pipeline/domain/adapters/base.py`**

```python
"""Base DomainAdapter contract per spec §4."""
from __future__ import annotations
from abc import ABC, abstractmethod

class DomainAdapter(ABC):
    @property
    @abstractmethod
    def domain(self) -> str: ...

    @abstractmethod
    def nli_model_id(self) -> str: ...

    def source_credibility(self) -> dict[str, float]:
        """Per-domain overrides on the global source-credibility table."""
        return {}

    @abstractmethod
    def calibration_set_id(self) -> str: ...

    def decomposition_hints(self) -> str | None:
        return None

    def claim_pair_relatedness_threshold(self) -> float:
        return 0.45
```

- [ ] **Step 4: Implement the 5 adapter classes**

Each one is a few lines: domain name, NLI head ID (initially the shared Phase 2 head; Task 3.4 introduces per-domain heads), calibration set ID `<domain>:any`, optional source-credibility overrides, optional relatedness threshold override.

`SocialAdapter` and `LegalAdapter` set `claim_pair_relatedness_threshold = 0.30`.
`BiomedicalAdapter.source_credibility` sets `pubmed=0.95, cochrane=0.94`.
`LegalAdapter.source_credibility` sets `court_opinion=0.95, statute=0.93`.
`CodeAdapter.source_credibility` sets `official_docs=0.95, stable_repo=0.85, stack_overflow=0.65`.
`SocialAdapter.source_credibility` sets `factcheck_org=0.95, social_media=0.30`.

- [ ] **Step 5: Implement `DomainRegistry`**

```python
class DomainRegistry:
    def __init__(self, adapters: list[DomainAdapter]) -> None: ...
    @classmethod
    def default(cls) -> "DomainRegistry":
        return cls([GeneralAdapter(), BiomedicalAdapter(), LegalAdapter(),
                    CodeAdapter(), SocialAdapter()])
    def get(self, domain: str) -> DomainAdapter: ...
    def adapters(self) -> list[DomainAdapter]: ...
```

- [ ] **Step 6: Define the `Domain` Literal alias**

In `models/domain.py`:
```python
from typing import Literal
Domain = Literal["general", "biomedical", "legal", "code", "social"]
```

Update `ClaimVerdict.domain` and `ClaimVerdict.domain_assignment_weights` to use it (replace `str` and `dict[str, float]` annotations).

- [ ] **Step 7: Run tests + commit**

```
pytest tests/api/pipeline/domain/ -v
ruff format . && ruff check . && mypy src/api
```

```bash
git add src/api/pipeline/domain/ src/api/models/domain.py src/api/models/results.py tests/api/pipeline/domain/
git commit -m "feat(L2): Domain enum + registry + 5 vertical adapters"
```

---

### Task 3.2: Domain router classifier

**Files:**
- Create: `src/api/pipeline/domain/router.py`
- Create: `src/api/adapters/domain_router_distilbert.py`
- Create: `scripts/domain_router/train.py`
- Create: `scripts/domain_router/build_dataset.py` (synthesize labeled dataset from existing benchmark domains + 1k hand-labeled seed)
- Create: `tests/api/pipeline/domain/test_router.py`
- Create: `tests/api/pipeline/domain/test_soft_assignment.py`

**Interface:**

```python
class DomainRouter(Protocol):
    async def route(self, claim: Claim) -> "DomainAssignment": ...

@dataclass(frozen=True)
class DomainAssignment:
    weights: dict[str, float]   # sums to 1
    primary: str                # argmax
    soft: bool                  # True if top1 - top2 < 0.15
```

**Implementation outline:**

- Train script fine-tunes `distilbert-base-uncased` on the constructed dataset. Output: `infra/local/models/domain-router-<date>/` (config + weights + tokenizer).
- `DomainRouterAdapter` loads the model on init; `route` runs single-sample inference (or batched in the orchestrator); maps logits → softmax → `DomainAssignment`.
- Phase 3 test set: 200 hand-labeled claims per domain (via the adjudicator sprint), held out from training.

**Tests:**
- Held-out macro-F1 ≥ 0.85 (gated; can be marked `xfail` until adjudicator sprint completes).
- Soft assignment trigger: contrived 50/50 test case → `soft=True`.

**Acceptance:** F1 gate passes after the adjudicator sprint.

**Commit:** `feat(L2): DistilBERT domain router with soft-assignment support`

---

### Task 3.3: Mixture conformal for soft assignments

**Files:**
- Create: `src/api/pipeline/conformal/mixture.py`
- Modify: `src/api/pipeline/conformal/split_conformal.py` (route to mixture when soft)
- Create: `tests/api/pipeline/conformal/test_mixture.py`

**Implementation outline:** weighted exchangeability per Tibshirani 2019. Given soft `weights`, compute weighted quantile over the union of per-domain calibration sets, with each example weighted by `weights[d] / |D_d|`.

**Tests:** synthetic two-domain case where one domain is dominant in weight; assert resulting quantile ≈ that domain's pure quantile (within 0.02).

**Acceptance:** tests pass.

**Commit:** `feat(L5): mixture conformal for soft domain assignments (Tibshirani 2019)`

---

### Task 3.4: Per-domain NLI heads (cold start by fine-tuning)

**Files:**
- Create: `scripts/nli/cold_start_per_domain.py`
- Create: `infra/local/models/nli-heads/<domain>/<date>/` (output)
- Modify: `src/api/pipeline/nli/cross_encoder.py` (lazy-load per-domain heads)
- Modify: `src/api/adapters/nli_huggingface.py` (head swap based on adapter.nli_model_id())
- Create: `tests/api/pipeline/nli/test_per_domain_heads.py`

**Implementation outline:**

- Per-domain training data: each adapter exposes a `training_corpus()` returning a list of (premise, hypothesis, label) tuples. Sources per spec §4: SciFact + PubMedQA-NLI for biomedical; ContractNLI + LegalBench for legal; synthesized via ingestion (Task 3.5) for code; MultiFC + LIAR + ClimateFEVER + COVID-Fact for social; FEVER + ANLI baseline for general.
- The cold-start script loads the Phase 2 shared head as starting point, fine-tunes the head only (frozen encoder) for 3 epochs per domain, saves to S3.
- `HuggingFaceNLIService` loads the right head based on `adapter.nli_model_id()`; only domains touched by the active document are pinned (LRU cache of 3 heads).

**Tests:**
- Each domain's head loads without error.
- Per-domain head outperforms shared head on its in-domain test set (regression test).

**Acceptance:** all 5 heads cold-started; per-domain F1 on in-domain test set ≥ shared head's F1 + 2pt.

**Commit:** `feat(L3): per-domain NLI head cold-start fine-tuning`

---

### Task 3.5: Code-domain NLI training data synthesis

**Files:**
- Create: `scripts/nli/synthesize_code_nli.py`
- Output: `infra/local/data/code-nli/{train,dev,test}.jsonl`

**Implementation outline:**

- For Python and JavaScript: scrape official docs (cpython, MDN) via existing ingestion pipeline; pair API descriptions with synthetic claims via templates; generate (premise=API doc snippet, hypothesis=API behavior claim, label) triples.
- Validation: a held-out set of 500 manually-labeled triples to bound noise.

**Tests:** synthesized dataset has the expected schema; held-out validation set agreement ≥ 80%.

**Acceptance:** dataset committed; ≥ 5k training triples per language.

**Commit:** `feat(data): synthesize code-domain NLI training set from official docs`

---

### Task 3.6: Adjudicator labeling sprint coordination

**Files:**
- Create: `docs/operations/phase3-labeling-sprint.md`
- Create: `scripts/sprint/sample_for_labeling.py` (samples 200 claims per domain from Phase 1/2 production traffic, balanced for diversity)
- Create: `scripts/sprint/import_labels.py` (imports adjudicator-labeled CSV into `calibration_set` with `source_tier='adjudicator'`)

**Implementation outline:**

- This is partly a coordination task (recruit 2-3 adjudicators, distribute the per-domain claim sheets, collect labels).
- The scripts handle the data plumbing.
- 200 × 5 = 1000 labels total. At ~30s per label, that's ~8 person-hours per domain — tractable.

**Tests:** unit tests on the import script (CSV → DB row mapping; dedup on collision).

**Acceptance:** `calibration_set` has ≥ 200 entries per domain × any-claim-type partition.

**Commit:** `feat(sprint): adjudicator labeling sprint scripts`

---

### Task 3.7: Mondrian stratification

**Files:**
- Create: `src/api/pipeline/conformal/mondrian.py`
- Modify: `src/api/pipeline/conformal/split_conformal.py` (route to Mondrian when stratum has `n ≥ 50`)
- Modify: `src/api/pipeline/assembly/claim_verdict.py` (set `fallback_used` correctly per stratum availability)
- Create: `tests/api/pipeline/conformal/test_mondrian.py`

**Implementation outline:**

- For each `(domain, claim_type)` stratum: if `n ≥ 50`, compute and store a stratum-specific quantile.
- L5 cascade per spec §7:
  1. Stratum-specific quantile if `n ≥ 50`.
  2. Per-domain quantile if `|D_d| ≥ 200`. Set `fallback_used="domain"`.
  3. Global general quantile. Set `fallback_used="general"`, `coverage_target=null`.

**Tests:** synthetic per-stratum calibration; assert correct cascade behavior; coverage measured on held-out splits.

**Acceptance:** coverage maintained per spec gate `[0.88, 0.92]` for strata with sufficient data; fallback semantics match spec §7.

**Commit:** `feat(L5): Mondrian stratification with explicit fallback cascade`

---

### Task 3.8: Wire L2 into the orchestrator

**Files:**
- Modify: `src/api/pipeline/pipeline.py` (replace placeholder router; pass `DomainAdapter` per claim through L3/L5)
- Modify: `src/api/config/dependencies.py` (DI for `DomainRegistry`, `DomainRouter`)
- Create: `tests/api/pipeline/test_pipeline_phase3.py`

**Implementation outline:** for each claim from L1, call `router.route(claim)`, store the `DomainAssignment`, pick the primary `DomainAdapter`, pass it to L3 calls and L5 calls. Soft assignments propagate to L5 via mixture conformal.

**Tests:** end-to-end with mocked router returning soft assignment; assert `domain_assignment_weights` populated correctly in the verdict; assert mixture conformal invoked on soft case.

**Acceptance:** tests pass.

**Commit:** `feat(pipeline): wire L2 domain router into orchestrator with per-claim adapter dispatch`

---

### Task 3.9: Phase 3 acceptance benchmark

**Files:**
- Modify: `scripts/benchmark/run_v2_benchmarks.py` (add `--engine v2-phase3`, run per-domain benchmarks)
- Modify: `.github/workflows/ci.yml` (add `phase3-acceptance` job)

**Implementation outline:** new CI job runs each domain's primary benchmark (PubMedQA, LegalBench-Entailment, code-fact-eval, LIAR/MultiFC); compares to v2-phase2 general-only baseline; fails if F1 not improved by ≥ 5pt on at least 4 of 5 domains.

**Acceptance:** CI green.

**Commit:** `chore(benchmark): Phase 3 acceptance gate — ≥5pt F1 on ≥4/5 domains vs Phase 2`

---

## Phase 4 — Active learning closure (the moat)

**Goal:** Close the loop — `/feedback` endpoint with consensus filter and trust tiers, L6 information-gain hook, nightly retraining DAG (Stages 1–9), online EWC for NLI heads, full re-scoring of calibration set, coverage validation gate, public `/calibration/report` endpoint with nightly HTML snapshot.

**Acceptance gate:** End-to-end loop demonstrably improves F1 ≥ 10% on a prioritized-review subset within 2 weeks of feedback collection. Coverage gate blocks any deployment that fails empirical coverage.

### Task 4.1: Feedback store (Postgres-backed, replaces Phase 1 in-memory)

**Files:**
- Create: `src/api/adapters/postgres_feedback.py`
- Create: `src/api/models/feedback.py` (FeedbackSubmission, CalibrationEntry, FeedbackId)
- Modify: `src/api/pipeline/conformal/calibration_store.py` (Postgres backend)
- Create: `tests/api/adapters/test_postgres_feedback.py`

**Worked example — full TDD bite-size steps:**

- [ ] **Step 1: Write failing tests against the spec §12 SQL behavior**

```python
# tests/api/adapters/test_postgres_feedback.py
import pytest
from uuid import uuid4
from adapters.postgres_feedback import PostgresFeedbackStore
from models.feedback import FeedbackSubmission

@pytest.fixture
async def store(postgres_test_db):
    return PostgresFeedbackStore(dsn=postgres_test_db)

async def test_submit_inserts_into_feedback_pending(store):
    sub = FeedbackSubmission(
        request_id=uuid4(), claim_id=uuid4(), label="true",
        labeler_kind="user", labeler_id_hash="abc"*16,
        rationale=None, evidence_corrections=[], ip_hash=None,
    )
    fid = await store.submit(sub)
    assert fid is not None

async def test_consensus_promotes_after_three_distinct_user_labels(store):
    cid = uuid4()
    for i in range(3):
        await store.submit(FeedbackSubmission(
            request_id=uuid4(), claim_id=cid, label="true",
            labeler_kind="user", labeler_id_hash=f"hash-{i}"*8,
            rationale=None, evidence_corrections=[], ip_hash=None,
        ))
    promoted = await store.promote_consensus()
    assert promoted == 1
    entries = await store.get_calibration_set("general:any")
    assert any(e.claim_id == cid and e.true_label == "true" for e in entries)

async def test_disputed_claims_routed_to_disputed_queue(store):
    cid = uuid4()
    # 3 say true
    for i in range(3):
        await store.submit(FeedbackSubmission(
            request_id=uuid4(), claim_id=cid, label="true",
            labeler_kind="user", labeler_id_hash=f"true-hash-{i}"*8,
            rationale=None, evidence_corrections=[], ip_hash=None,
        ))
    # 3 say false
    for i in range(3):
        await store.submit(FeedbackSubmission(
            request_id=uuid4(), claim_id=cid, label="false",
            labeler_kind="user", labeler_id_hash=f"false-hash-{i}"*8,
            rationale=None, evidence_corrections=[], ip_hash=None,
        ))
    await store.promote_consensus()
    disputed = await store.list_disputed_claims()
    assert cid in {d.claim_id for d in disputed}

async def test_trusted_label_overrides_consensus(store):
    cid = uuid4()
    # build consensus on "true"
    for i in range(3):
        await store.submit(FeedbackSubmission(
            request_id=uuid4(), claim_id=cid, label="true",
            labeler_kind="user", labeler_id_hash=f"u-{i}"*16,
            rationale=None, evidence_corrections=[], ip_hash=None,
        ))
    await store.promote_consensus()
    # adjudicator says false
    await store.submit(FeedbackSubmission(
        request_id=uuid4(), claim_id=cid, label="false",
        labeler_kind="adjudicator", labeler_id_hash="adj-1"*16,
        rationale="evidence X actually refutes", evidence_corrections=[], ip_hash=None,
    ))
    entries = await store.get_calibration_set("general:any")
    final = [e for e in entries if e.claim_id == cid][0]
    assert final.true_label == "false"
    assert final.source_tier == "adjudicator"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/api/adapters/test_postgres_feedback.py -v -m infra
```
Expected: FAIL on import.

- [ ] **Step 3: Implement `models/feedback.py`**

```python
"""Feedback domain models. See spec §10, §12."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

@dataclass(frozen=True)
class EvidenceCorrection:
    evidence_id: str
    correct_classification: Literal["supports", "refutes", "irrelevant"]

@dataclass(frozen=True)
class FeedbackSubmission:
    request_id: UUID
    claim_id: UUID
    label: Literal["true", "false", "unverifiable", "abstain"]
    labeler_kind: Literal["user", "expert", "adjudicator"]
    labeler_id_hash: str
    rationale: str | None
    evidence_corrections: list[EvidenceCorrection]
    ip_hash: str | None

@dataclass(frozen=True)
class CalibrationEntry:
    id: UUID
    claim_id: UUID
    true_label: str
    source_tier: Literal["consensus", "trusted", "adjudicator"]
    n_concordant: int
    calibration_set_partition: str
    posterior_at_label_time: float
    model_versions_at_label_time: dict
```

- [ ] **Step 4: Implement `adapters/postgres_feedback.py`**

```python
"""Postgres-backed feedback store implementing the spec §12 SQL."""
from __future__ import annotations
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from interfaces.feedback import FeedbackStore
from models.feedback import CalibrationEntry, FeedbackSubmission

class PostgresFeedbackStore(FeedbackStore):
    def __init__(self, *, dsn: str) -> None:
        self._dsn = dsn

    async def submit(self, sub: FeedbackSubmission) -> str:
        fid = uuid4()
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO feedback_pending
                       (id, claim_id, label, labeler_kind, labeler_id_hash,
                        rationale, evidence_corrections_jsonb, ip_hash)
                       VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                       ON CONFLICT (claim_id, labeler_id_hash, label) DO NOTHING
                       RETURNING id""",
                    (fid, sub.claim_id, sub.label, sub.labeler_kind,
                     sub.labeler_id_hash, sub.rationale,
                     "[]" if not sub.evidence_corrections
                     else json.dumps([asdict(c) for c in sub.evidence_corrections]),
                     sub.ip_hash),
                )
                row = await cur.fetchone()
                await conn.commit()
                return str(row[0]) if row else None

    async def promote_consensus(self) -> int:
        # Exactly the spec §12 SQL — 4-step CTE
        sql = """ ... spec §12 SQL verbatim ... """
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql)
                count = cur.rowcount
                await conn.commit()
                return count

    async def get_calibration_set(self, partition: str) -> list[CalibrationEntry]:
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """SELECT * FROM calibration_set
                       WHERE calibration_set_partition = %s AND retired_at IS NULL""",
                    (partition,),
                )
                rows = await cur.fetchall()
                return [CalibrationEntry(**self._map_row(r)) for r in rows]

    async def list_disputed_claims(self) -> list:
        ...
```

- [ ] **Step 5: Run tests + commit**

```
pytest tests/api/adapters/test_postgres_feedback.py -v -m infra
git add src/api/adapters/postgres_feedback.py src/api/models/feedback.py tests/api/adapters/test_postgres_feedback.py
git commit -m "feat(L6): Postgres feedback store with spec §12 consensus + dispute SQL"
```

---

### Task 4.2: `/api/v2/feedback` endpoint with trust tiers

**Files:**
- Create: `src/api/server/routes/feedback.py`
- Create: `src/api/server/schemas/feedback.py`
- Create: `src/api/server/middleware/labeler_token.py` (validates `X-OHI-Labeler-Token`, sets `request.state.labeler_kind`)
- Modify: `src/api/server/middleware/rate_limit.py` (tighter limit on `/feedback`: 3/min/IP)
- Create: `tests/api/server/routes/test_feedback.py`

**Implementation outline:**

- The schema mirrors spec §10 `POST /feedback` payload.
- Route handler hashes IP + labeler_id with a per-deployment salt before storage.
- `labeler_kind` validated against the token (or absence) — user can't claim `expert` without a valid expert token.
- Rationale text stored verbatim; will be auto-dropped after 90d by a scheduled cleanup job (Task 4.7).
- Idempotency on `(request_id, claim_id, labeler_id_hash)` via UNIQUE constraint in spec §12 schema.

**Tests:** rate-limit gating works at 3/min; honeypot field rejection; `expert` kind without token → 403; valid expert token → entry written with `source_tier='trusted'`.

**Acceptance:** tests pass.

**Commit:** `feat(api): /feedback endpoint with trust tiers and rate limiting`

---

### Task 4.3: L6 information-gain hook in the request path

**Files:**
- Create: `src/api/pipeline/active_learning/__init__.py`
- Create: `src/api/pipeline/active_learning/information_gain.py`
- Create: `src/api/pipeline/active_learning/review_queue.py`
- Modify: `src/api/pipeline/pipeline.py` (invoke L6 between L5 and L7)
- Create: `tests/api/pipeline/active_learning/test_information_gain.py`

**Interface:**

```python
def compute_information_gain(
    p_true: float,
    interval_lower: float,
    interval_upper: float,
    domain_uncertainty_weight: float,
) -> float:
    """Spec §8: H(b) · (P_upper - P_lower) · domain_uncertainty_weight."""

class ReviewQueue(Protocol):
    async def enqueue_if_high_information(
        self, claim: Claim, ig: float, verdict: ClaimVerdict, *,
        per_doc_cap: int, per_doc_already: int,
    ) -> bool: ...
```

**Implementation outline:**

- `domain_uncertainty_weight` = `clip(1.0 - n_calibration / 1000.0, 0.1, 1.0)`. Sparse domains get high weight, mature domains get low weight.
- Threshold for enqueue: `IG > 0.3` (configurable). Per-document cap: 3.
- Queue-side dedup against existing items via `claim.normalized_form` cosine similarity > 0.95 (best-effort; fire-and-forget so queue lookups don't block the request).
- L6 sends the message via a fire-and-forget asyncio task (not awaited in the request path); ≤ 5ms added.

**Tests:** `H(0.5)` is maximal; `H(0.0) == H(1.0) == 0`; high-IG claim queues; low-IG claim doesn't.

**Acceptance:** tests pass; load test shows < 10ms p95 added by L6 hook.

**Commit:** `feat(L6): information-gain hook with fire-and-forget review queue`

---

### Task 4.4: Calibration re-scoring (nightly DAG Stage 3)

**Files:**
- Create: `scripts/retraining/rescore_calibration.py`
- Create: `tests/scripts/retraining/test_rescore.py`

**Implementation outline:**

- Loads the entire `calibration_set` for a given partition; runs the *current* L4 inference on each claim using its preserved evidence; writes fresh `posterior_at_label_time` as `posterior_under_current_model`. (We do not overwrite `posterior_at_label_time` — that field is historical; we write the fresh values into a `nonconformity_scores` table or as a column in the current calibration snapshot.)
- This is what preserves split-conformal exchangeability — calibration scores and test scores both come from the *same* model.

**Tests:** unit test on a 10-entry mock calibration set; assert all entries have a fresh score after re-scoring.

**Acceptance:** test passes; manual run on a 100-entry set takes < 5 min on the inference cluster.

**Commit:** `feat(retrain): Stage 3 — full calibration set re-scoring under current model`

---

### Task 4.5: Online EWC fine-tuning (nightly DAG Stage 2)

**Files:**
- Create: `scripts/retraining/finetune_nli_head.py`
- Create: `src/api/pipeline/active_learning/ewc.py`
- Create: `tests/api/pipeline/active_learning/test_ewc.py`

**Implementation outline:**

- Implements online EWC per Schwarz et al. 2018: load previous head + Fisher diagonal; compute new Fisher diagonal on incoming training data; apply EMA update with γ=0.95; train with cross-entropy + λ · Σ F · (θ - θ_prev)².
- Up-weight previous-model errors via `1 / max(0.1, posterior_at_label_time)` (capped to prevent extreme weights).
- Save new head + new Fisher to S3.

**Tests:** synthetic continual-learning task: train on task A, then task B with EWC; assert task A accuracy retained ≥ 95% of original after task B fine-tuning.

**Acceptance:** test passes.

**Commit:** `feat(retrain): online EWC NLI head fine-tuning with EMA Fisher`

---

### Task 4.6: Nightly retraining DAG orchestrator

**Files:**
- Create: `scripts/retraining/nightly_dag.py`
- Create: `tests/scripts/retraining/test_nightly_dag.py`

**Implementation outline:**

- Wires Stages 1-9 per spec §12 sequentially. Each stage writes its artifacts; failure at any stage halts the DAG (no partial promote).
- Stage 7 (coverage gate): runs `kfold_cross_coverage` on the re-scored calibration set; asserts ∈ [0.88, 0.92]; failure → no promote, alarm, full report to S3.
- Stage 8 (promote): atomic S3 manifest swap (write new `model_versions/current.json`, signal workers via S3 event or polling).
- Stage 9 (source-credibility): Beta-Bernoulli posterior update with ±0.05 daily cap.

**Tests:** end-to-end DAG test with mocked stages; assert correct ordering; assert promote skipped when coverage gate fails.

**Acceptance:** test passes; manual end-to-end run on a small fixture completes in < 30 min.

**Commit:** `feat(retrain): nightly DAG orchestrator (Stages 1-9 per spec §12)`

---

### Task 4.7: Worker hot-reload + manual rollback CLI

**Files:**
- Create: `scripts/ops/ohi_models.py`
- Modify: `src/api/adapters/nli_huggingface.py` (poll `model_versions/current.json` every 60s, atomic swap at request boundary)
- Modify: `src/api/server/middleware/retention.py` (90-day rationale cleanup job hook)
- Create: `tests/scripts/ops/test_ohi_models.py`

**Implementation outline:**

- `ohi_models status` prints current deployed manifest.
- `ohi_models rollback --to <date>` overwrites `current.json` with a previous manifest, pushes alarm.
- `ohi_models list` lists all available historical manifests in S3.
- Worker poll loop: every 60s, fetch `current.json`; if changed, queue head reload at next request boundary.

**Tests:** rollback writes the right S3 object; worker swap triggered on manifest change.

**Acceptance:** tests pass; manual e2e: deploy v1, run a request, rollback to v0, run another request — second request uses v0.

**Commit:** `feat(ops): worker hot-reload + ohi-models rollback CLI`

---

### Task 4.8: `/api/v2/calibration/report` endpoint + nightly HTML snapshot

**Files:**
- Create: `src/api/server/routes/calibration.py`
- Create: `scripts/calibration/render_html_report.py`
- Modify: `scripts/retraining/nightly_dag.py` (call render after Stage 8)
- Create: `tests/api/server/routes/test_calibration_report.py`

**Implementation outline:**

- `/calibration/report` returns the spec §10 JSON shape; data sourced from a Postgres view `v_calibration_report` joining `calibration_set` aggregates with the latest run's coverage measurements.
- `render_html_report.py` produces a static HTML page (per-domain coverage table + interval-width histograms via matplotlib SVGs) committed to `s3://ohi-artifacts/calibration-reports/YYYY-MM-DD.html`.
- Public CDN-fronts the latest HTML report (handled in the infra sub-project).

**Tests:** route returns valid JSON conforming to spec §10; HTML render produces a valid HTML file with all 5 domains.

**Acceptance:** tests pass.

**Commit:** `feat(api): /calibration/report JSON endpoint + nightly static HTML snapshot`

---

### Task 4.9: Disputed-claims adjudicator CLI

**Files:**
- Create: `scripts/ops/disputed_claims.py`
- Create: `tests/scripts/ops/test_disputed_claims.py`

**Implementation outline:** simple CLI: `list` (queue), `show <claim_id>` (display claim + competing labels + rationales), `adjudicate <claim_id> <label>` (writes adjudicator label, marks queue row resolved).

**Acceptance:** tests pass.

**Commit:** `feat(ops): disputed-claims adjudicator CLI`

---

### Task 4.10: Phase 4 acceptance benchmark

**Files:**
- Modify: `scripts/benchmark/run_v2_benchmarks.py` (add `--engine v2-phase4` + active-learning F1 measurement)
- Modify: `.github/workflows/ci.yml` (add `phase4-acceptance` job)
- Create: `docs/operations/active-learning-runbook.md`

**Implementation outline:**

- Acceptance protocol: simulate 2 weeks of `/feedback` traffic against a held-out test set with deliberately-weak baseline labels; run nightly DAG end-to-end every simulated day; measure F1 on a prioritized-review subset (where IG was high) before and after.
- Pass criterion: ≥ 10% relative F1 improvement on the prioritized-review subset.

**Acceptance:** CI green.

**Commit:** `chore(benchmark): Phase 4 acceptance gate — ≥10% F1 improvement on prioritized-review subset`

---

## Cross-cutting concerns

These are not phase-specific; they apply throughout and are added as discovered during phase execution.

### Logging & metrics

- Per-layer structured logging (`structlog` or stdlib `logging` with JSON formatter).
- Latency metrics per layer emitted via `prometheus-client`; scraped by the infra sub-project.
- Every request logged with `request_id`, `rigor`, `model_versions`, `processing_time_ms`, per-layer breakdown.

### Secret management

- All secrets (`INTERNAL_BEARER_TOKEN`, expert/adjudicator tokens, DB password, S3 keys) loaded via `pydantic-settings` from environment.
- In CI, secrets sourced from GitHub Actions secrets.
- In prod (infra sub-project), AWS Secrets Manager.

### Documentation deliverables (Phase 1+ ongoing)

- `docs/algorithm/v2-overview.md` — Phase 1 deliverable (the system as it ships in P1).
- `docs/algorithm/v2-calibration.md` — Phase 4 deliverable.
- `docs/api/v2-reference.md` — Phase 1; regenerated from FastAPI.
- `docs/operations/active-learning-runbook.md` — Phase 4.

### Test pyramid

- Unit (80%+ coverage gate per module).
- Integration (per phase deliverable; live Postgres + MinIO + LLM).
- Benchmark regression (every PR touching pipeline runs full benchmark suite in CI).
- Calibration validation (nightly job's coverage gate runs in CI on frozen calibration snapshot).
- Load tests (per phase, k6 or Locust).
- Chaos tests (per spec §16).
- Adversarial tests (per spec §16).

---

## Acceptance gates (full table)

| Phase | CI job | Pass criterion |
|---|---|---|
| 0 | `benchmark-baseline-presence` | `v1_baseline_*.summary.json` exists in `benchmark_results/` with all 6 benchmarks scored |
| 1 | `phase1-acceptance` | `v2-phase1` F1 ≥ baseline F1 - 1pt on FActScore + TruthfulQA |
| 2 | `phase2-acceptance` | F1 +3pt over Phase 1 on FActScore + TruthfulQA + HaluEval (≥2/3); coverage ∈ [0.88, 0.92] on FEVER+ANLI held-out |
| 3 | `phase3-acceptance` | F1 +5pt over Phase 2 general-only on ≥4/5 domain benchmarks |
| 4 | `phase4-acceptance` | ≥10% F1 improvement on prioritized-review subset after 2 simulated weeks of `/feedback` |

Each gate is enforced as a CI check; downstream phases' branches are blocked from merging to `main` until the prior gate is green.

---

## Out-of-scope reminders

- AWS infrastructure with Terraform — separate sub-project.
- Next.js frontend — separate sub-project.
- Token-level streaming verification, `/rewrite` auto-correction, cross-language support — explicit non-goals per spec §1.
- Pretraining models from scratch — we fine-tune existing checkpoints only.
- Adaptive Conformal Inference (ACI) — Phase 5+ research extension; not in this plan.

---

*End of implementation plan. Spec reference: [docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md](../specs/2026-04-16-ohi-v2-algorithm-design.md).*



