# OHI v2 — Handoff Checkpoint

**Last updated:** 2026-04-16 (end of session)
**Purpose:** Let a fresh Claude instance resume work with zero context loss.

---

## 1. Where we are in the three-sub-project plan

| # | Sub-project | Status |
|---|---|---|
| 1 | **Algorithm** | Phase 1 foundation **shipped to `origin/feat/ohi-v2-foundation`** (9 commits, 147 tests green). Phase 1 tasks 1.9 (SSE), 1.10 (rate-limit), 1.14 (acceptance bench) + all of Phase 2/3/4 still pending. |
| 2 | **Infrastructure** | Brainstorming **Section 1 approved** (architecture + €150/mo scope). Sections 2–7, spec, plan, and implementation all pending. **This is the active sub-project.** |
| 3 | **Frontend (Next.js rewrite)** | Not started. |

---

## 2. What's on the remote branch `feat/ohi-v2-foundation`

Pushed as of this checkpoint. 9 commits above `origin/main`:

```
8b566e7  chore(gitignore): exclude /docs/superpowers/ from git
b018fd3  refactor(L1): Task 1.4 — split retrieval into pipeline/retrieval/
9b470dc  feat(api): Tasks 1.8 + 1.11 + 1.12 — HTTP surface, retention, /health/deep
1131e39  feat(pipeline): Tasks 1.5 + 1.6 + 1.7 — L7 assembly + L5 conformal + orchestrator
e3bea32  feat(L1): Task 1.3 — chunked decomposition + normalization
131fe84  feat(interfaces): Task 1.2 — v2 port Protocols + value objects
3f0a40f  feat(v2): throw away v1 + build v2 result models (Tasks 1.13 + 1.1)
6a13b27  feat(benchmark): engine-agnostic harness with F1, ECE, coverage, runner
e46156c  feat(infra): local Postgres + MinIO dev stack with spec §12 schemas
```

**Conflict-resolution caveat for PR reviewers:** cherry-picked with `-X theirs`. Silently preferred worktree code over any origin/main side edits to v1 files we deleted (`scorer.py`, `oracle.py`, `admin.py`, `redis_cache.py`). The Task 1.4 retrieval-package split landed on top (commit `b018fd3`) rather than in its natural order, so individual intermediate commits are not bisectable — **squash-merge is the recommended PR strategy**.

PR URL: https://github.com/shiftbloom-studio/open-hallucination-index/pull/new/feat/ohi-v2-foundation

---

## 3. Local-only artifacts (not on any remote)

| Path | Purpose | Survives session boundaries? |
|---|---|---|
| `docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md` | Algorithm v2 spec (final) | Yes — on local `main` |
| `docs/superpowers/plans/2026-04-16-ohi-v2-implementation.md` | Algorithm v2 implementation plan (47 tasks) | Yes — on local `main` |
| `docs/superpowers/checkpoints/2026-04-16-infra-brainstorm-handoff.md` | **This file** | Yes — on local `main`; gitignored on the feat branch |
| `.claude/worktrees/ohi-v2-foundation/` | Original algorithm worktree (pre-push, same 8 code commits) | Yes — `EnterWorktree` can resume it |

These are gitignored on `feat/ohi-v2-foundation` (see `.gitignore` entry for `/docs/superpowers/`). They are tracked on local `main`, which is 5 commits ahead of `origin/main` and will never be pushed without explicit user direction.

---

## 4. Infrastructure sub-project — decisions locked in from Section 1

### Budget + operational stance
- **Hard cap: ~€150/month total AWS** (external LLM usage is outside this cap and budget-capped separately).
- Non-profit, open-source, publicly accessible, manually rate-limited (PC on/off = service on/off).
- Expected traffic: < 100 monthly users.
- **No** backups, failover, zero-downtime, multi-AZ, dev/staging env, local LLM, scaling, or GDPR work (all explicitly deferred).

### Architecture — "AWS compute / PC data"

**On user's PC (4 Docker containers, no Python app):**
- `neo4j:latest` — graph
- `qdrant/qdrant:latest` — vectors
- `postgres:16-alpine` — spec §12 tables
- `redis:alpine` — cache + rate-limit counters
- `cloudflare/cloudflared` — outbound-only tunnel (reuses existing `docker/cloudflared/`)

**In AWS (prod-only env):**
- CloudFront + AWS WAF (Core Rule Set + rate-limit rule) + Route53 + ACM
- **Lambda container** running FastAPI via **AWS Lambda Web Adapter**
- Secrets Manager (Gemini key, internal bearer, Cloudflare Tunnel credentials)
- S3 (NLI artifacts, calibration reports)
- CloudWatch Logs (short retention)

**External LLM (outside €150):**
- **`gemini-3-flash-preview`** as sole Phase 1 provider (user's explicit pick)
- Claude Opus 4.7 + OpenAI slots reserved but no multi-provider routing in Phase 1

**Traffic path:**
```
USER → CloudFront (cache + WAF + TLS)
     → Lambda (FastAPI + Pipeline orchestrator)
         → Cloudflare Tunnel → PC (Neo4j/Qdrant/Postgres/Redis)
         → Google Gemini API
```

When PC is off: tunnel 503 → Lambda catches → structured `{"status":"resting"}` JSON. Optional CloudFront custom error page for extended outages.

### Estimated monthly cost
- AWS subtotal: ~€9–25 (CloudFront + WAF + Lambda + S3 + Secrets + CloudWatch + Route53)
- Gemini API (budget-capped): €0–30
- **Total: ~€9–55/month** — well under the €150 ceiling, leaves headroom for growth or Claude/OpenAI experiments later

### Calibration cold-start — already addressed by algorithm spec §15

At <100 users/month, user `/feedback` can't fill calibration sets. **No spec change needed** — synthesize initial calibration from HuggingFace dev splits (FEVER + ANLI + SciFact + PubMedQA-NLI + MultiFC + LIAR + ClimateFEVER + COVID-Fact) through the pipeline. Thousands of examples per domain, zero humans required. Phase 3 optionally adds a 200-claim adjudicator sprint per domain. **Infra plan should include a first-deploy Lambda or CI step** that runs `scripts/calibration/synthesize_phase2_calibration.py` (created in algorithm Task 2.9) and writes the initial `calibration_set.json` to S3.

---

## 5. What remains in the brainstorming flow

Section 1 approved. Remaining sections to work through **one at a time** with the user (use `AskUserQuestion` for multi-choice, keep questions tight):

- **Section 2 — Terraform module layout** — single-repo under `infra/terraform/`, split by layer (`edge/`, `compute/`, `storage/`, `secrets/`, `observability/`, `_shared/`). Backend: S3 + DynamoDB state locking. Provider `hashicorp/aws ~> 5.x`.
- **Section 3 — CI/CD** — GitHub Actions, OIDC federation (no long-lived keys), two jobs (`plan` on PR, `apply` on main push + manual approval gate). Separate workflow for the Lambda container image build → push to ECR → `terraform apply` bumps image tag.
- **Section 4 — Secrets + env config** — Secrets Manager entries, Lambda cold-start SDK reads, in-memory caching. No auto-rotation in Phase 1.
- **Section 5 — Observability + cost controls** — CloudWatch structured logs (structlog), metric filters (`pipeline_error`, `rate_limit_triggered`, `pc_origin_timeout`), AWS Budget alarm at €100 + €140, Gemini daily cost ceiling enforced in Lambda.
- **Section 6 — Phased rollout** — I.0 bootstrap (TF state + OIDC + domain + ACM), I.1 core prod (CloudFront + WAF + stub Lambda + S3 + Secrets), I.2 real Lambda image, I.3 PC ↔ AWS tunnel, I.4 budget + observability.
- **Section 7 — Risks + deferred items** — residential ISP flakiness, Cloudflare Tunnel auth-token exfiltration, Lambda cold-start, Gemini API deprecation.

Then: write spec to `docs/superpowers/specs/2026-04-17-ohi-v2-infrastructure-design.md`, run spec-reviewer loop, get user approval, invoke `superpowers:writing-plans`.

**Terminal step: `superpowers:writing-plans`** — do not invoke `mcp-builder`, `frontend-design`, or any other implementation skill from brainstorming.

---

## 6. Operating conventions (stay consistent across sessions)

- **Local commits only** unless the user explicitly says "push".
- **Windows + Git Bash**: Bash `cwd` resets between tool calls — use absolute paths.
- **AskUserQuestion** preferred over free-form Q&A. Schema may need reloading via `ToolSearch` after context compaction.
- **Concise, structured responses** with clear section headers and multi-choice questions.
- User's pre-existing WIP (`gui_ingestion_app/*`, `src/api/config/dependencies.py`, `src/api/config/settings.py`, `src/api/adapters/null_graph.py`, `nul`, `api/`) is **not related to v2 work** — leave it alone.
- Spec doc follows `superpowers:brainstorming` format. Plan doc follows `superpowers:writing-plans` format.

---

## 7. The one thing blocking the next session

**Ask the user:** "Resume infra brainstorming at Section 2 (Terraform module layout)?" — or, if they'd rather switch to a different sub-project first, pivot. The brainstorming HARD-GATE applies: do not write Terraform, scaffold files, or invoke any implementation skill until the infra spec is written and approved.

End of checkpoint.
