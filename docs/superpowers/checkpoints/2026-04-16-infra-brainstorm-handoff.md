# OHI v2 — Infra Sub-project Brainstorm Handoff

**Created:** 2026-04-16
**Purpose:** Allow a fresh Claude instance (or this one with compacted context) to pick up the infrastructure sub-project brainstorming exactly where it left off.

---

## The three sub-projects at a glance

The OHI v2 redesign is decomposed into three sequential sub-projects:

1. **Algorithm** — ✅ Phase 1 foundation shipped (8 local commits; not merged to `main`; all tests green)
2. **Infrastructure (this sub-project)** — 🚧 brainstorming in progress; Section 1 approved
3. **Frontend (Next.js rewrite)** — ⏸️ not started

Each sub-project gets its own spec → plan → implementation cycle. Algorithm's spec and plan are committed on `main`:
- Spec: `docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md`
- Plan: `docs/superpowers/plans/2026-04-16-ohi-v2-implementation.md`

---

## Algorithm sub-project — state snapshot (for context only, not for re-work)

Worktree: `.claude/worktrees/ohi-v2-foundation/` on branch `worktree-ohi-v2-foundation`.

Commits (oldest → newest):
```
94f3801  Task 0.1  Local Postgres + MinIO docker-compose (spec §12 schemas)
a080a39  Task 0.2  Engine-agnostic benchmark harness (F1 / ECE / coverage)
5a8163e  Task 1.13+1.1  v1 thrown away + v2 result models (ClaimVerdict, DocumentVerdict, NLI, PCG)
849ae45  Task 1.2  v2 port Protocols (Domain, NLI, PCG, Conformal, Feedback)
5ef2afe  Task 1.4  pipeline/retrieval/ split + source_credibility module
3b40f56  Task 1.3  L1 chunked decomposition + date/number/entity normalization
adf1c98  Task 1.5+1.6+1.7  L7 copula + L5 conformal stub + Pipeline orchestrator
370de3f  Task 1.8+1.11+1.12  /verify + /verdict routes + retention middleware + /health/deep
```

Tests: 147 non-infra + 2 infra (against live docker) all green as of `370de3f`.

Phase 1 progress: 12 of 14 tasks shipped. Remaining Phase 1 tasks (1.9 SSE, 1.10 rate-limit middleware, 1.14 acceptance benchmark) are deferred — they want real AWS resources or the v1 baseline (Task 0.3), neither of which exists yet.

**This sub-project is not being worked on right now. The worktree is the reference, not the target.** When infra work completes, the user will decide whether to merge the algorithm commits to `main` and continue Phase 2+.

---

## Infrastructure sub-project — decisions locked in from brainstorming

### Budget + operational stance

- **Hard cap: ~€150 / month** total AWS cost.
- **Non-profit, open-source, publicly accessible, manually rate-limited.**
- **Expected traffic: very, very low** (< 100 monthly users).
- **No backups, no failover, no zero-downtime, no multi-AZ.** Users expect intermittent availability.
- **No dev/staging environment.** Local Docker Compose is dev; AWS hosts prod only.
- **No local LLM.** It would bottleneck at >1-2 parallel users.
- **GDPR explicitly out of scope for this sub-project.** Deferred.
- **No scaling needed.** If traffic ever grows, deal with it then.

### Architecture split — "AWS compute / PC data"

**On user's PC (Docker Compose, 4 containers, no Python app):**
- `neo4j:latest` — graph store
- `qdrant/qdrant:latest` — vector store
- `postgres:16-alpine` — spec §12 tables (verifications, claim_verdicts, feedback, calibration_set, etc.)
- `redis:alpine` — cache + rate-limit counters
- `cloudflare/cloudflared` — outbound-only tunnel to a Cloudflare subdomain

The existing `docker/compose/docker-compose.yml` is the starting point and can be adapted heavily. A `cloudflared` Dockerfile already exists at `docker/cloudflared/`.

**In AWS (prod only, no dev env):**
- **CloudFront** distribution + **AWS WAF** (Core Rule Set + rate-limit rule)
- **Route53** hosted zone + **ACM** certificate
- **Lambda** container image running FastAPI via **AWS Lambda Web Adapter** (transparent ASGI proxy)
- **Secrets Manager** — Gemini API key, internal bearer token, Cloudflare Tunnel credentials, Postgres password
- **S3** — NLI-head artifacts (Phase 2+), calibration reports (Phase 4), nightly `/calibration/report` HTML snapshot
- **CloudWatch Logs** — Lambda execution (7-day retention), CloudFront access (3-day retention)

**External LLM (usage-based, outside the €150):**
- **Primary: `gemini-3-flash-preview`** — user's choice. Gemini's free tier handles dev traffic; paid tier ~€0.075/M input tokens at low volume.
- Claude Opus 4.7 + OpenAI GPT-4.x: infra-ready (Secrets Manager slots + feature flag), but **no multi-provider routing in Phase 1**. Routing arrives in Phase 2+ when benchmark data justifies it.

### Traffic path

```
USER
  → CloudFront (cache + WAF + TLS)
  → Lambda container (FastAPI + Pipeline orchestrator)
      ↙                                    ↘
  Cloudflare Tunnel                   Google Gemini API
   ↓ (private HTTPS)
  USER's PC:
    Neo4j / Qdrant / Postgres / Redis
```

When the PC is off: Cloudflare Tunnel returns 503 → Lambda catches → returns structured `{"status": "resting"}` JSON → CloudFront can serve a static S3 page on extended outages via custom error response.

### Monthly cost estimate (at very-low traffic)

| Line item | € / month |
|---|---|
| CloudFront | 0–3 |
| AWS WAF (CRS + rate rule) | 5–8 |
| Route53 + ACM | 0.50 |
| Lambda (container, pay-per-request) | 0–3 |
| Secrets Manager (3-4 secrets) | 1–2 |
| S3 | 0–2 |
| CloudWatch Logs | 1–3 |
| Data transfer | 1–3 |
| **AWS subtotal** | **~9–25 €** |
| Gemini 3 Flash (budget-capped) | 0–30 |
| **Total** | **~9–55 €** |

Way under the €150 ceiling, leaving real headroom for growth or Phase 2 multi-provider experiments.

### Calibration cold-start — addressed by existing algorithm spec §15

**User concern:** at <100 users/month, we won't get enough `/feedback` for real calibration.

**Resolution:** already solved by the algorithm spec. Synthesize initial calibration from HuggingFace dev splits (FEVER + ANLI + SciFact + PubMedQA-NLI + MultiFC + LIAR + ClimateFEVER + COVID-Fact). Run them through the pipeline → get `(posterior, true_label)` pairs → thousands of calibration examples per domain without a single human labeler. User `/feedback` grows the set over time via spec §12's 3-concordant consensus filter but is **not required for Phase 1 quality**. Phase 3 launch optionally adds a 200-claim adjudicator sprint per domain for tighter intervals.

**No spec change needed.** The `docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md` spec already calls for this in §15 "tentative recommendations". Flag to carry forward: make sure the infra plan includes an S3-hosted job or Lambda that runs the synthesis script on first deploy and writes the initial `calibration_set.json` to S3 (from where the local Postgres can restore it on PC startup).

---

## Brainstorming Section 1 — **APPROVED** by user

Architecture split (above) and cost breakdown approved. Three confirmation answers already given:

1. **Gemini Flash as sole Phase 1 LLM** — ✅ **user chose `gemini-3-flash-preview` specifically** (not 2.5 Flash)
2. **Lambda container via AWS Lambda Web Adapter** — ✅ approved
3. **Cloudflare Tunnel as PC→AWS bridge** — ✅ approved (reuses existing `docker/cloudflared/` scaffolding)

---

## What remains in the brainstorming flow (for the continuing instance to do)

Per `superpowers:brainstorming` skill the remaining sections are:

### Section 2 — AWS resource details + Terraform module layout
- Terraform organization: recommend single-repo monolith under `infra/terraform/`, split by layer:
  - `edge/` — CloudFront, WAF, Route53, ACM
  - `compute/` — Lambda container image, IAM roles, function URL
  - `storage/` — S3 buckets + bucket policies
  - `secrets/` — Secrets Manager entries (values injected out-of-band, not in TF)
  - `observability/` — CloudWatch log groups, metric filters, alarms
  - `_shared/` — provider config, tags, naming helpers
- Backend: **S3 + DynamoDB state locking** in a separate bootstrap stack (also Terraform).
- Provider: `hashicorp/aws ~> 5.x`. Pin versions.
- Variable strategy: one `terraform.tfvars` committed with prod defaults, override via env vars in CI.

### Section 3 — CI/CD + deployment pipeline
- GitHub Actions workflow: `.github/workflows/deploy-infra.yml`.
- **OIDC federation** to AWS (no long-lived access keys).
- Two jobs: `plan` (on PR) and `apply` (on push to main, manual approval gate).
- Separate workflow for the Lambda container image: build → push to ECR → `terraform apply` bumps the image tag → Lambda rolls.
- **Image build strategy**: `docker buildx` multi-arch (ARM64 native for cheaper Lambda) + layer caching via `actions/cache`.

### Section 4 — Secrets + environment configuration
- Secrets stored in Secrets Manager:
  - `ohi/prod/gemini-api-key`
  - `ohi/prod/internal-bearer-token`
  - `ohi/prod/cloudflare-tunnel-token`
  - `ohi/prod/postgres-password` (for the Postgres running on the PC, injected into the docker-compose via systemd)
- Lambda reads them via SDK at cold-start, caches in-memory for the execution env's lifetime.
- Secret rotation: manual for Phase 1 (documented in README); automatic rotation not worth the complexity at €150/mo scale.

### Section 5 — Observability + cost controls
- CloudWatch Logs with structured JSON via Python `structlog`.
- Log metric filters for: `pipeline_error` count, `rate_limit_triggered` count, `pc_origin_timeout` count.
- **Budget alarm**: AWS Budgets notification at €100/month and €140/month. Sends an email to the user.
- **Gemini cost ceiling**: Lambda-side daily counter in the local Postgres (or Redis). When exceeded, `/verify` returns 503 "budget exhausted" until UTC midnight.

### Section 6 — Phased rollout
Proposed phases for the infra sub-project:
- **Phase I.0** — bootstrap: Terraform state bucket + DynamoDB table, GitHub OIDC role, domain + ACM cert.
- **Phase I.1** — core prod: CloudFront + WAF + Lambda stub (returns `{"hello": "world"}`) + S3 + Secrets Manager + CloudWatch.
- **Phase I.2** — app wire-up: bake the real Lambda container image from the algorithm worktree, deploy, verify `/health` works end-to-end from CloudFront.
- **Phase I.3** — PC ↔ AWS tunnel: bring up Cloudflare Tunnel on the user's machine, configure Lambda env vars to point at the tunnel URL, verify `/verify` works end-to-end against the real data stores.
- **Phase I.4** — budget + observability: CloudWatch alarms, AWS Budget alert, dashboards.

Each phase committed + tested before moving to the next. Acceptance gate per phase (CloudFront resolves, Lambda returns 200, tunnel reachable, etc.).

### Section 7 — Risks + deferred items
To be filled in. Expected entries: residential ISP flakiness, Cloudflare Tunnel auth-token exfiltration, Lambda cold-start latency on Python container, Gemini API deprecation risk.

---

## After brainstorming — handoff to writing-plans

Per the `superpowers:brainstorming` skill, the terminal state is invoking `superpowers:writing-plans`. **Do NOT invoke `mcp-builder`, `frontend-design`, or any other implementation skill from brainstorming.** The only allowed next step is writing-plans, which creates the implementation plan from the finalized spec.

Spec path (when written): `docs/superpowers/specs/2026-04-17-ohi-v2-infrastructure-design.md` (use today's date; this file may live one day earlier).

Plan path (when written): `docs/superpowers/plans/2026-04-17-ohi-v2-infrastructure.md`.

---

## How to pick this up (continuing instance instructions)

1. **Read this checkpoint file first**, plus:
   - `docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md` (especially §11 for the infra constraints the algorithm assumes)
   - `docs/superpowers/plans/2026-04-16-ohi-v2-implementation.md` (algorithm plan; infra plan will be separate)
   - `AGENTS.md` for repo-wide conventions
   - `docker/compose/docker-compose.yml` (existing compose; starting point for PC-side config)
2. **Skip re-doing Section 1 of brainstorming.** It's approved.
3. **Proceed with Sections 2-7** outlined above. One question at a time per the brainstorming skill.
4. **Write the infra spec** once the user approves all sections.
5. **Run the spec reviewer loop** (dispatch a subagent with the spec path + relevant context; iterate on "Issues Found" verdicts).
6. **Get user sign-off on the written spec.**
7. **Invoke `superpowers:writing-plans`.** Do not invoke any other skill.

### Operational notes

- User prefers **concise, structured responses** with clear section headers + multi-choice questions where possible.
- **`AskUserQuestion` tool** is preferred over free-form Q&A; its schema may require reloading via `ToolSearch` after context compaction.
- **Git discipline**: commits are local only; user has explicitly said "do not push".
- **Windows environment**: user is on Windows with Git Bash. Use absolute paths when Bash `cwd` resets between tool calls.
- **Worktree isolation**: the algorithm sub-project lives in `.claude/worktrees/ohi-v2-foundation/`. The infra sub-project should get its own worktree via `EnterWorktree` when implementation starts — pick a name like `ohi-v2-infra`.

---

## One open thread

The calibration cold-start concern from the user's last message was addressed by noting spec §15 already handles it via HuggingFace-synthesized calibration sets. **Worth capturing as an explicit task in the infra plan**: build a first-deploy Lambda (or CI step) that runs `scripts/calibration/synthesize_phase2_calibration.py` (this script will be created in algorithm Task 2.9) and writes the initial `calibration_set.json` to S3, from where the PC's Postgres can restore on startup.

End of checkpoint.
