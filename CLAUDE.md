# CLAUDE.md — OHI repo agent instructions

You are working on the Open Hallucination Index (OHI) project. This file
gives you the hard-won operational knowledge the current owner
(Fabian, `fabian@shiftbloom.studio`) and prior agents have accumulated.
**Read it fully before making any non-trivial change.**

Production is a **Cloudflare Worker** (`cloudflare/ohi-worker/`) — Workers
runtime, D1, Vectorize, R2, Queues, Durable Objects, Workflows, and
Workers AI. There is no AWS, Neo4j, Bedrock, Qdrant, Vercel, or
Windows/Git-Bash dependency in the production path. The FastAPI tree
under `src/api/`, `docker/`, and `infra/` (Terraform) are retained for
local development and migration reference only — see
[docs/CURRENT_ARCHITECTURE.md §4](docs/CURRENT_ARCHITECTURE.md#4-local-development).

## Before anything

1. Read [`docs/CURRENT_ARCHITECTURE.md`](docs/CURRENT_ARCHITECTURE.md) —
   the **single source of truth** for production topology, Cloudflare
   resource bindings, and the "last verified against prod" anchor at the
   top of the file (date + Worker version + corpus size). If anything in
   THIS file (CLAUDE.md) contradicts CURRENT_ARCHITECTURE.md, trust
   CURRENT_ARCHITECTURE.md and update CLAUDE.md in the same session.
2. Check `git log --oneline -20` and `gh run list --limit 10` for the
   real current state before claiming anything about what's deployed —
   this repo has a documented history of docs drifting behind reality
   (that's exactly why this file was rewritten on 2026-07-05).
3. If the task touches the verification pipeline, read
   `cloudflare/ohi-worker/src/index.ts` (request routing, NLI ensemble,
   evidence retrieval, admin/turnstile auth) and
   `cloudflare/ohi-worker/src/knowledge-tools.ts` (the shared MCP +
   pipeline evidence connectors) first, to see what's actually wired.
4. If the task touches corpus growth, read
   `cloudflare/ohi-worker/src/corpus-ingestion.ts` (the
   `ohi-corpus-ingestion` Workflow) before assuming what a run does.

## Single Source of Truth for architecture

`docs/CURRENT_ARCHITECTURE.md` is authoritative for:

- Production topology (what talks to what, which Cloudflare products,
  which external APIs)
- The Cloudflare resource table (Workers, D1, Vectorize, R2, Queues,
  Durable Objects, Workflows, Turnstile, WAF)
- Planned-but-unshipped changes
- Last-verified-against-prod anchor (date, Worker version, corpus size)

**Rule:** any architectural change (new Cloudflare resource, changed
binding, changed model, changed external API dependency) MUST update
CURRENT_ARCHITECTURE.md in the **same commit** as the code change.
Reviewers should reject PRs that change architecture without updating
the SSoT. **This rule was in the old version of this file too and was
not followed for months — treat it as non-negotiable, not aspirational.**

Other docs (`README.md`, `docs/API.md`, `docs/FRONTEND.md`) may
summarize or reference architecture but should NOT duplicate the
authoritative details — link back to CURRENT_ARCHITECTURE.md instead.
This CLAUDE.md file follows the same rule: it intentionally does not
restate topology, resource IDs, or "current state as of <date>" prose
that belongs in CURRENT_ARCHITECTURE.md, because that's exactly the
content that went stale last time.

## Anti-hallucination protocol (mandatory for every agent)

Verified session-lapses on this project include: citing a file's line
number from memory (was wrong by 40+ lines); writing smoke assertions
that were structurally unreachable given the pipeline's current state;
assuming handoff files were saved to disk when they were only pasted
into chat; and — the reason this file was rewritten — an architecture
doc update landing without the corresponding CLAUDE.md update, leaving
this file describing AWS Lambda/Neo4j/Windows infra for months after
the production system moved to Cloudflare Workers. All cases:
memory-recall or stale-doc-recall without tool verification. This
section is the project-wide countermeasure.

**Every agent (including the coordinator) MUST:**

- **Read before claim.** Never state what a file contains (line
  numbers, function signatures, imports, specific strings, types,
  interfaces) without having `Read` it in the current session. A
  prior session's read does not count if the file may have changed.
- **Grep before claim.** Never quote a symbol (function name,
  variable, class, import path) from memory. Run `Grep` first.
- **Git-command before claim.** Never state what's on a branch,
  what's merged, or what the `main` tip is without running `git log`,
  `git diff`, `git rev-parse`, `git branch`, or `git merge-base`.
- **Query prod before claim.** Before claiming the Worker's deployed
  version, `/health/deep` state, D1 row counts, Vectorize index size,
  or any other prod condition, run the `wrangler` / `curl` / `gh`
  command — don't trust a prior session's snapshot of prod state.
- **Fetch docs before API claims.** Never assert parameter names,
  return-type shapes, or behaviour of an external library/SDK/API
  (Cloudflare Workers runtime APIs included — they change fast) from
  training-data recall. Use `WebFetch` to pull current docs; pin
  `wrangler`/SDK versions in code that depends on the shape.
- **Run the tests before saying they pass.** Never state "tests green"
  from memory. Run the relevant suite: frontend `pnpm run test:run`
  (`src/frontend/`), Worker `pnpm run check` + `pnpm run build`
  (`cloudflare/ohi-worker/` — typecheck + `wrangler deploy --dry-run`),
  or legacy-tree `pytest -q tests -m "not infra"` if the change touches
  `src/api/`.
- **Test or runbook-source every shell recipe for destructive
  operations.** Especially secret rotation (`wrangler secret put`),
  D1 migrations against `--remote`, and anything touching the
  `ohi-corpus-prod` R2 bucket or `ohi-prod` D1 database. If writing a
  new recipe, validate each step before handing it to another agent.
- **Trace the code path before writing smoke assertions.** Before
  asserting a specific `p_true`, `relevance_score`, or verdict shape in
  a smoke script, `Read` the current `index.ts` NLI/evidence logic to
  confirm the expected value is actually produced by this deploy's
  code. Asserting forward on not-yet-shipped behavior is structurally
  unreachable and will make a real regression look like a smoke-script
  bug (or vice versa).
- **`ls` or `Read` before claiming a file was saved.** Producing
  content in chat is not the same as persisting to disk. Confirm.

**When uncertain, state uncertainty.** "I don't know," "I'd need to
check," and "let me verify" are first-class responses. Strictly
better than a confident-wrong claim.

**Self-critique trigger (before any factual message):** ask "what
tool did I run to know this, and what did it output?" If you can't
name the tool + output, you haven't verified — go run it or flag
the claim as unverified.

**Template for agent prompts** (paste verbatim into every new stream
prompt under "First actions"):

```
<investigate_before_answering>
Never speculate about code, files, git state, deployed state, or API
shapes you have not verified with a tool. Before any factual claim
about this project: (a) for file content — `Read` the file; (b) for
code symbols — `Grep` the symbol; (c) for git state — run the `git`
command; (d) for deployed state — run the `wrangler` / `curl` / `gh`
command; (e) for external APIs — `WebFetch` the docs. State
uncertainty explicitly when you haven't verified. "I don't know" is
a valid and preferred response to a confident-wrong one.
</investigate_before_answering>
```

## Environment assumptions

- **OS:** macOS (Darwin). No Windows/Git-Bash quirks apply — standard
  POSIX paths, standard `cd`, no path-mangling or charmap traps.
- Two independent pnpm projects, each with its own lockfile — `cd` into
  the right one before running scripts:
  - `src/frontend/` — Next.js 16, statically exported (`pnpm run build`
    → `out/`), which the Worker serves via Static Assets.
  - `cloudflare/ohi-worker/` — the Worker itself (`wrangler`,
    TypeScript, `@modelcontextprotocol/sdk`, `agents`).
- **`wrangler`** (v4, see `cloudflare/ohi-worker/package.json`) is
  authenticated locally via OAuth token (`wrangler whoami` to confirm
  identity/account). CI (`cloudflare-production.yml`) instead uses the
  `CLOUDFLARE_API_TOKEN` repo secret — don't assume the two auth paths
  behave identically if you hit a permissions error.
- **`gh` CLI is authenticated** in this environment — use it to check
  real Actions run status (`gh run list`, `gh run view <id>`) instead
  of assuming a workflow succeeded because the code looks right.
- Node 24 (matches `.github/workflows/cloudflare-production.yml`).
- Legacy local dev (`src/api/` FastAPI, `docker/compose/pc-data.yml`,
  `tests/` pytest suite) still works for local-only iteration but is
  not part of the deploy path and is not exercised by
  `cloudflare-production.yml`. Canonical pytest runline if you touch
  that tree: `pytest -q tests -m "not infra"`.

## Deployment (Cloudflare)

Deployment lives entirely in `cloudflare/ohi-worker/`. Canonical manual
flow (also what CI runs, minus the secret-sync steps):

```bash
# 1. build the frontend static export first — the Worker's assets binding
#    points at src/frontend/out
cd src/frontend
NEXT_PUBLIC_API_BASE=https://ohi.shiftbloom.studio/api/v2 \
NEXT_PUBLIC_SITE_URL=https://ohi.shiftbloom.studio \
pnpm run build

# 2. build + typecheck + deploy the Worker
cd ../../cloudflare/ohi-worker
pnpm install
pnpm run types    # wrangler types -> src/worker-configuration.d.ts
pnpm run check    # tsc --noEmit
pnpm run build    # wrangler deploy --dry-run --outdir dist
pnpm run deploy   # wrangler deploy
```

Apply D1 migrations (`cloudflare/ohi-worker/migrations/`) with:

```bash
cd cloudflare/ohi-worker
pnpm exec wrangler d1 migrations apply ohi-prod --remote
```

**CI/CD gate:** `.github/workflows/cloudflare-production.yml`, triggered
on push to `main` or manual `workflow_dispatch`. `verify` job runs
frontend lint/test/build + Worker typecheck/dry-run-build; `deploy` job
(needs `verify` green) builds the frontend, applies D1 migrations,
syncs the two Worker secrets below from repo secrets, runs
`wrangler deploy`, and curls `/health/live`, `/health/ready`,
`/health/deep` as a post-deploy smoke check. There is no separate
staging environment — `main` green is the gate (see
`environment: production` in the workflow).

**Secrets are Worker secrets, not `wrangler.jsonc` vars:**
- `ADMIN_TOKEN` — gates `GET/POST /api/v2/admin/*` (corpus overview,
  corpus run start/status). Missing → those endpoints 503 with
  `admin_token_not_configured`. Send it as `Authorization: Bearer
  <token>` or `X-OHI-Admin-Token: <token>`.
- `TURNSTILE_SECRET_KEY` — server-side verification for the Turnstile
  widget gating `/api/v2/verify`. Missing → Turnstile check is
  effectively disabled (see `requireTurnstile` in `src/index.ts`).

Both are set via `wrangler secret put <NAME>` and are re-synced from
the `OHI_ADMIN_TOKEN` / `TURNSTILE_SECRET_KEY` GitHub repo secrets on
**every successful CI deploy** (`cloudflare-production.yml` steps
"Configure admin token secret" / "Configure Turnstile secret"). If you
rotate either manually with `wrangler secret put`, the next green CI
deploy silently overwrites it back to the repo-secret value — update
the GitHub repo secret too, or your manual rotation won't stick.

## Corpus ingestion (D1 + Vectorize growth)

The `ohi-corpus-ingestion` Cloudflare Workflow
(`cloudflare/ohi-worker/src/corpus-ingestion.ts`, binding
`CORPUS_WORKFLOW`) is how the evidence corpus grows. It's started via
admin endpoints (`ADMIN_TOKEN` required):

```bash
# start a run
curl -X POST https://ohi.shiftbloom.studio/api/v2/admin/corpus/runs \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "content-type: application/json" \
  -d '{"strategy": "random", "limit": 2000}'
# -> {"workflow_instance_id": "...", "status": "queued"}

# check overall corpus state
curl https://ohi.shiftbloom.studio/api/v2/admin/corpus -H "Authorization: Bearer $ADMIN_TOKEN"

# check one run
curl https://ohi.shiftbloom.studio/api/v2/admin/corpus/runs/<run_id> -H "Authorization: Bearer $ADMIN_TOKEN"
```

The Workflow fans work out through the `ohi-corpus-ingest` Queue
(dead-letters to `ohi-corpus-ingest-dlq`), embeds with Workers AI,
writes documents/chunks to D1 (`ohi-prod`), archives raw JSON to R2
(`ohi-corpus-prod`), and upserts vectors into Vectorize
(`ohi-evidence-bge-m3`). Queue delivery is at-least-once — consumers
key on `(run_id, batch)` to stay idempotent; don't "fix" apparent
duplicate processing by removing that key. Read
`corpus-ingestion.ts` before assuming what a new `strategy` value does
— don't guess from the field name.

## Known architectural choices (do NOT reverse without discussion)

Full topology is in CURRENT_ARCHITECTURE.md — these are the choices
worth knowing the *why* of before you suggest reversing them:

1. **One Worker serves frontend + API + MCP on one custom domain**,
   not a split frontend host + separate API backend. Same-origin
   avoids the CORS/Host-header class of problems the prior
   Lambda+CDN setup had, and Workers Static Assets serves the Next.js
   static export natively.
2. **D1 + Vectorize + R2 are the entire corpus store** — no external
   graph DB or vector DB. Everything is a native Workers binding, so
   there's no network hop or CF-tunnel dependency to keep alive.
3. **NLI classification is an ensemble of two Workers AI models**
   (`@cf/google/gemma-3-12b-it` + `@cf/meta/llama-3.3-70b-instruct-fp8-fast`)
   run concurrently, with disagreement downgrading toward
   neutral/refute, plus a separate `relevance_score` that gates
   whether evidence counts toward the support/refute signal at all.
   This replaced a single-model classifier that was scoring basic
   false claims as ~70% true because one loosely-related "support"
   evidence item could dominate `p_true` unpenalized — see the
   evidence-evaluation rework note at the top of
   CURRENT_ARCHITECTURE.md before changing this logic.
4. **Cloudflare Turnstile + zone WAF/rate-limit rules gate public
   traffic**, not an app-level account system. This is a public,
   unauthenticated, non-profit service — bot mitigation has to happen
   without requiring users to sign up.
5. **Corpus growth runs through a Cloudflare Workflow**, not a one-shot
   script, specifically for durable execution across many external
   fetches (Wikipedia/Wikidata) with retry/resume built in.

## Critical operational traps

1. **A legacy AWS-era GitHub Actions workflow (`v2-main-deploy.yml`) is
   still wired to `push: branches: [main]`** and still attempts an ECR
   build + Lambda deploy on every push to `main`, alongside the real
   `cloudflare-production.yml` gate. It is not part of the production
   path and its pass/fail says nothing about the Cloudflare deploy.
   Don't read a green/red on that workflow as a signal about prod, and
   don't "fix" it without first confirming with Fabian whether the AWS
   Lambda side is meant to be decommissioned entirely (deleting the
   workflow file is a plausible fix but is a decision, not a given —
   check first). Run `gh run list --workflow=cloudflare-production.yml`
   specifically when you need real deploy status.
2. **GitHub Actions on this repo can be billing-locked** — jobs fail in
   ~3-5s with an annotation like "The job was not started because your
   account is locked due to a billing issue," which looks identical to
   a fast config error at a glance. `gh run view <id>` and read the
   annotation before assuming a workflow bug. When this happens, manual
   `wrangler deploy` from a local authenticated session is the
   legitimate fallback — but **update CURRENT_ARCHITECTURE.md's
   verified-Worker-version line yourself**, since GitHub Actions won't
   record the deploy for you.
3. **`ADMIN_TOKEN` / `TURNSTILE_SECRET_KEY` get overwritten on every
   green CI deploy** from the `OHI_ADMIN_TOKEN` / `TURNSTILE_SECRET_KEY`
   GitHub repo secrets (see Deployment section above). A manual
   `wrangler secret put` rotation that isn't mirrored to the repo
   secret will silently revert on the next successful deploy.
4. **`/api/v2/verify` and `/mcp` trip the zone WAF's scripted-client
   challenge for non-browser clients** (plain `curl`), even with a
   valid admin token — this is intentional bot mitigation, not a bug.
   `GET/POST /api/v2/admin/*` is not behind that challenge. If you need
   to probe `/verify` end-to-end outside a real browser, expect the WAF
   to block it and don't spend time trying to defeat it — that's the
   point of the rule.
5. **Queue consumers must stay idempotent per `(run_id, batch)`** (or
   per job id for the verify queue) — `max_retries` + a DLQ mean
   at-least-once delivery is guaranteed, not exactly-once. Don't add
   logic that assumes a message is processed only once.
6. **`wrangler deploy --dry-run --outdir dist` is the Worker's `build`
   script**, not a real build step separate from deploy validation —
   it's how CI/local checks catch config errors before a real
   `wrangler deploy`. Don't skip it thinking it's redundant with `tsc`.

## Legacy code (retained, not production)

`src/api/` (FastAPI verification service), `src/ohi-mcp-server/`
(standalone MCP package), `gui_ingestion_app/`, `gui_benchmark_app/`,
`docker/`, and `infra/` (Terraform for the pre-Cloudflare AWS stack)
remain in the repo for local development and as migration reference.
None of them are exercised by `cloudflare-production.yml` and none are
in the live request path — see
[docs/CURRENT_ARCHITECTURE.md §1](docs/CURRENT_ARCHITECTURE.md#1-production-architecture)
("No production path depends on..."). Treat changes there as
local-dev-only unless a task explicitly says otherwise, and don't infer
production behavior from that code — read `cloudflare/ohi-worker/src/`
instead.

The user has an explicit hard-rule pattern: **"DO NOT push commits to
remote without explicit user approval."** Local commits are fine; push
only after asking. This applies regardless of which part of the repo
you're touching.

## Git workflow

- **Active branch is `main`.** `main` auto-deploys to Cloudflare via
  `.github/workflows/cloudflare-production.yml` on every push (see
  Deployment above, and trap #1 about the stale AWS workflow that also
  fires).
- **Push to `main` still requires Fabian's explicit approval (Gate
  G2)**, once per session. Subsequent pushes in the same session are
  fine after G2 opens.
- **Per-feature branches**: `f/<name>`, `fix/<name>`, `feat/<name>` off
  `main`. These can be pushed freely to remote for PRs / draft state —
  only the merge into `main` is gated.
- Commit messages: Conventional Commits. Heavy on the *why*, files
  involved, and "end state verified by" lines with concrete test
  evidence. No emoji. Commit-body length is fine — the git log is the
  audit trail for a public-facing open-source project.
- Any commit that changes Cloudflare topology, bindings, models, or env
  vars must update `docs/CURRENT_ARCHITECTURE.md` in the same commit —
  see the SSoT section above.

## User communication

- Be brief. End-of-turn summary: one or two sentences. What changed
  and what's next.
- Don't pad with apologies or meta-commentary.
- Be honest about scope — but **do not give time estimates in hours /
  days / weeks for OHI work**. Fabian reads those as hallucinated
  under-estimates. Frame scope in **code-surface metrics** (files
  touched, LOC, module count) instead. See memory
  `feedback_no_time_scales.md`.
- **Do not conflate different capacity dimensions** when citing numbers
  for any Cloudflare product — e.g. D1 storage size vs. row count,
  Vectorize dimension count vs. index storage, R2 object count vs.
  bytes stored, Workers CPU-time limit vs. wall-clock duration. Always
  state which dimension a number refers to. See memory
  `feedback_gb_ram_vs_storage.md` (written for the old AWS stack — the
  underlying principle still applies to Cloudflare's resource limits).
- When you hit 2-3 consecutive fixes that reveal new symptoms, **stop**
  and present architectural alternatives. Fabian's exact feedback:
  "den Wald vor lauter Bäumen nicht sehen."
- When in doubt about whether an action is destructive or irreversible,
  ask. This includes D1 migrations against `--remote`, secret rotation,
  and force-push.

## Useful runbooks

`docs/runbooks/` predates the Cloudflare migration and most of it
describes decommissioned AWS/Terraform/PC-Tailscale infrastructure
(`bootstrap-cold-start.md`, `rollback-deploy.md`, `rotate-edge-secret.md`,
`rotate-secret.md`, `vercel-setup.md`, `google-cloud-quota-cap.md`,
`incident-response-basic.md`). **Do not follow their command examples
against production** — verify against `cloudflare/ohi-worker/` and
CURRENT_ARCHITECTURE.md first. Still generally applicable:

- `docs/runbooks/cloudflare-api-token-rotate.md` — CF token scope
  reference (some scopes listed are for the retired Terraform layers;
  check against what `wrangler`/CI actually need before granting).
- `docs/runbooks/pc-compose-start.md` — local Docker stack for legacy
  local-dev services (`docker/compose/pc-data.yml`), unrelated to prod.

No Cloudflare-native runbooks exist yet (deploy/rollback/secret-rotate
procedures for the Worker are documented inline in this file and in
CURRENT_ARCHITECTURE.md instead). Writing dedicated ones is a
reasonable follow-up if this file grows unwieldy.

## Useful references

- Live topology, resource table, deploy commands, last-verified anchor:
  [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) (SSoT).
- API surface: [docs/API.md](docs/API.md).
- Frontend architecture (polling, MCP client): [docs/FRONTEND.md](docs/FRONTEND.md).
- CI/CD definition: `.github/workflows/cloudflare-production.yml`.
- Worker source: `cloudflare/ohi-worker/src/index.ts` (routing, pipeline,
  admin/turnstile auth), `knowledge-tools.ts` (evidence connectors),
  `corpus-ingestion.ts` (corpus Workflow).
