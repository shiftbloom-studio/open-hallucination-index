# CLAUDE.md — OHI repo agent instructions

You are working on the Open Hallucination Index (OHI) project. This file
gives you the hard-won operational knowledge the current owner
(Fabian, `fabian@shiftbloom.studio`) and prior agents have accumulated.
**Read it fully before making any non-trivial change.**

## Before anything

1. Read [`docs/CURRENT_ARCHITECTURE.md`](docs/CURRENT_ARCHITECTURE.md) —
   the **single source of truth** for production topology, env vars, and
   the drift note about Lambda image vs. ECR `:prod` tag. If anything in
   THIS file (CLAUDE.md) contradicts CURRENT_ARCHITECTURE.md, trust
   CURRENT_ARCHITECTURE.md and update CLAUDE.md.
2. Read the most recent checkpoint in `docs/superpowers/checkpoints/`
   (sorted by date). It describes the live state of the infra and what
   is known-broken vs known-placeholder.
3. Read the spec and plan files in `docs/superpowers/specs/` and
   `docs/superpowers/plans/` if present (gitignored, local-only — do
   not try to commit them).
4. If the task is a product/pipeline change, read
   `src/api/config/dependencies.py::_initialize_adapters` and
   `src/api/pipeline/pipeline.py::verify` first to understand what's
   actually wired and what's `None`.

## Single Source of Truth for architecture

`docs/CURRENT_ARCHITECTURE.md` is authoritative for:

- Production topology (what talks to what, which AWS services, which
  PC services, which external APIs)
- Live Lambda env-var reference
- Planned-but-unshipped changes
- Last-verified-against-prod anchor (date + `main` tip + Lambda digest)

**Rule:** any architectural change (new service, moved service,
changed env var, changed backend, changed external API dependency)
MUST update CURRENT_ARCHITECTURE.md in the **same commit** as the code
or TF change. Reviewers should reject PRs that change architecture
without updating the SSoT.

Other docs (`README.md`, `docker/README.md`, `docs/API.md`,
`docs/FRONTEND.md`) may summarize or reference architecture but should
NOT duplicate the authoritative details — link back to
CURRENT_ARCHITECTURE.md instead. Runbooks describe procedures, not
topology; if a runbook needs to state a topology fact, it should cite
CURRENT_ARCHITECTURE.md.

## Anti-hallucination protocol (mandatory for every agent)

Verified session-lapses on this project include: citing a file's line
number from memory (was wrong by 40+ lines); writing a shell recipe
for ECR retagging that silently corrupted the manifest digest;
writing smoke assertions that were structurally unreachable given the
pipeline's current state (caused two deploy rollbacks in a row); and
assuming handoff files were saved to disk when they were only pasted
into chat. All cases: memory-recall without tool verification. This
section is the project-wide countermeasure.

**Every agent (including the coordinator) MUST:**

- **Read before claim.** Never state what a file contains (line
  numbers, function signatures, imports, specific strings, types,
  interfaces) without having `Read` it in the current session. A
  prior session's read does not count if the file may have changed.
- **Grep before claim.** Never quote a symbol (function name,
  variable, class, import path) from memory. Run `Grep` first.
- **Git-command before claim.** Never state what's on a branch,
  what's merged, what the feat tip is, or what a merge-base is
  without running `git log`, `git diff`, `git rev-parse`,
  `git branch`, or `git merge-base`.
- **Query prod before claim.** Before claiming Lambda's current
  image, `/health/deep` state, DynamoDB record contents, Vercel
  deployment state, or any other prod condition, run the `aws` /
  `curl` / `gh` / `ls` command.
- **Fetch docs before API claims.** Never assert parameter names,
  return-type shapes, or behaviour of an external library/SDK/API
  from training-data recall. Use `WebFetch` (or `context7` MCP when
  connected) to pull the current docs; pin SDK versions in code that
  depends on the shape.
- **Run the tests before saying they pass.** Never state "pytest
  green" or "228 passed" from memory. Run the canonical runline:
  `pytest -q tests -m "not infra"`.
- **Test or runbook-source every shell recipe for destructive
  operations.** Especially for force-push, ECR retag, secret
  rotation, `terraform destroy`, `update-function-code`. If writing
  a new recipe, validate each step before handing it to another
  agent.
- **Trace the code path before writing smoke assertions.** Before
  asserting `p_true > X`, `refuting_evidence non-empty`,
  `fallback_used != "general"`, or any semantic property in a smoke
  script, Read the current pipeline code to confirm the expected
  value is actually produced by this deploy's codebase. Asserting
  forward on not-yet-shipped behavior is structurally unreachable
  and will roll back the deploy.
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
command; (d) for deployed state — run the `aws` / `curl` / `gh`
command; (e) for external APIs — `WebFetch` the docs. State
uncertainty explicitly when you haven't verified. "I don't know" is
a valid and preferred response to a confident-wrong one.
</investigate_before_answering>
```

This mirrors Anthropic's official Opus 4.7 prompting-guide pattern
for reducing hallucinations in agentic coding workflows.

## Environment assumptions

- **OS:** Windows 11 + Git Bash. cwd does **not** persist between `Bash`
  tool calls — always `cd /c/Users/Fabia/Documents/shiftbloom/git/open-hallucination-index`
  explicitly, or use absolute paths.
- **`source ~/.ohi-deploy-env` at the start of every `Bash` call that
  needs AWS / Cloudflare / Vercel credentials.** It sets AWS_PROFILE,
  CLOUDFLARE_API_TOKEN, VERCEL_API_TOKEN, and several TF_VAR_* exports.
  The file is chmod 600 and not in git.
- Prefix `MSYS_NO_PATHCONV=1` on any AWS CLI call that passes a path
  starting with `/aws/`, `/var/`, `/opt/` etc. — otherwise Git Bash
  mangles it into `C:\Program Files\Git\aws\...` and AWS rejects.
- **AWS CLI v2 has a Windows `charmap` Unicode bug.** Log lines that
  contain non-cp1252 characters (e.g. `→`, `—`, emoji) crash
  `aws logs tail` / `filter-log-events` with a UnicodeEncodeError mid-
  stream. Use **boto3** via Python when fetching logs, and wrap every
  printed string in `.encode('ascii', errors='backslashreplace').decode()`.
  Same class of trap applies to **Python stdout on Windows** — prefix
  smoke scripts that print Unicode with `PYTHONIOENCODING=utf-8`.
- **Git-Bash `/tmp/...` paths are not resolvable from Windows-native
  Python stdlib `open()`.** If a smoke script needs to read a `/tmp/`
  file, pipe via `cat /tmp/foo.json | python -c "import sys, json; d =
  json.loads(sys.stdin.read()); ..."` instead of `open('/tmp/foo.json')`.
- **Canonical pytest runline:** `pytest -q tests -m "not infra"` (NOT
  `pytest -q src/api/tests tests -m "not infra"` — `src/api/tests/`
  doesn't exist; tests live under `tests/`). `pytest.ini` declares
  `testpaths = tests` so the arg is redundant but harmless.
- **Cross-worktree sys.modules purge** is handled session-scoped by
  `tests/unit/conftest.py` (landed at commit `dfd2fe6` as a D1 follow-up).
  New test files should NOT copy a per-test purge snippet — trust the
  conftest. If extending the conftest's purge set, guard on
  `sys.modules["interfaces"].__file__` (not `server`) to avoid breaking
  module-identity checks in other tests.

## Forbidden files (user WIP — do NOT edit without explicit approval)

Historical list — the user has relaxed some of these on a case-by-case
basis. **Ask every time before touching:**

- `src/api/config/settings.py`
- `src/api/config/dependencies.py` (unlocked during bring-up but still
  treat as high-sensitivity; always show the user the diff first)
- `src/api/adapters/null_graph.py`
- `gui_ingestion_app/*`
- `nul` (stray file; do not touch)
- `api/` (the older module tree; current code lives under `src/api/`)

The user has an explicit hard-rule pattern: "DO NOT push commits to
remote without explicit user approval". Local commits are fine; push
only after asking.

## Known architectural choices (do NOT reverse without discussion)

1. **Lambda + API Gateway HTTP API + CF proxy** for the API. NOT Lambda
   Function URL as the CF origin — API Gateway is required because
   Function URLs validate the HTTP Host header against their own
   hostname and CF free/pro cannot override Host (Origin Rules are
   Enterprise-only). See `infra/terraform/compute/api_gateway.tf` +
   `infra/terraform/cloudflare/api_gateway_custom_domain.tf`.
2. **Flat CF naming** (`ohi-api.shiftbloom.studio`, `ohi-neo4j.*`, etc.).
   NOT nested `*.ohi.shiftbloom.studio`. CF free-tier Universal SSL
   covers only 1-level wildcards. DNS + Access apps are all at zone root
   with an `ohi-<label>` prefix. The `apex_subdomain="ohi"` variable in
   cloudflare/ TF exists solely to keep the frontend apex at
   `ohi.shiftbloom.studio`.
3. **Native Gemini adapter** at `src/api/adapters/gemini.py` for LLM
   calls. The OpenAI-compat shim (`adapters/openai.py`) is kept in-tree
   as a fallback but NOT wired in prod (select via `LLM_BACKEND=openai`
   env var). Gemini's OpenAI-compat shim silently drops
   `safetySettings` + `generationConfig.thinkingConfig` — unacceptable
   for a hallucination-detection product that MUST process factually-
   wrong claims about real people.
4. **Neo4j Aura Pro** (managed, AWS Frankfurt) is the graph store — NOT
   the PC neo4j. Switched because bolt (TCP:7687) doesn't work through
   CF's free-tier HTTPS tunnel. Aura instance id `0193408e`, URI in
   compute TF's `var.neo4j_uri`, password in SM `ohi/neo4j-credentials`.
5. **Embedding service runs on AWS Bedrock** (Titan Text Embeddings V2,
   1024-dim). `OHI_EMBEDDING_BACKEND=bedrock` in Lambda env, consumed by
   the tri-mode facade in `src/api/adapters/embeddings.py`
   (`local` = in-process sentence-transformers for dev,
   `remote` = HTTP to `pc-embed` container for legacy,
   `bedrock` = prod). Historic note: embeddings ran on PC
   (`docker/pc-embed/`, MiniLM-L12-v2) until 2026-04-21 — the
   container is retained for local-dev use.
6. **Reranking runs on AWS Bedrock** (Cohere `rerank-v3-5`).
   `BEDROCK_RERANK_ENABLED=true`. `GraphRetriever` cascade is
   Qdrant ANN (40 candidates) → Aura passage fetch → Cohere rerank
   (top-12). Both collections are empty until Wave 3 Phase E ingestion
   runs, so today the active evidence path is MCP sources (MediaWiki +
   Wikidata + DBpedia).
7. **Redis is currently disabled** (`REDIS_ENABLED=false` in compute TF
   env). The PC webdis-over-CF-tunnel path is too lossy (webdis speaks
   HTTP, `redis-py` speaks native Redis TCP). Planned migration:
   ElastiCache for Valkey on `cache.t4g.micro` in a new VPC with fck-nat
   for Lambda egress (~$15-20/mo total) once AWS Activate credits land.
8. **Neo4j Aura Pro → PC-local Neo4j over Tailscale (planned).** The
   Aura 64 GB RAM limit + managed-service cost are incompatible with
   full biomed ingestion (PubMed + OpenAlex + enwiki + Wikidata) and
   the non-profit / intermittent-availability use-case. Tailscale
   solves the CF-free-tier bolt/TCP:7687 exposure problem that
   originally forced the move TO Aura. Migration is not shipped yet —
   track in `docs/CURRENT_ARCHITECTURE.md §4`.

## Critical operational traps

Every one of these has bitten us. Watch for them.

1. **Lambda Function URL 403 `AccessDeniedException`** has four distinct
   causes that look identical:
   - Resource policy needs BOTH `FunctionURLAllowPublicAccess`
     (InvokeFunctionUrl) AND `FunctionURLAllowInvokeAction` (InvokeFunction
     with `lambda:InvokedViaFunctionUrl=true`) on this account. Fabian
     added the second one via Console.
   - Container crashed during init (bad entrypoint, missing module,
     read-only-filesystem writes). Surfaces as 403 on URL invocation.
   - Host header mismatch (URL validates Host). That's why we went to
     API Gateway + custom domain.
   - Account-level "Block public access for function URLs" — this
     account didn't have it, but other AWS accounts might.
2. **CF free-tier rate-limit rules** are capped at period=10s and
   mitigation_timeout=10s. Any other value returns "not entitled to use
   the period X". Cap is also 1 rule per `http_ratelimit` phase per zone.
3. **CF Transform Rules cannot modify Host header** (error 20086/20087).
   Origin Rules can, but **Origin Rules host_header override is
   Enterprise-only**. Don't keep retrying.
4. **CF free tier only supports creating ROOT zones**, not subdomain
   zones. `ohi.shiftbloom.studio` as a CF zone fails with "Please ensure
   you are providing the root domain and not any subdomains" (error
   1116). This is why we flattened naming instead.
5. **AWS Budgets only accepts USD.** Not EUR. compute/ TF already in USD.
6. **Docker Lambda builds** require `--provenance=false --platform
   linux/amd64` or Lambda rejects the image manifest as "not supported".
7. **`terraform apply` exit 0 can be a lie.** The pattern we use is
   `(... ; echo "DONE exit=$?") > /tmp/x.log 2>&1` — the outer shell
   group's exit is whatever `echo` returns (always 0). Always grep the
   log for `Error:`.
8. **Gemini's OpenAI-compat endpoint rejects null-valued optional fields**
   (e.g. `stop: null` → HTTP 400 "Value is not a string: null"). Our
   adapter omits None values from the kwargs dict. Don't reintroduce
   `stop=None` style.
9. **Compute TF's `aws_lambda_permission.public_url`** has a history of
   state drift — the permission exists in AWS but TF thinks it needs
   to be created, so apply hangs 5m on "Still creating" before 409ing.
   Already imported once; can recur. Treat the 409 as benign since
   Lambda function itself gets modified before that error fires.
10. **Qdrant client (qdrant-client 1.16) hides its httpx AsyncClient**
    four levels deep: `self._client._client.http.<any>_api.api_client._async_client`.
    `src/api/adapters/qdrant.py` walks to it via `dir()` + `endswith("_api")`
    to inject the CF-Access-Client-Id/Secret headers. If qdrant-client
    restructures, rewrite the walker.
11. **`aws_secretsmanager_secret_version` resources have
    `recovery_window_in_days=0`.** `put-secret-value` REPLACES with no
    recovery. Always save the old value to a password manager before
    rotating.
12. **`aws lambda get-function-configuration --query ImageUri` returns
    literal `"None"` for image-backed Lambdas** (Stream E1 discovery).
    Use `aws lambda get-function --query 'Code.ImageUri'` instead —
    that always returns the resolved-digest URI (`...@sha256:...`) of
    the currently running image. Pin the immutable digest form, not a
    mutable tag like `:prod`, when saving rollback anchors.
13. **Response field is `document_score`, NOT `doc_score`** (E1
    discovery). Any smoke parser that uses `d.get('doc_score')`
    silently returns `None`. Use `d.get('document_score')`.
14. **ECR `:prod` tag drifts after a rollback** (E1 trap). Rolling
    Lambda back via `update-function-code --image-uri <prev-digest>`
    updates Lambda but leaves `:prod` pointing at the failed image in
    ECR. Always re-tag `:prod` to the rollback digest via
    `aws ecr put-image --image-tag prod --image-manifest "$MANIFEST"`
    so next-deploy agents don't reason incorrectly about what's in prod.
15. **Autonomous deploy protocol** — Fabian disabled manual G3/G8 gates
    on deploys after Wave 1 pacing pain. Deploy agents follow the
    protocol in `docs/superpowers/plans/2026-04-18-phase2-orchestration.md`
    §5.3: pre-deploy checkpoint → TF-plan safety audit → image build +
    deploy → two-tier smoke (per-deploy + optional end-of-phase) →
    auto-rollback on smoke fail → handoff with status. **One deploy
    attempt per session, one rollback attempt, no retry loops.**
16. **Lambda digest ≠ ECR `:prod` tag after a rollback.** Verify BOTH
    before claiming prod state. As of 2026-04-22 the Lambda runs
    `sha256:0e3676a2...` (hotfix-20260421-nli-refute) while ECR `:prod`
    points at `sha256:7578dd4b...` (post-Bedrock deploy of `cf77b24`).
    Always run `aws lambda get-function --query 'Code.ImageUri'` for
    the live digest and `aws ecr describe-images --image-ids
    imageTag=prod` for the tag pointer, and compare. See trap #14 for
    the retag recipe.

## Git workflow

- **Active branch is `main`.** The `feat/ohi-v2-foundation` branch was
  merged into `main` at commit `c5010cc` ("v2.0 Wave 2 cutover — main
  retired, feat is content"). All v2 development since then has landed
  directly on `main`, and `main` auto-deploys Lambda via the
  `.github/workflows/v2-main-deploy.yml` workflow (build → ECR → Lambda
  update → health check → auto-rollback).
- **Push to `main` still requires Fabian's explicit approval (Gate
  G2)**, once per session. Subsequent pushes in the same session are
  fine after G2 opens.
- **Per-feature branches**: `f/<name>`, `fix/<name>`, `feat/<name>` off
  main. These can be pushed freely to remote for PRs / draft state —
  only the merge into `main` is gated.
- Commit messages: Conventional Commits. Heavy on the *why*, files
  involved, and "end state verified by" lines with concrete test
  evidence. No emoji. Commit-body length is fine — the git log is the
  audit trail for a public-facing open-source project.
- Current `main` tip at time of this doc update: `cf77b24` (docs: add
  CURRENT_ARCHITECTURE + OHI_onepager.pdf). Update on next major
  milestone.

## User communication

- Be brief. End-of-turn summary: one or two sentences. What changed
  and what's next.
- Don't pad with apologies or meta-commentary.
- Be honest about scope — but **do not give time estimates in hours /
  days / weeks for OHI work**. Fabian reads those as hallucinated
  under-estimates. Frame scope in **code-surface metrics** (files
  touched, LOC, module count, stream count) instead. See memory
  `feedback_no_time_scales.md`.
- **Do not conflate RAM vs storage vs vector-index carve-out** when
  citing GB numbers for any cloud service (Aura, DynamoDB, OpenSearch,
  etc.). Always explicit: `<RAM>GB RAM / <disk>GB disk`. See memory
  `feedback_gb_ram_vs_storage.md`.
- When you hit 2-3 consecutive fixes that reveal new symptoms, **stop**
  and present architectural alternatives. Fabian's exact feedback:
  "den Wald vor lauter Bäumen nicht sehen."
- When in doubt about whether an action is destructive or irreversible,
  ask. The user has explicit hard rules about force-push, rotating
  secrets without saving, and touching WIP files.

## Useful runbooks

- `docs/runbooks/bootstrap-cold-start.md` — for recreating from scratch
- `docs/runbooks/cloudflare-api-token-rotate.md` — CF token scopes list
- `docs/runbooks/pc-compose-start.md` — PC docker stack bring-up
- `docs/runbooks/rollback-deploy.md` — emergency Lambda image rollback
- `docs/runbooks/rotate-secret.md` — generic SM secret rotation
- `docs/runbooks/google-cloud-quota-cap.md` — Gemini API cost backstop
- `docs/runbooks/vercel-setup.md` — first-time Vercel wiring

## Useful references

- Spec + plan + checkpoints + handoffs: `docs/superpowers/` (gitignored,
  local). Four subdirs: `specs/`, `plans/`, `checkpoints/`, `handoffs/`.
- Active plan: `docs/superpowers/plans/2026-04-18-phase2-orchestration.md`
  — Wave 2+3 orchestration, autonomous deploy protocol (§5.3), gates
  table (§6.1), rollback recipe (§6.3).
- Live traffic path diagram: `docs/CURRENT_ARCHITECTURE.md` §1 (SSoT).
- Secret locations: latest checkpoint, §5.
- Phase 2 backlog and priorities: latest checkpoint, §7.
- Stream handoffs (sequential dev state record): `docs/superpowers/handoffs/`
  — stream-a → stream-b → stream-d1 → stream-e1 → stream-d2 → stream-e2.

## Current v2 state (2026-04-22)

`main` tip `cf77b24`. Wave 1 + Wave 2 shipped; Wave 3 code-side
largely landed (PCG + cc-NLI live; corpus ingestion pipeline in
place, collections empty pending Phase E full run):

- **Wave 1–2 (Streams A / B / D1 / F / D2)**: MediaWiki MCP,
  Gemini 3 Pro claim-evidence NLI, NLI wired into the pipeline with
  `asyncio.gather + Semaphore(10)`, Lambda `timeout_s=180`, async
  `/verify` polling (202 + `job_id`, DynamoDB `ohi-verify-jobs`,
  self-async-invoke, frontend poll loop).
- **Wave 2.5 — Bedrock migration (commits `65b9818` → `d2b617a`)**:
  embeddings moved PC → Bedrock Titan v2 (1024-dim); Bedrock Cohere
  rerank-v3-5 added as a new `GraphRetriever` stage;
  `EmbeddingSettings` + `RetrievalSettings` Bedrock fields + DI
  wiring; Lambda IAM for `bedrock:InvokeModel` + `bedrock:Rerank`.
- **Wave 3 — PCG + corpus ingestion**: `pcg_belief_propagation`
  adapter (TRW-BP + LBP fallback + Gibbs MCMC sanity);
  `ClaimClaimNliDispatcher` (OpenAI GPT-5.4 primary + Gemini
  fallback — primary inactive today because `OHI_OPENAI_API_KEY` is
  unset); offset-based checkpoint recovery for all ingestion passes;
  `ohi-ingest` console script; `EndToEndHealth` dashboard +
  `/health/live` + `/health/ready`.
- **Post-deploy drift**: Lambda is currently on the pre-Bedrock-merge
  hotfix digest (`sha256:0e3676a2...`, `hotfix-20260421-nli-refute`)
  after a silent rollback on 2026-04-21 21:29 UTC. ECR `:prod` points
  at `sha256:7578dd4b...` (`cf77b24` content). Env vars are
  Bedrock-configured and the hotfix image is recent enough to support
  the Bedrock backend selector. Running prod is functional but not on
  the content of `main` tip — next deploy will roll forward.

**Next planned change: Neo4j Aura → PC-local over Tailscale.** Driver:
Aura 64 GB RAM limit + cost vs. full biomed ingestion + non-profit
use-case. Requires Lambda networking adjustment (Tailscale extension
/ sidecar), `NEO4J_URI` env change, and clean 503 behaviour when the
Tailnet peer is offline. Full detail in
[docs/CURRENT_ARCHITECTURE.md §4](docs/CURRENT_ARCHITECTURE.md#4-planned-changes-not-yet-shipped).
