# CLAUDE.md — OHI repo agent instructions

You are working on the Open Hallucination Index (OHI) project. This file
gives you the hard-won operational knowledge the current owner
(Fabian, `fabian@shiftbloom.studio`) and prior agents have accumulated.
**Read it fully before making any non-trivial change.**

## Before anything

1. Read the most recent checkpoint in `docs/superpowers/checkpoints/`
   (sorted by date). It describes the live state of the infra and what
   is known-broken vs known-placeholder.
2. Read the spec and plan files in `docs/superpowers/specs/` and
   `docs/superpowers/plans/` if present (gitignored, local-only — do
   not try to commit them).
3. If the task is a product/pipeline change, read
   `src/api/config/dependencies.py::_initialize_adapters` and
   `src/api/pipeline/pipeline.py::verify` first to understand what's
   actually wired and what's `None`.

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
5. **Embedding service runs on PC** (`docker/pc-embed/`), Lambda calls
   it via CF tunnel — NOT bundled into the Lambda image. Includes
   torch + sentence-transformers + pre-baked `all-MiniLM-L12-v2`
   model. Lambda stays slim (~500 MB instead of 2.4 GB).
6. **Redis is currently disabled** (`REDIS_ENABLED=false` in compute TF
   env). The PC webdis-over-CF-tunnel path is too lossy (webdis speaks
   HTTP, `redis-py` speaks native Redis TCP). Planned migration:
   ElastiCache for Valkey on `cache.t4g.micro` in a new VPC with fck-nat
   for Lambda egress (~$15-20/mo total) once AWS Activate credits land.

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

## Git workflow

- Branch: `feat/ohi-v2-foundation` is the active dev branch. `main` is
  protected (no force-push) and auto-deploys Vercel.
- Merge pattern: squash/fast-forward locally when possible, otherwise
  `--no-ff` with a `Merge feat: ...` message. Push feat and main
  separately after each logical unit of work.
- Commit messages: Conventional Commits. Heavy on the *why*, files
  involved, and "end state verified by" lines with concrete test
  evidence. No emoji. Commit-body length is fine — the git log is the
  audit trail for a public-facing open-source project.
- **Never** push without the user's explicit approval the first time in
  a session. Subsequent pushes within the same session are fine if the
  user has already greenlit the pattern.

## User communication

- Be brief. End-of-turn summary: one or two sentences. What changed
  and what's next.
- Don't pad with apologies or meta-commentary.
- Be honest about scope. Several times this project we've hit "this
  isn't 15 minutes, it's days" and the user wants to hear it early.
  Phase 2 work (NLI, PCG, corpus ingestion, calibration data) is days.
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

- Spec + plan + checkpoints: `docs/superpowers/` (gitignored, local)
- Live traffic path diagram: latest checkpoint, §1
- Secret locations: latest checkpoint, §5
- Phase 2 backlog and priorities: latest checkpoint, §7
