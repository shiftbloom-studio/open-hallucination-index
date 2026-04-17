# OHI v2 — Infra Sub-Project Execution Complete (Handoff)

**Date:** 2026-04-17
**Purpose:** Let the sub-project-3 (frontend) agent resume with zero infra context loss. Pure status + decisions record; no instructions.

---

## 1. Where everything lives

| Artifact | Branch / location |
|---|---|
| **Infra code** (Terraform, Docker, workflows, middleware, runbooks, validation scripts) | `feat/ohi-v2-foundation` (local, not pushed) — 47 new commits on top of the original 9 algorithm commits |
| Infra spec | `docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md` (local `main` only — gitignored on feat branches) |
| Infra implementation plan | `docs/superpowers/plans/2026-04-16-ohi-v2-infrastructure-implementation.md` (same — local `main` only) |
| Algorithm spec (companion) | `docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md` |
| Pre-session checkpoint | `docs/superpowers/checkpoints/2026-04-16-infra-brainstorm-handoff.md` |
| **This checkpoint** | `docs/superpowers/checkpoints/2026-04-17-infra-execution-complete.md` |
| Pre-infra-work user WIP | `git stash@{0}` on `main`, message `"user-wip-preserved-infra-2026-04-17"` — 4 files unrelated to v2 |

Nothing has been pushed to remote. The branch structure locally is:

```
origin/feat/ohi-v2-foundation  (9 algorithm commits)
         └── feat/ohi-v2-foundation  (local) — 47 more infra commits
```

---

## 2. What's been decided and locked in (across sub-projects 1 + 2)

### Sub-project 1 (algorithm) — shipped
Already locked in algorithm spec. Unchanged by infra work.

### Sub-project 2 (infrastructure) — shipped (this work)

**Edge + public traffic**
- Public edge is **Cloudflare free tier**, not AWS CloudFront. Single delegated zone `ohi.shiftbloom.studio` managed entirely by Cloudflare via Terraform.
- All of Cloudflare is Terraformed: zone records, WAF custom rules, rate limits (100/min on `/verify`, 1000/h global), cache rules, Tunnel, Access applications + service token, Transform Rule that injects `X-OHI-Edge-Secret`.
- Free-tier capabilities used: Free Managed Ruleset + Bot Fight Mode + custom rules. Not the paid CF Managed / OWASP Core rulesets.

**AWS surface**
- AWS = Lambda (container via AWS Lambda Web Adapter) + Secrets Manager + S3 (private + public buckets) + KMS + CloudWatch + SNS + Budgets + ECR + DynamoDB (state lock). No CloudFront, no WAF, no ACM, no Route53.
- Region: `eu-central-1`. No `us-east-1` alias (no CloudFront ⇒ no ACM requirement).
- Lambda Function URL `auth_type = NONE`, `invoke_mode = RESPONSE_STREAM`. Middleware (`EdgeSecretMiddleware`) rejects any request missing `X-OHI-Edge-Secret`.

**Data-store location**
- All four data stores (Neo4j, Qdrant, Postgres, Redis) live on the user's home PC in Docker Compose.
- Cloudflare Tunnel (outbound-only, no inbound ports) exposes them as HTTP endpoints. PostgREST + WebDIS sidecars convert Postgres + Redis to HTTP.
- Compose has two profiles: `pc-prod` (no host ports, cloudflared enabled) and `local-dev` (host ports bound on 127.0.0.1 only, separate volumes, cloudflared disabled).

**CI/CD**
- GitFlow: `feature/* → develop → main → tag v*.*.*`
- Tag `v*.*.*` auto-deploys **compute layer only**. Other layers (storage, secrets, cloudflare, observability) apply via manual `workflow_dispatch`.
- OIDC federation for AWS (no long-lived keys). Cloudflare API token lives in GH secret `CLOUDFLARE_API_TOKEN`.
- Five workflows: `test.yml`, `infra-plan.yml`, `infra-apply.yml`, `release.yml`, `bootstrap-drift.yml`. Sixth workflow `calibration-seed.yml` is manual dispatch.

**Cost posture**
- AWS hard cap €150/mo, forecast alarm €100, actual alarm €140. Expected steady-state AWS: ~€7–12/mo.
- Gemini API uncapped in Phase 1 per explicit user directive (R10 accepted risk) — Google Cloud console-side hard quota cap is the ultimate tripwire.

**Secrets**
- 8 AWS Secrets Manager entries. Manual-seeded (gemini key, internal bearer, labeler tokens, pc-origin creds, neo4j creds, cf-edge-secret) vs TF-managed values (cloudflared tunnel token, cf-access-service-token — values written back by the `cloudflare/` layer).
- `SecretsLoader` in `src/api/config/secrets_loader.py`, 10-min TTL cache. Bootstrap-grace semantics for the two CF-populated secrets (tolerate missing/empty during cold start between layer applies).

**Observability**
- CloudWatch dashboard, 4 metric filters (PipelineError, RateLimitApp, PCOriginTimeout, ColdStart), SNS → email to `fabian@shiftbloom.studio`.
- Log retention 7d.

**DNS**
- Single delegated CF zone `ohi.shiftbloom.studio`. User adds 2–4 NS records at their current DNS provider once; every other DNS record is Terraformed.

**Naming + tagging**
- Prefix `ohi-` on every resource; tags: `Project=ohi`, `Environment=prod`, `Layer=<…>`, `ManagedBy=terraform`, `CostCenter=ohi`.

### Sub-project 3 (frontend) — not started, NO decisions yet

---

## 3. Single open decision that blocks the frontend sub-project

The infra `cloudflare/dns.tf` currently points the apex `ohi.shiftbloom.studio` directly at the Lambda Function URL:

```hcl
resource "cloudflare_record" "apex" {
  name    = "@"
  content = local.lambda_fn_hostname  # API
  proxied = true
}
```

That means **as of today, the API is at the root of the domain** and there is no slot for a frontend at the root. Every plausible frontend architecture requires this to change in one of three ways.

### Option A — frontend at apex, API at `/api/*`

| Aspect | Value |
|---|---|
| Apex `ohi.shiftbloom.studio` → | **Frontend origin** |
| API → | `https://ohi.shiftbloom.studio/api/v2/*` via CF routing |
| Frontend host options | Cloudflare Pages, Vercel, S3 + CF origin |
| Infra change scope | Rewrite `cloudflare/dns.tf` apex CNAME; add a CF Worker or "Configuration Rules" route that sends `/api/*` to the Lambda Function URL; add `X-OHI-Edge-Secret` injection on the API path only |

### Option B — API at apex (current), frontend at subdomain

| Aspect | Value |
|---|---|
| Apex `ohi.shiftbloom.studio` → | **Lambda (current)** |
| Frontend → | `app.ohi.shiftbloom.studio` (or similar) |
| Frontend host options | Cloudflare Pages, Vercel, S3 + CF origin |
| Infra change scope | Minimal: add one `cloudflare_record` for the frontend subdomain in `cloudflare/dns.tf`; no WAF/rate-limit/cache changes needed |

### Option C — frontend embeds API calls via Next.js server routes

| Aspect | Value |
|---|---|
| Apex `ohi.shiftbloom.studio` → | **Next.js frontend** |
| API → | `/api/*` handled by Next.js server-side proxy (matches the existing V1 pattern at `src/frontend/src/app/api/ohi/[...path]/route.ts`) |
| Frontend host options | Vercel (natural fit for Next.js server routes) or Cloudflare Pages with Workers |
| Infra change scope | Same as Option A on the Cloudflare side; additionally, Lambda's `X-OHI-Edge-Secret` validation may need relaxing if the Next.js server reads it server-side and forwards — trust model to be designed |

---

## 4. My recommendation for the frontend architecture plan

**Option A with Cloudflare Pages as the frontend host.**

Trade-offs, honestly:

| Dimension | A + CF Pages (recommended) | A + Vercel | B + anything | C + Vercel |
|---|---|---|---|---|
| Cost | €0 (free tier) | €0 (hobby) | €0 | €0 (hobby) |
| Vendor consolidation | 1 (Cloudflare) | 2 (Cloudflare + Vercel) | 1–2 | 2 |
| API token scope reuse | Yes — same CF API token can Terraform Pages via `cloudflare_pages_project` | Needs Vercel token/API | Yes | Needs Vercel token |
| CDN/edge quality | Excellent | Excellent | Excellent | Excellent |
| Next.js RSC / ISR support | Good (Pages has Workers runtime for Next; some edge-function limits) | Best-in-class (Next is Vercel's thing) | N/A | Best-in-class |
| UX at apex (single URL) | Yes | Yes | No (needs subdomain) | Yes |
| Infra churn vs. what I shipped | Medium (apex CNAME rewrite + `/api/*` route) | Medium (same as A) | Tiny (add subdomain record) | Medium-plus (auth trust model) |
| Streaming SSE for `/verify/stream` through the layer | Works — CF Worker routes to Lambda Function URL, streaming preserved end-to-end | Works — Vercel forwards to Lambda Function URL | N/A | Works, but Next.js server is in the critical path — one more hop |

**Why I'd pick A + CF Pages:**
- One vendor, one API token, one billing surface. Clean operational boundary.
- Terraformed end to end (the `cloudflare/` layer can be extended with `cloudflare_pages_project` + `cloudflare_pages_domain` resources — very small addition).
- The `/api/*` route is a single Cloudflare Configuration Rule or Worker — fits cleanly into the existing `cloudflare/` Terraform layer.
- No second identity provider to worry about for deploys.

**Why I'd seriously consider Vercel instead (Option A + Vercel):**
- Next.js features (ISR, RSC, server actions, middleware) are first-class on Vercel, second-class on Pages.
- If the frontend does any non-trivial server-side rendering, Vercel gives you less friction out of the box.
- Adds one more vendor, but Vercel's free tier is generous enough that it doesn't materially affect the €150/mo budget.

**Why I'd actively avoid Option B:**
- Two URLs (`ohi.shiftbloom.studio` for API, `app.ohi.shiftbloom.studio` for frontend) is a worse UX for an open-access public service. The root should be the thing users see.

**Why I'd avoid Option C unless the frontend-plan agent specifically wants Next.js server routes as a proxy:**
- Puts the Next.js server on every API call's critical path — extra hop, extra cold-start, extra trust boundary.
- Complicates the edge-secret model (Next.js server has to hold the edge-secret; rotation is across two systems).

---

## 5. What specifically will change in infra once sub-project 3 picks an option

For the frontend agent's benefit — this is what their architecture plan will need to coordinate with. I'm listing the exact files and lines, not prescribing what to do.

**If Option A (any frontend host):**
- `infra/terraform/cloudflare/dns.tf` — `cloudflare_record.apex` stops pointing at the Lambda Function URL, starts pointing at the frontend host. Lambda Function URL becomes reachable only via a CF Configuration Rule (new resource, currently absent).
- `infra/terraform/cloudflare/waf.tf` / `cache.tf` — `/api/*` path patterns may need separation from root-path patterns; specifically the rate-limit rule on `/api/v2/verify` stays the same, but cache rules may need adjustment since root is no longer the API.
- `infra/terraform/cloudflare/edge_secret.tf` — the Transform Rule currently fires on `(true)` (every request). With the apex becoming the frontend, we likely want it to fire only on `(http.request.uri.path matches "^/api/")` so we don't inject the header into frontend asset requests. Small change.
- New file likely needed: `infra/terraform/cloudflare/pages.tf` (if CF Pages) OR a Terraform Vercel provider block in a new `infra/terraform/vercel/` layer (if Vercel).

**If Option B (API at apex, frontend at subdomain):**
- `infra/terraform/cloudflare/dns.tf` — add one `cloudflare_record` for the frontend subdomain. That's it.
- No WAF, rate-limit, cache, or edge-secret changes needed.

**If Option C (Next.js server as API proxy):**
- All of Option A's changes.
- Plus: `src/api/server/middleware/edge_secret.py` needs a design decision — does Next.js server forward the header (and hold the secret)? Or does Lambda trust a different signal from the Next.js server?
- A new secret entry in `infra/terraform/secrets/main.tf` (a "frontend-to-Lambda bearer") may be required.

---

## 6. The infra agent (me) is paused waiting for input

I will adapt everything above once the sub-project-3 frontend-plan agent's output is shared with me. Concretely I will:
1. Read the frontend plan in full.
2. Revise the infra spec (`docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md`) to reflect the integration points — NOT by re-deriving, but by editing the §0, §1, §7 sections to match.
3. Land new TF code on `feat/ohi-v2-foundation` (a second wave of commits on top of the current 47).
4. Re-run the phase validation scripts to confirm nothing broke.

I will NOT pre-empt the frontend decision. The apex CNAME, the WAF path patterns, the edge-secret rule expression, and the Pages-vs-Vercel choice are all frontend-agent decisions.

---

## 7. One residual infra concern worth the frontend agent knowing

The existing `src/api/server/app.py` (on `feat/ohi-v2-foundation`) registers `EdgeSecretMiddleware` env-gated: `if OHI_CF_EDGE_SECRET_ARN is set`. Terraform compute-layer always sets this in Lambda's environment, so in prod the middleware is ALWAYS active. Every incoming request must carry the right `X-OHI-Edge-Secret` header.

`/health/live` is exempt for Lambda's own runtime liveness checks. All other paths — including any frontend-facing API call — require the header. The frontend plan needs to decide:
- If the frontend calls the API through Cloudflare (Options A or C), CF's Transform Rule injects the header. Fine.
- If the frontend calls the API from client-side JS via a public API URL like `ohi.shiftbloom.studio/api/*` that goes through CF, Transform Rule still injects. Fine.
- If the frontend ever tries to call the Lambda Function URL directly (`<fn-id>.lambda-url.eu-central-1.on.aws`), it gets 403. By design.

No decision needed — just awareness.

**End of handoff.**
