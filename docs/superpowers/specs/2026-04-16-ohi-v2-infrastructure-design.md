# OHI v2 — Infrastructure Design

**Status:** Draft, awaiting review
**Author:** Claude (Opus 4.7) for Fabian Zimber
**Date:** 2026-04-16
**Scope:** AWS infrastructure sub-project of the three-part OHI v2 rewrite. Does NOT cover the algorithm (already specified) or the Next.js frontend rewrite (separate sub-project).
**Companion docs:**
- `docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md` (algorithm spec — §11 defines WAF / rate-limit / cost-ceiling / Secrets constraints this spec must satisfy)
- `docs/superpowers/plans/2026-04-16-ohi-v2-implementation.md` (algorithm implementation plan)
- `docs/superpowers/checkpoints/2026-04-16-infra-brainstorm-handoff.md` (handoff; Section 1 locked here)

---

## 0. What this spec decides vs defers

**Decides (ready for implementation-plan authoring):**
1. End-to-end AWS architecture and traffic path
2. Terraform repo layout, layering, state backend, bootstrap approach
3. CI/CD pipeline (GitHub Actions + OIDC + ECR + plan-on-PR, apply-on-dispatch)
4. Secrets Manager entries, seeding discipline, rotation posture
5. Observability (CloudWatch dashboard, metric filters, alarms) and cost controls (AWS Budgets, Gemini daily ceiling)
6. PC-side Docker Compose for the four data stores + Cloudflare tunnel client + two HTTP proxy sidecars
7. DNS + Cloudflare Tunnel wiring (what records you add at your DNS provider)
8. Phased rollout (five ordered phases I.0 → I.4 with explicit gate criteria)
9. Risks + items explicitly deferred to a future phase

**Defers (out of scope for Phase 1, to be reconsidered later):**
- Multi-AZ / multi-region / backups / failover
- Dev and staging environments
- Automated secret rotation
- GDPR subject-access-request tooling
- Horizontal scaling / provisioned concurrency
- Local-LLM hosting (algorithm phase 1 uses remote Gemini only)
- Third-party APM (Datadog, Sentry)

---

## 1. Architecture recap (from brainstorm Section 1)

```
┌──────────┐       HTTPS          ┌─────────────────────────┐
│  User    │ ───────────────────> │ CloudFront + AWS WAF    │
│  (any)   │                      │ (global, cached, TLS)   │
└──────────┘                      └───────────┬─────────────┘
                                              │  AWS_IAM auth on
                                              │  Lambda Function URL
                                              │  (CloudFront signs via OAC)
                                              ▼
                                  ┌─────────────────────────┐
                                  │ Lambda (container)      │
                                  │  FastAPI +              │
                                  │  AWS Lambda Web Adapter │
                                  │  Region: eu-central-1   │
                                  └─────┬───────────────┬───┘
                                        │               │
                      HTTPS             │               │      HTTPS
                      to Gemini         │               │      to CF Tunnel
                                        ▼               ▼
                              ┌─────────────────┐  ┌──────────────────────┐
                              │ Google Gemini   │  │ Cloudflare Tunnel    │
                              │ gemini-3-flash- │  │ (CF edge, Access     │
                              │ preview         │  │  service-token auth) │
                              └─────────────────┘  └──────────┬───────────┘
                                                              │
                                                              ▼
                                         ┌──────────────────────────────────┐
                                         │ User's PC (Windows + Docker)     │
                                         │  cloudflared ingress             │
                                         │  → neo4j   (HTTP :7474)          │
                                         │  → qdrant  (HTTP :6333)          │
                                         │  → pg-rest (PostgREST → pg:5432) │
                                         │  → webdis  (WebDIS → redis:6379) │
                                         └──────────────────────────────────┘
```

### 1.1 Hard numbers (brainstorm Section 1)

| Parameter | Value | Source |
|---|---|---|
| AWS monthly hard cap | €150 | Budget alarm at €100/140, dashboard at €50 forecast |
| Expected traffic | < 100 monthly users | Sizing assumption |
| LLM provider (Phase 1) | `gemini-3-flash-preview` | User choice; budget-capped separately |
| Region (workload) | `eu-central-1` (Frankfurt) | Lowest PC↔Lambda RTT |
| Region (CloudFront ACM) | `us-east-1` | AWS hard requirement |
| Public hostname | `ohi.shiftbloom.studio` | User-provided; DNS managed at user's existing DNS provider (not Route53) |
| Cloudflare tunnel hostnames | `*.tunnel.shiftbloom.studio` | Delegated subdomain, zone managed by Cloudflare |
| Data layer | On user's PC (Docker) | Zero RDS / ElastiCache / OpenSearch cost |
| Estimated total | ~€9–55/month | Well under cap |

### 1.2 Failure model

- **PC off → tunnel 503.** Lambda catches `httpx.ReadTimeout` / `cloudflared`-origin 502, returns structured `{"status":"resting","reason":"origin-unreachable"}` with HTTP 503 + `Retry-After: 300`.
- **Gemini API error** → L7 fast-fail with `{"status":"llm_unavailable"}`.
- **Gemini daily cost ceiling** hit → `/verify` returns 503 `{"status":"budget_exhausted"}`; `/verdict/{id}`, `/calibration`, `/health/*` continue serving from PC/cache.
- **Lambda cold-start** (~1.5–3s) → accepted; user-facing latency budget is dominated by Gemini (~8–15s), so cold-start is noise.

---

## 2. Terraform module layout (brainstorm Section 2, APPROVED)

### 2.1 Tooling

- **HashiCorp Terraform**, pin `required_version >= 1.10.0, < 2.0.0`.
- Provider `hashicorp/aws ~> 5.80`.
- Second provider alias `aws.use1` pointing to `us-east-1` for CloudFront ACM.
- No Terraform workspaces (one env, `prod`).
- Pre-commit checks in CI: `terraform fmt -check`, `terraform validate`, `tflint`, `checkov`.

### 2.2 Directory layout

```
infra/terraform/
├── README.md                    # apply order, bootstrap runbook, troubleshooting
├── bootstrap/                   # LOCAL state, committed to git
│   ├── main.tf                  # state S3 bucket (SSE-KMS, versioned, block public),
│   │                            # DynamoDB lock table, GitHub OIDC provider,
│   │                            # IAM role for CI (assume-role from OIDC),
│   │                            # ECR repo `ohi-api` (image-scan-on-push, lifecycle keep-10)
│   ├── outputs.tf               # role ARN, ECR URI, bucket name, lock table name
│   ├── variables.tf             # GitHub org/repo, project prefix
│   ├── versions.tf              # required_version + required_providers
│   ├── terraform.tfvars         # project = "ohi", repo = "shiftbloom-studio/open-hallucination-index"
│   └── .gitignore               # *.tfstate*, .terraform/, .terraform.lock.hcl kept
├── _shared/                     # consumed by each layer as `module.shared`
│   ├── main.tf                  # empty; module is values-only
│   ├── outputs.tf               # exposes tags, name_prefix, project, region
│   ├── variables.tf             # layer (string), region (string, default eu-central-1)
│   └── versions.tf              # no backend (values-only module)
├── storage/                     # S3 artifacts bucket (NLI heads, calibration, eval snapshots)
│   ├── main.tf
│   ├── outputs.tf
│   ├── variables.tf
│   ├── versions.tf              # backend "s3" { key = "prod/storage/terraform.tfstate" }
│   └── terraform.tfvars
├── secrets/                     # Secrets Manager placeholder entries (values seeded out-of-band)
│   ├── main.tf                  # 7 secrets; see §4.1
│   ├── outputs.tf               # secret ARNs
│   ├── variables.tf
│   ├── versions.tf              # key = "prod/secrets/terraform.tfstate"
│   └── terraform.tfvars
├── compute/                     # Lambda container, Function URL, IAM, log group, alarms
│   ├── main.tf
│   ├── lambda.tf                # aws_lambda_function (Image package), Function URL
│   │                            # with auth_type=AWS_IAM, invoke_mode=RESPONSE_STREAM
│   │                            # (needed for /verify/stream SSE)
│   ├── iam.tf                   # execution role, Secrets-read policy, CloudWatch policy
│   ├── logs.tf                  # aws_cloudwatch_log_group, retention 14d
│   ├── outputs.tf               # function ARN, Function URL, alias ARN
│   ├── variables.tf             # image_tag (input from CI), memory_mb, timeout_s, env_vars
│   ├── versions.tf              # key = "prod/compute/terraform.tfstate"
│   └── terraform.tfvars         # defaults: memory_mb=2048, timeout_s=60
├── edge/                        # CloudFront, WAF, ACM (us-east-1)
│   ├── main.tf
│   ├── acm.tf                   # aws_acm_certificate (us-east-1) + aws_acm_certificate_validation
│   ├── waf.tf                   # aws_wafv2_web_acl (scope=CLOUDFRONT, in us-east-1) + rules
│   ├── cloudfront.tf            # distribution, origin = Lambda Function URL via OAC,
│   │                            # aliases = [hostname], WAF ACL attached
│   ├── oac.tf                   # aws_cloudfront_origin_access_control (signing_behavior=always,
│   │                            # origin_access_control_origin_type=lambda); plus
│   │                            # aws_lambda_permission targeting the Lambda ARN from
│   │                            # compute's remote state, granting
│   │                            # cloudfront.amazonaws.com:InvokeFunctionUrl with
│   │                            # source_arn = the distribution ARN (same layer, no cycle)
│   ├── outputs.tf               # distribution domain, ACM validation records, WAF ACL ARN
│   ├── variables.tf
│   ├── versions.tf              # key = "prod/edge/terraform.tfstate"; declares aws.use1 alias
│   └── terraform.tfvars         # hostname = "ohi.shiftbloom.studio"
└── observability/               # CloudWatch dashboard, metric filters, Budgets, SNS
    ├── main.tf
    ├── budgets.tf               # two budgets: €100 forecast alarm, €140 actual alarm
    ├── dashboard.tf             # aws_cloudwatch_dashboard (Lambda + CF + WAF + Gemini-cost gauge)
    ├── metric_filters.tf        # 5 filters on Lambda log group (§5.2)
    ├── sns.tf                   # SNS topic `ohi-alerts`, email subscription
    ├── outputs.tf
    ├── variables.tf
    ├── versions.tf              # key = "prod/observability/terraform.tfstate"
    └── terraform.tfvars         # alert_email = "fabian@shiftbloom.studio"
```

### 2.3 Apply order and cross-layer wiring

```
bootstrap
  └─► storage ─┐
       secrets ─┼─► compute ─► edge ─► observability
                │      ▲                      │
                └──────┘                      │
                                              │
              (observability reads compute + edge remote state
               for resource ARNs and function names)
```

Cross-layer outputs flow via `data "terraform_remote_state"` reads keyed to the deterministic S3 path `prod/<layer>/terraform.tfstate`. Each consuming layer declares a `data.terraform_remote_state.<layer>` block at the top of its `main.tf`.

Each layer also includes the shared module:
```hcl
module "shared" {
  source = "../_shared"
  layer  = "compute"   # or whichever
}
locals {
  tags = module.shared.tags
}
```
This keeps tag/name conventions DRY without resorting to symlinks (Windows-friendly).

### 2.4 State backend convention

- Bucket: `ohi-tfstate-<aws-account-id>` (SSE with a bootstrap-created KMS key; versioning on; public access blocked; bucket policy restricts access to the `ohi-terraform-apply` role).
- Key template: `prod/<layer>/terraform.tfstate`.
- Lock table: `ohi-tfstate-lock` (single table for all layers).

### 2.5 Tagging

All resources get default tags via `provider "aws" { default_tags { ... } }`:

| Tag | Value | Purpose |
|---|---|---|
| `Project` | `ohi` | Cost Explorer filter |
| `Environment` | `prod` | Future-proofing (not used today) |
| `Layer` | `bootstrap` / `storage` / `secrets` / `compute` / `edge` / `observability` | Per-layer cost attribution |
| `ManagedBy` | `terraform` | Distinguish from hand-created |
| `CostCenter` | `ohi` | Budget enforcement |

### 2.6 Naming

- Prefix all named resources with `ohi-` (e.g., `ohi-api`, `ohi-alerts`, `ohi-verify-rate-limit`).
- No environment suffix (single env).
- Where AWS requires globally-unique names (S3 buckets), append `<aws-account-id>`.

---

## 3. CI/CD (brainstorm Section 3)

### 3.1 Posture

- **GitHub Actions** as the single runner. No self-hosted runners.
- **OIDC federation** (`token.actions.githubusercontent.com`) → `ohi-terraform-apply` IAM role. No long-lived AWS keys anywhere.
- **Plan on PR**, **apply on manual dispatch** (not on merge). Reason: the user is sole maintainer; PR review + intentional dispatch is a cheaper, safer gate than a protected environment with self-approval.
- **GitHub Environment `prod`** exists, but its only purpose is to scope the OIDC role-assumption (workflow step `environment: prod`). It does NOT require a reviewer (self-review is theater).
- Concurrency group `terraform-apply` ensures only one apply at a time across all layers.

### 3.2 Workflows

Four workflow files under `.github/workflows/`:

#### 3.2.1 `infra-plan.yml` — PR gate
- **Trigger:** `pull_request` paths `infra/terraform/**`
- **Job matrix:** one job per layer (`storage`, `secrets`, `compute`, `edge`, `observability`). `bootstrap` is excluded from CI — apply it locally only.
- **Steps:**
  1. Checkout
  2. Configure AWS creds via OIDC (role `ohi-terraform-apply`, session name `gha-plan-<layer>-<pr-number>`)
  3. `terraform init` (reads backend.tf with remote state in S3)
  4. `terraform fmt -check -recursive`
  5. `terraform validate`
  6. `tflint --minimum-failure-severity=warning`
  7. `checkov -d . --framework terraform --quiet` (soft fail, warnings annotated on PR)
  8. `terraform plan -no-color -out=tfplan`
  9. `terraform show -no-color tfplan | head -c 60000 > plan.txt`
  10. Post plan as sticky PR comment via `marocchino/sticky-pull-request-comment`
- **Required to merge:** all plan jobs green. Checkov warnings do not block.

#### 3.2.2 `infra-apply.yml` — Manual dispatch
- **Trigger:** `workflow_dispatch` with inputs:
  - `layer`: enum (`storage` | `secrets` | `compute` | `edge` | `observability`)
  - `confirm`: text input — must equal `apply` to proceed (drift-protection against mis-click)
- **Job:**
  1. Guard: fail if `inputs.confirm != "apply"`
  2. Checkout, OIDC, `terraform init`, `terraform plan -out=tfplan`
  3. Echo full plan to job log
  4. `terraform apply tfplan`
  5. Post outputs to Job Summary

#### 3.2.3 `image-build.yml` — Lambda container image
- **Trigger:** `push` to `main`, paths `src/api/**`, `docker/api/**`, `pyproject.toml`, `pip.lock`
- **Steps:**
  1. Checkout
  2. Configure AWS creds via OIDC (role `ohi-terraform-apply`, session `gha-image-build`)
  3. `docker buildx build` the Lambda image from `docker/lambda/Dockerfile` (new file; §3.3)
  4. Tag with `$GITHUB_SHA` (immutable) AND `prod` (moving)
  5. Push both tags to ECR `ohi-api`
  6. Write `$GITHUB_SHA` to a GitHub Actions artifact `image-sha.txt`
- **No Terraform apply**. A human dispatches `infra-apply.yml` with `layer=compute` and sets the `image_tag` tfvar via `terraform.tfvars` or `-var` (see §3.4).

#### 3.2.4 `bootstrap-permissions.yml` — (optional, post-I.0) drift check
- **Trigger:** `schedule` nightly + `workflow_dispatch`
- **Steps:** `terraform plan` for bootstrap — alerts on drift. Does not apply.
- Exists because bootstrap applies locally; we want a sanity check that the CI role hasn't been tampered with.

### 3.3 Lambda container image

**New file:** `docker/lambda/Dockerfile`. Multi-stage.

```dockerfile
# Stage 1: dependency install
FROM public.ecr.aws/docker/library/python:3.12-slim AS builder
WORKDIR /build
COPY src/api/pyproject.toml src/api/pip.lock ./
RUN pip install --no-cache-dir --target /deps -r pip.lock

# Stage 2: runtime
FROM public.ecr.aws/lambda/python:3.12
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=builder /deps ${LAMBDA_TASK_ROOT}
COPY src/api/ ${LAMBDA_TASK_ROOT}/api/
ENV AWS_LAMBDA_EXEC_WRAPPER=/opt/extensions/lambda-adapter
ENV PORT=8080
ENV READINESS_CHECK_PATH=/health/ready
CMD ["uvicorn", "api.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Notes:
- `aws-lambda-adapter` converts Lambda events → HTTP 8080 → FastAPI with response streaming support (needed for `/verify/stream` SSE).
- Lambda Function URL `invoke_mode = "RESPONSE_STREAM"` is required for SSE; buffered mode caps responses at 6MB and loses streaming semantics.
- `AWS_LAMBDA_EXEC_WRAPPER` path is `/opt/extensions/lambda-adapter` for the `0.8.4` release; the implementation plan must verify this against the adapter's published README at apply time since earlier releases used `/opt/bootstrap`.
- Image size budget: < 500 MB compressed. Gemini SDK, httpx, structlog, numpy, scikit-learn (for conformal), Cypher/Qdrant/Redis clients. Within limits.

### 3.4 Image-tag handoff (CI → Terraform)

**Digest pinning is mandatory** — Terraform can't detect that a tag now points at a new image, so we must resolve to the digest at plan time:

```hcl
# compute/variables.tf
variable "image_tag" {
  type    = string
  default = "prod"
}

# compute/lambda.tf
data "aws_ecr_repository" "api" {
  name = "ohi-api"
}

data "aws_ecr_image" "api" {
  repository_name = data.aws_ecr_repository.api.name
  image_tag       = var.image_tag
}

resource "aws_lambda_function" "api" {
  function_name = "ohi-api"
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.api.repository_url}@${data.aws_ecr_image.api.image_digest}"
  # ...
}
```

`repository_url` is `<account>.dkr.ecr.<region>.amazonaws.com/ohi-api`, and `image_digest` is `sha256:…`. Concatenating with `@` produces the correct immutable digest URI that Lambda requires.

`data.aws_ecr_image` is re-read on every plan, so pushing a new image under the `prod` tag changes the digest, which shows as a delta in `terraform plan` and triggers a redeploy on apply. For a pinned deploy, dispatch `infra-apply.yml` with `image_tag=<git-sha>` as an optional input.

### 3.5 Bootstrap is local-only

`bootstrap/` is applied from Fabian's laptop once (Phase I.0). Rationale:
- It creates the state bucket + OIDC provider + the CI role itself — bootstrapping CI with CI is circular.
- Destroy is rare and intentional.
- The lockfile `.terraform.lock.hcl` IS committed so provider versions are reproducible.

Runbook steps captured in `infra/terraform/bootstrap/README.md` (written as part of the implementation plan).

---

## 4. Secrets + env config (brainstorm Section 4)

### 4.1 Secrets Manager entries (created empty by Terraform, seeded out-of-band)

| Secret name | Purpose | Shape | Consumer |
|---|---|---|---|
| `ohi/gemini-api-key` | Google Generative AI key | string | Lambda |
| `ohi/internal-bearer-token` | Long random for MCP/CI/benchmark trusted callers | string (≥64 chars) | Lambda (auth) |
| `ohi/cloudflared-tunnel-token` | Tunnel credential for PC cloudflared container | string | PC (not read by Lambda) |
| `ohi/cf-access-service-token` | Cloudflare Access service-token pair for Lambda→tunnel auth | JSON `{client_id, client_secret}` | Lambda |
| `ohi/labeler-tokens` | Expert + adjudicator labeler tokens (algorithm §11) | JSON `{expert: [...], adjudicator: [...]}` | Lambda |
| `ohi/pc-origin-credentials` | Basic-auth creds for PostgREST and WebDIS proxies on PC (defense-in-depth behind CF Access) | JSON `{pg_rest_user, pg_rest_pass, webdis_user, webdis_pass}` | Lambda + PC |
| `ohi/neo4j-credentials` | Neo4j auth | JSON `{user, password}` | Lambda + PC |

### 4.2 Terraform shape

Each secret is an `aws_secretsmanager_secret` with:
- `kms_key_id` = bootstrap-created KMS key `ohi-secrets`
- `recovery_window_in_days = 0` (force immediate delete on destroy — this is single-env)
- Tag `SecretRole=<purpose>`

**Runbook callout:** `recovery_window_in_days = 0` means an accidental `terraform destroy` on the `secrets/` layer permanently loses the rotated values with no AWS-side recovery. Acceptable given single-env and the fact that Gemini/tunnel keys can be re-issued from source — but the rotation runbook explicitly warns about this and instructs operators to back up Gemini keys to a password manager before rotation.

NO `aws_secretsmanager_secret_version` in Terraform. The initial value is seeded via AWS CLI after apply:

```bash
aws secretsmanager put-secret-value \
  --secret-id ohi/gemini-api-key \
  --secret-string "$(cat ~/.ohi/gemini-api-key.txt)"
```

Rationale: keeping secret values out of Terraform state means state files don't need TS-level protection beyond the bucket ACL, and rotation doesn't cause drift.

### 4.3 Lambda reads at cold start

Application reads secrets via `boto3.client("secretsmanager").get_secret_value(...)` in a singleton `SecretsLoader` class, with:
- **Lazy fetch** on first call per secret (not all at cold start — some endpoints don't need Gemini key)
- **In-memory TTL cache**, TTL 10 min. Manual rotation via CLI takes effect within 10 min without redeploy.
- **Decryption errors** → raise `ConfigurationError` at import time; Lambda cold-start fails loudly rather than returning subtly-broken responses.

IAM policy attached to Lambda execution role:
```hcl
statement {
  actions   = ["secretsmanager:GetSecretValue"]
  resources = [for s in aws_secretsmanager_secret.ohi : s.arn]
}
statement {
  actions   = ["kms:Decrypt"]
  resources = [aws_kms_key.ohi_secrets.arn]
  condition {
    test     = "StringEquals"
    variable = "kms:ViaService"
    values   = ["secretsmanager.eu-central-1.amazonaws.com"]
  }
}
```

### 4.4 Environment variables (Lambda)

Only non-secret config goes in `aws_lambda_function.environment.variables`. Secrets are fetched at runtime (§4.3).

| Var | Example | Source |
|---|---|---|
| `OHI_ENV` | `prod` | tfvars |
| `OHI_REGION` | `eu-central-1` | tfvars |
| `OHI_LOG_LEVEL` | `INFO` | tfvars |
| `OHI_GEMINI_MODEL` | `gemini-3-flash-preview` | tfvars (tunable) |
| `OHI_GEMINI_DAILY_CEILING_EUR` | `1.00` | tfvars |
| `OHI_CF_TUNNEL_HOSTNAME_NEO4J` | `neo4j.tunnel.shiftbloom.studio` | tfvars |
| `OHI_CF_TUNNEL_HOSTNAME_QDRANT` | `qdrant.tunnel.shiftbloom.studio` | tfvars |
| `OHI_CF_TUNNEL_HOSTNAME_PG_REST` | `pg.tunnel.shiftbloom.studio` | tfvars |
| `OHI_CF_TUNNEL_HOSTNAME_WEBDIS` | `redis.tunnel.shiftbloom.studio` | tfvars |
| `OHI_S3_ARTIFACTS_BUCKET` | `ohi-artifacts-<acct>` | `data.terraform_remote_state.storage` |
| `OHI_INTERNAL_BEARER_SECRET_ARN` | full ARN | `aws_secretsmanager_secret.internal_bearer.arn` |
| `OHI_GEMINI_KEY_SECRET_ARN` | full ARN | (same) |
| _(other secret ARNs)_ | | |

No secret values in env vars — only ARNs.

### 4.5 Rotation posture

- No automated rotation in Phase 1.
- Manual rotation playbook in `docs/runbooks/rotate-secret.md`: `aws secretsmanager put-secret-value`, wait 10 min, verify via `/health/deep`.
- Gemini key rotation: Google console → regenerate → put-secret-value → verify.
- Internal bearer: generate with `openssl rand -base64 64` → put-secret-value → update consumer (MCP server, CI).

---

## 5. Observability + cost controls (brainstorm Section 5)

### 5.1 Log pipeline

- Lambda writes **structured JSON** via `structlog` to stdout → CloudWatch Logs group `/aws/lambda/ohi-api`, retention **14 days**.
- Log lines carry: `ts`, `level`, `msg`, `request_id`, `route`, `ip_hash`, `pipeline_stage`, `duration_ms`, `status_code`, plus pipeline-specific fields.
- CloudFront access logs → S3 bucket `ohi-cf-access-logs-<acct>`, lifecycle expire at 30 days. NOT CloudWatch-shipped (saves cost).

### 5.2 Metric filters

Five CloudWatch Logs metric filters under namespace `OHI/App`, all counting `1` per match:

| Filter | Pattern | Metric | Purpose |
|---|---|---|---|
| pipeline-error | `{ $.level = "ERROR" && $.pipeline_stage = * }` | `PipelineError` | Any L1–L7 error |
| rate-limit-app | `{ $.msg = "rate_limit_triggered" }` | `RateLimitApp` | L1/L2 app-layer rate-limit hits |
| pc-origin-timeout | `{ $.msg = "pc_origin_timeout" }` | `PCOriginTimeout` | Cloudflare tunnel origin unreachable |
| gemini-ceiling-hit | `{ $.msg = "gemini_cost_ceiling_hit" }` | `GeminiCeilingHit` | Daily Gemini €€ ceiling exceeded |
| cold-start | `{ $.msg = "lambda_cold_start" }` | `LambdaColdStart` | Cold-start diagnostic |

### 5.3 Alarms

| Alarm | Metric | Threshold | Action |
|---|---|---|---|
| Budget-forecast-100€ | AWS/Billing (Budget) | forecast ≥ €100 | SNS → email |
| Budget-actual-140€ | AWS/Billing (Budget) | actual ≥ €140 | SNS → email |
| Lambda-5xx-rate | AWS/Lambda `Errors / Invocations` | > 0.10 for 3×5-min | SNS → email |
| PC-origin-timeout-rate | `OHI/App/PCOriginTimeout` | > 5 in 15 min | SNS → email |
| Gemini-ceiling-daily | `OHI/App/GeminiCeilingHit` | ≥ 1 in a day | SNS → email (diagnostic only) |
| WAF-block-spike | AWS/WAFV2 `BlockedRequests` | > 500 in 5 min | SNS → email |

SNS topic `ohi-alerts` has a single email subscription `fabian@shiftbloom.studio`. Second subscription (Slack webhook or mobile push) deferred.

### 5.4 Budgets

Two AWS Budgets in `observability/budgets.tf`:

1. **Forecast alarm** — €100/month, forecast exceed → SNS (early warning).
2. **Actual alarm** — €140/month, actual exceed → SNS (hard-stop warning).

No Budget **Action** (e.g., stop Lambda) — too drastic for Phase 1, and Lambda doesn't integrate cleanly with Budget Actions anyway.

### 5.5 Gemini daily cost ceiling (algorithm spec §11 L3)

- Lambda maintains a per-day counter in **PC Redis** via WebDIS: `gemini_cost_eur:{YYYY-MM-DD}`.
- Before each Gemini call, Lambda reads counter + projected cost. If `counter + projected > OHI_GEMINI_DAILY_CEILING_EUR`: emit `gemini_cost_ceiling_hit` log line → return 503.
- After each Gemini call, Lambda increments counter by actual cost (from Gemini response metadata).
- If Redis unreachable (tunnel down), fall back to **in-Lambda ephemeral counter** (accepts overrun risk during outages; logged as `gemini_counter_fallback`).

### 5.6 CloudWatch dashboard

Single dashboard `ohi-prod`:
- Row 1: Lambda invocations, errors, throttles, duration p50/p95/p99 (Frankfurt).
- Row 2: CloudFront requests, 4xx%, 5xx%, cache-hit ratio (global).
- Row 3: WAF allowed vs blocked; top 10 blocked rule IDs.
- Row 4: Metric filters (PipelineError, RateLimitApp, PCOriginTimeout, GeminiCeilingHit) sparklines.
- Row 5: AWS Cost per day (text widget with `cur:` query or a manual update). Budgets widget.

### 5.7 Cost tracking convention

- Every resource tagged `CostCenter=ohi`.
- Monthly review: Cost Explorer filter `CostCenter=ohi`, group by `Layer`. Target < €50/mo steady state.

---

## 6. PC-side container stack

### 6.1 New file: `docker/compose/pc-data.yml`

Standalone profile — does NOT modify the existing `docker-compose.yml` (V1 stack stays functional). User runs:

```bash
docker compose -f docker/compose/pc-data.yml --env-file docker/compose/.env.pc-data up -d
```

Services:

```yaml
services:
  neo4j:                          # algorithm graph
    image: neo4j:5-community
    ports: ["7474:7474", "7687:7687"]
    volumes: ["neo4j-data:/data"]
    env_file: .env.pc-data

  qdrant:                         # algorithm vectors
    image: qdrant/qdrant:v1.12.0
    ports: ["6333:6333"]
    volumes: ["qdrant-data:/qdrant/storage"]

  postgres:                       # algorithm §12 tables
    image: postgres:16-alpine
    ports: ["5432:5432"]
    volumes: ["pg-data:/var/lib/postgresql/data"]
    env_file: .env.pc-data

  postgrest:                      # HTTP proxy for Postgres
    image: postgrest/postgrest:v12.2.0
    env_file: .env.pc-data
    depends_on: [postgres]

  redis:                          # cache + rate-limit counters + Gemini cost counter
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: ["redis-data:/data"]

  webdis:                         # HTTP proxy for Redis
    image: nicolas/webdis:0.1.23   # pinned; exposes port 7379 internally
    env_file: .env.pc-data
    depends_on: [redis]

  cloudflared:                    # outbound tunnel to CF edge
    image: cloudflare/cloudflared:2026.3.0  # pinned to a known-good release
    command: tunnel --no-autoupdate run
    env_file: .env.pc-data        # TUNNEL_TOKEN=...

volumes:
  neo4j-data:
  qdrant-data:
  pg-data:
  redis-data:
```

### 6.2 Why two HTTP proxies on PC

Cloudflare Tunnel exposes HTTP/HTTPS hostnames to Lambda, but Lambda cannot run `cloudflared access tcp` client. So raw-TCP services (Postgres, Redis) are exposed as HTTP:
- **PostgREST** → Postgres. Lambda builds requests like `GET /feedback_pending?claim_id=eq.X` or `POST /rpc/<fn>`. PostgREST handles row-level security + auth.
- **WebDIS** → Redis. Lambda calls `GET /SET/key/value` and `GET /INCR/key`. Simple subset of Redis operations Lambda actually needs.

Neo4j (HTTP Cypher at :7474) and Qdrant (HTTP at :6333) are natively HTTP — no proxy needed.

### 6.3 `.env.pc-data` template (new file, gitignored)

User fills values from Secrets Manager seeds:

```dotenv
# Cloudflare Tunnel
TUNNEL_TOKEN=eyJ...

# Neo4j
NEO4J_AUTH=neo4j/<strong-password>

# Postgres
POSTGRES_USER=ohi_app
POSTGRES_PASSWORD=<strong-password>
POSTGRES_DB=ohi_prod

# PostgREST
PGRST_DB_URI=postgres://ohi_app:<strong-password>@postgres:5432/ohi_prod
PGRST_DB_SCHEMA=public
PGRST_JWT_SECRET=<long-random>

# WebDIS basic-auth (defense-in-depth)
WEBDIS_HTTP_AUTH=<user:password>
```

### 6.4 Initial schema seed

A one-shot Phase I.3 step runs the Postgres schema DDL from the algorithm repo (`scripts/db/init.sql`) against the PC postgres container:
```bash
docker compose -f docker/compose/pc-data.yml exec postgres \
  psql -U ohi_app -d ohi_prod -f /docker-entrypoint-initdb.d/init.sql
```

---

## 7. DNS + Cloudflare Tunnel wiring

### 7.1 Split-brain DNS

| Zone | Managed by | Records needed |
|---|---|---|
| `shiftbloom.studio` | Your current DNS provider | `CNAME ohi → <cloudfront-domain>.cloudfront.net`, ACM validation CNAME (from Terraform output) |
| `tunnel.shiftbloom.studio` | **Delegated to Cloudflare** via `NS tunnel → cloudflare NS servers` record at current DNS provider | Auto-created by Cloudflare Tunnel for each hostname |

### 7.2 Records the user adds (step-by-step in the runbook)

**Phase I.0 (after bootstrap apply):** none.

**Phase I.1 (after edge-layer first plan):** Terraform output emits the ACM validation CNAME name + value. User adds:
```
CNAME _<random>.ohi.shiftbloom.studio → <random>.acm-validations.aws.
```
Then re-runs the edge-layer apply — `aws_acm_certificate_validation` completes, CloudFront dist is created.

**Phase I.1 (after edge-layer first apply):** user adds:
```
CNAME ohi.shiftbloom.studio → <cloudfront-domain>.cloudfront.net.
```

**Phase I.3 (tunnel):** user creates tunnel at `one.dash.cloudflare.com → Zero Trust → Networks → Tunnels → Create tunnel`, picks a name `ohi-pc`, copies the tunnel token to Secrets Manager (`ohi/cloudflared-tunnel-token`), and in tunnel → Public Hostnames adds four routes:

| Hostname | Service |
|---|---|
| `neo4j.tunnel.shiftbloom.studio` | `http://neo4j:7474` |
| `qdrant.tunnel.shiftbloom.studio` | `http://qdrant:6333` |
| `pg.tunnel.shiftbloom.studio` | `http://postgrest:3000` |
| `redis.tunnel.shiftbloom.studio` | `http://webdis:7379` |

### 7.3 Cloudflare Access (zero-trust auth)

For each of the four tunnel hostnames above, a **Cloudflare Access Application** is configured with **Service Auth** policy:
- Allow = `Service Auth` with token issued to Lambda (`ohi-lambda-service-token`).
- Session duration: 24h.
- Lambda presents `CF-Access-Client-Id` + `CF-Access-Client-Secret` on every tunnel request (read from `ohi/cf-access-service-token` secret).

Cloudflare Access configuration is **not Terraformed** in Phase 1 (cloudflare provider adds cognitive load; the UI is fine for 4 one-time rules). Captured in runbook `docs/runbooks/cloudflare-access-setup.md`. Flagged as a Phase 2 Terraform-ification candidate.

### 7.4 Zone delegation gotcha

Delegating `tunnel.shiftbloom.studio` to Cloudflare requires adding NS records at your current DNS provider:
```
NS tunnel.shiftbloom.studio → <your-cloudflare-assigned-ns>.ns.cloudflare.com
```
(Cloudflare shows the NS pair when you add `tunnel.shiftbloom.studio` as a subdomain zone.) This is a one-time manual step in Phase I.0.

---

## 8. Phased rollout

Each phase has explicit entry and exit criteria. Phases are ordered — do not skip.

### 8.1 Phase I.0 — Bootstrap (local)

**Goal:** establish AWS account, state bucket, OIDC, ECR, DNS prerequisites.
**Where applied:** Fabian's laptop, `infra/terraform/bootstrap/`.

**Steps:**
1. AWS account ready, CLI profile `ohi-admin` configured with admin creds (temporary).
2. Delegate `tunnel.shiftbloom.studio` NS to Cloudflare at current DNS provider.
3. `cd infra/terraform/bootstrap && terraform init && terraform apply`.
4. Commit updated `terraform.lock.hcl` (but NOT `terraform.tfstate`).
5. Verify outputs: state bucket, DynamoDB table, ECR repo `ohi-api`, OIDC provider, CI role ARN.

**Exit gate:**
- `aws s3 ls s3://ohi-tfstate-<acct>` works.
- `aws iam get-role --role-name ohi-terraform-apply` returns the role.
- GitHub secrets `AWS_ROLE_ARN` + `AWS_REGION` set.
- Admin CLI profile can be revoked; OIDC handles everything from now.

### 8.2 Phase I.1 — Edge + storage + secrets (stub Lambda)

**Goal:** public HTTPS endpoint returning 503 "not-ready" from a stub Lambda, with full WAF/CloudFront in place. No real app logic yet.

**Steps:**
1. Build a **stub Lambda image** (`docker/lambda/Dockerfile.stub`) that returns `{"status":"bootstrapping"}` with HTTP 503. Push to ECR as tag `stub`.
2. Dispatch `infra-apply.yml` with `layer=storage`.
3. Dispatch `infra-apply.yml` with `layer=secrets`.
4. Dispatch `infra-apply.yml` with `layer=compute` — `image_tag=stub`.
5. Seed secrets out-of-band via AWS CLI (Gemini key, internal bearer, etc.) — tunnel + CF Access secrets can be placeholder strings for now.
6. Dispatch `infra-apply.yml` with `layer=edge` — first apply fails at `acm_certificate_validation` because CNAME isn't in DNS yet.
7. Fetch ACM validation CNAME from TF output, add to DNS provider.
8. Re-dispatch `infra-apply.yml` with `layer=edge` — ACM completes, CloudFront distribution deploys.
9. Add `CNAME ohi.shiftbloom.studio → <dist>.cloudfront.net` at DNS provider.
10. Dispatch `infra-apply.yml` with `layer=observability`.

**Exit gate:**
- `curl -v https://ohi.shiftbloom.studio/health/live` returns 503 `{"status":"bootstrapping"}` (expected — stub).
- WAF blocks on 10 requests/sec from a single IP confirmed.
- SNS alarm email received for a forced budget-forecast test.

### 8.3 Phase I.2 — Real Lambda image + calibration seed

**Goal:** replace stub Lambda with real OHI API; seed initial calibration set.

**Steps:**
1. Merge algorithm Phase 1 tasks (1.9 SSE, 1.10 rate-limit, 1.14 acceptance bench) to `main`.
2. `image-build.yml` fires automatically → pushes `ohi-api:<sha>` + `ohi-api:prod` to ECR.
3. Dispatch `infra-apply.yml` with `layer=compute` — Lambda picks up `ohi-api:prod`.
4. Run `scripts/calibration/synthesize_phase2_calibration.py` **from Fabian's laptop** against HuggingFace dev splits → uploads `calibration_set.json` to `s3://ohi-artifacts-<acct>/calibration/` (requires seeded Gemini key + Lambda URL for pipeline runs).
5. Verify `/health/deep` returns green for all providers.

**Exit gate:**
- `curl -X POST https://ohi.shiftbloom.studio/api/v2/verify -d '{"text":"The Eiffel Tower is in Berlin."}'` returns a verdict (even if tunnel-failure 503 — tunnel isn't up yet in I.2).
- Cold-start duration logged; < 4s p99.

### 8.4 Phase I.3 — PC tunnel up (end-to-end)

**Goal:** Lambda reaches PC data stores through Cloudflare Tunnel; full `/verify` pipeline runs.

**Steps:**
1. Provision PC Docker Compose (`docker/compose/pc-data.yml`, `.env.pc-data`).
2. Create Cloudflare Tunnel `ohi-pc` in Cloudflare dashboard, copy tunnel token to `TUNNEL_TOKEN` env.
3. Add 4 public hostnames (neo4j, qdrant, pg, redis) pointing to container DNS names.
4. Configure Cloudflare Access with Service Auth for each hostname, issue token, copy to `ohi/cf-access-service-token` secret.
5. `docker compose -f docker/compose/pc-data.yml up -d`.
6. Run Postgres init schema.
7. Run ingestion (algorithm repo — `gui_ingestion_app`) to populate Neo4j + Qdrant with seed data.
8. `curl -X POST https://ohi.shiftbloom.studio/api/v2/verify -d '<claim>'` → full pipeline end-to-end.

**Exit gate:**
- Full verify response with evidence + calibrated verdict.
- PC-off test: stop `cloudflared` on PC → API returns 503 `{"status":"resting"}` within 5s.

### 8.5 Phase I.4 — Budget + observability hardening

**Goal:** confirm alarms fire, rotate admin creds, cut over.

**Steps:**
1. Run a **synthetic canary** hourly via `aws events` scheduled rule that hits `/health/deep` — add `canary.tf` to `observability/`. Optional; deferred if cost is a concern (Lambda invocation + data transfer is negligible, probably worth it).
2. Verify all alarms by deliberately tripping each (temp WAF rule, force a pipeline error, etc.).
3. Revoke local admin IAM user; keep only OIDC role.
4. Tag release: `git tag ohi-v2-prod-launch && git push --tags` (user-initiated).

**Exit gate:**
- All alarms test-fired and SNS email received.
- Cost Explorer shows a day of traffic; cost within forecast.
- `docs/runbooks/*.md` complete (rotate-secret, scale-up, incident-response-basic).

---

## 9. Risks + deferred items

### 9.1 Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Residential ISP flakiness** on PC side | High | Medium | Lambda 503 w/ Retry-After; CloudFront custom error page; public messaging that service is part-time |
| R2 | **Cloudflare Tunnel token exfiltration** | Low | High | Token in Secrets Manager only; tunnel routes protected by Access service tokens (double gate); rotation runbook documented |
| R3 | **Lambda cold-start** > 5s causing timeouts | Medium | Low | Image kept < 500MB; boto3 lazy import; only Gemini + secrets fetched lazily; no VPC (no ENI attach penalty) |
| R4 | **Gemini API deprecation** / preview model EOL | Medium | Medium | `OHI_GEMINI_MODEL` env var allows instant switch; algorithm spec already reserves Claude/OpenAI adapter slots |
| R5 | **CloudFront Function URL streaming limits** (response streaming on Function URLs has subtleties) | Medium | Medium | Explicit test in Phase I.2 on `/verify/stream`; fallback is polling `/verdict/{id}` (algorithm supports this) |
| R6 | **WAF rate-limit tuning mis-calibration** (too tight → legitimate users 429; too loose → abuse bill) | Medium | Low | Start with conservative defaults (global 2000/5min, per-IP 300/5min); monitor WAF blocks dashboard weekly; algorithm also enforces L1 app-layer rate limits |
| R7 | **PC power cut during write** → Postgres/Neo4j corruption | Low | Medium | Out of scope to prevent; documented that periodic PC-local backups are user responsibility (Phase 2 could add S3 backup script) |
| R8 | **ACM cert auto-renewal fails** (DNS validation record deleted or moved) | Low | High | Alarm on `ACM certificate expiring` (CloudWatch AWS/CertificateManager, DaysToExpiry < 30); runbook documented |
| R9 | **Budget alarm triggers at midnight on the 1st** (forecast resets, lag in AWS metrics) | Low | Low | Forecast alarm is informational; actual alarm at €140 is the real gate |

### 9.2 Explicitly deferred

- **Multi-AZ / multi-region / backups / failover** — single PC is a single point of failure, accepted.
- **Dev + staging environments** — `prod` only. Feature work validated locally per `AGENTS.md` workflow 1.
- **Automated secret rotation** — manual playbook only.
- **GDPR subject-access tooling** — privacy posture documented in algorithm §11 (raw text not persisted), no /right-to-erasure API.
- **Horizontal scaling / provisioned concurrency** — Lambda on-demand, cold-start accepted.
- **Local-LLM hosting** — remote Gemini only in Phase 1.
- **Cloudflare-provider-based Access rule management** — UI only for 4 rules.
- **Canary/synthetic monitoring as a separate tool (Pingdom, UptimeRobot)** — AWS EventBridge + Lambda self-canary covers it.
- **Terraform-managed Google Cloud Gemini project** — keys issued manually in GCP console.
- **PC backups to S3** — deferred; user takes responsibility for local backups of `neo4j-data`, `pg-data`, `qdrant-data`, `redis-data` volumes.

---

## 10. Cost estimate (from Section 1, re-confirmed)

| Line item | Monthly |
|---|---|
| CloudFront (50GB transfer, 500k requests) | €5–10 |
| AWS WAF (Core Rule Set + 2 custom rules, 500k requests) | €3–6 |
| Lambda (~100k invocations, 2GB, avg 3s) | €1–3 |
| S3 (artifacts + CF logs, < 5GB) | €0.50–1.50 |
| Secrets Manager (7 secrets) | €3 (€0.40 × 7) |
| CloudWatch (logs + 5 metric filters + 1 dashboard) | €1–3 |
| KMS (1 key, < 20k requests) | €1 |
| DynamoDB (state lock, negligible) | ~€0.10 |
| **AWS subtotal** | **~€15–28** |
| Gemini API (budget-capped) | €0–30 |
| Cloudflare Tunnel + Access (free tier covers this) | €0 |
| Domain (shiftbloom.studio already owned) | €0 |
| **Total** | **~€15–58** |

Comfortably under the €150 cap.

---

## 11. Decisions defaulted (flag anything you want to change)

Per your "best decisions on the way" directive, I took these calls without asking. Any of these can be changed in spec review.

1. **Email for alarms:** `fabian@shiftbloom.studio` (from session context).
2. **Gemini daily ceiling default:** €1/day (~€30/month envelope).
3. **Log retention:** 14d Lambda, 30d CloudFront S3.
4. **Image tagging:** `<git-sha>` immutable + `prod` moving.
5. **Image build trigger:** push to main when `src/api/**`, `docker/api/**`, `pyproject.toml`, `pip.lock` change.
6. **CI apply model:** manual `workflow_dispatch`, no auto-apply on merge.
7. **GitHub environment `prod`:** exists for OIDC scoping; no required reviewer (self-review is theater).
8. **Internal tunnel domain:** `tunnel.shiftbloom.studio` delegated to Cloudflare via NS record (cleanly separates public serving zone from internal Access-protected zone).
9. **PC HTTP proxies:** PostgREST (for Postgres) + WebDIS (for Redis). Neo4j and Qdrant exposed directly on their native HTTP.
10. **Cloudflare Access policies configured in UI,** not via `cloudflare/cloudflare` TF provider (4 rules once doesn't justify the provider).
11. **ECR lifecycle:** keep last 10 images, expire older.
12. **PC compose file:** new standalone `docker/compose/pc-data.yml`, does not modify the existing V1 `docker-compose.yml`.
13. **Calibration seeding:** run from Fabian's laptop, not from Lambda or CodeBuild. One-shot, needs Gemini + deployed API.
14. **KMS:** one CMK `ohi-secrets` created in bootstrap, used for both state bucket and Secrets Manager.
15. **WAF rate limits:** WAF global 2000/5min, per-IP 300/5min at edge; fine-grained per-route limits enforced at L1/L2 in the Lambda app (algorithm spec §11).

---

## 12. Out-of-scope explicitly

- Frontend rewrite (Next.js) — separate sub-project 3, not started.
- Algorithm beyond Phase 1 (Phase 2/3/4 features like multi-provider routing, adjudicator sprint tooling).
- Anything touching the existing V1 stack (`src/frontend/`, `src/ohi-mcp-server/`, the existing `docker-compose.yml`) — V2 infra lives alongside V1 until frontend rewrite replaces V1.

---

## Appendix A — What the implementation plan must produce

(Written here to constrain plan scope; the plan itself goes in `docs/superpowers/plans/2026-04-17-ohi-v2-infrastructure-implementation.md`.)

The implementation plan must yield, in order:

1. **`infra/terraform/bootstrap/`** — fully written, applyable locally, with README.md runbook.
2. **`infra/terraform/_shared/`** — tags, naming locals, shared vars.
3. **`infra/terraform/storage/`, `secrets/`, `compute/`, `edge/`, `observability/`** — each applyable via `infra-apply.yml`.
4. **`docker/lambda/Dockerfile`** + `docker/lambda/Dockerfile.stub`.
5. **`docker/compose/pc-data.yml`** + `.env.pc-data.example`.
6. **Four `.github/workflows/` files** — `infra-plan.yml`, `infra-apply.yml`, `image-build.yml`, `bootstrap-permissions.yml`.
7. **Runbooks** — `docs/runbooks/rotate-secret.md`, `cloudflare-access-setup.md`, `pc-compose-start.md`, `incident-response-basic.md`.
8. **Validation scripts** — Phase-gate assertion scripts that the plan's tasks can `./validate-phase-i0.sh` etc. to verify their own exit criteria without human judgment.

Every task in the plan must have:
- Explicit inputs (what files/outputs are required from prior tasks)
- Explicit outputs (what it writes)
- An automated validation step (the task is NOT done until the validation passes)
- No ambiguous "configure X" steps — every step has concrete commands

---

**End of spec.**
