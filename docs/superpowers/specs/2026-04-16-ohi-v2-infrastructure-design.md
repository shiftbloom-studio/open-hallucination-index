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
1. End-to-end AWS + Cloudflare architecture and traffic path (Cloudflare is the public edge, AWS hosts Lambda + Secrets + S3 + CloudWatch)
2. Terraform repo layout, layering, state backend, bootstrap approach, CF-provider integration
3. CI/CD pipeline (GitHub Actions + OIDC + ECR + GitFlow + release-tag-triggered rollout for compute, manual dispatch for other layers)
4. Secrets Manager entries, seeding discipline (mix of manual + TF-managed for CF-generated secrets), rotation posture
5. Observability (CloudWatch dashboard, 4 metric filters, AWS Budgets) and cost posture (Gemini uncapped Phase 1 with Google-side quota cap as safety net)
6. PC-side Docker Compose with `pc-prod` + `local-dev` profiles (four data stores + Cloudflare tunnel client + two HTTP proxy sidecars; **no inbound ports, no port forwarding, no firewall changes**)
7. DNS + Cloudflare wiring (single delegated `ohi.shiftbloom.studio` zone, NS records at user's current DNS provider)
8. Local dev workflow (no AWS dev env; Compose `local-dev` profile with host-port bindings on 127.0.0.1 only and separate volumes)
9. Phased rollout (six ordered phases I.0 → I.5 with explicit gate criteria and automated validation scripts)
10. Risks + items explicitly deferred to a future phase

**Defers (out of scope for Phase 1, to be reconsidered later):**
- Multi-AZ / multi-region / backups / failover
- AWS dev/staging environments (local-dev profile covers the need)
- Automated secret rotation
- GDPR subject-access-request tooling
- Horizontal scaling / provisioned concurrency
- Local-LLM hosting (Phase 1 = remote Gemini only)
- Third-party APM (Datadog, Sentry)
- Cloudflare Logpush (paid) — CF dashboard analytics cover Phase 1 needs
- Frontend rewrite (Next.js) — separate sub-project 3

---

## 1. Architecture recap (brainstorm Section 1, revised per user feedback 2026-04-16)

**Pivot from the earlier CloudFront+AWS-WAF design:** we consolidate the public edge onto **Cloudflare's free tier** (user is already using CF Tunnel; use their CDN / WAF / Access / TLS too — one fewer surface, one fewer bill, and no AWS-WAF/CloudFront spend).

```
┌──────────┐       HTTPS          ┌─────────────────────────────────────┐
│  User    │ ───────────────────> │ Cloudflare (free tier)              │
│  (any)   │                      │  - Proxied DNS + Universal TLS       │
└──────────┘                      │  - Free Managed Ruleset + Bot Fight  │
                                  │  - Rate limiting rules               │
                                  │  - Cache rules                       │
                                  │  - Bot Fight Mode                    │
                                  └───────────┬─────────────────────────┘
                                              │ HTTPS; CF adds header
                                              │   X-OHI-Edge-Secret: <shared>
                                              │ via Transform Rule
                                              ▼
                                  ┌─────────────────────────────────────┐
                                  │ Lambda Function URL                 │
                                  │  auth_type = NONE                   │
                                  │  invoke_mode = RESPONSE_STREAM      │
                                  │  Middleware rejects requests        │
                                  │  without matching edge secret       │
                                  │  Region: eu-central-1               │
                                  │  FastAPI + Lambda Web Adapter       │
                                  └─────┬───────────────────────────┬───┘
                                        │                           │
                      HTTPS             │                           │  HTTPS
                      to Gemini         │                           │  to CF Tunnel origins
                                        ▼                           ▼
                              ┌─────────────────┐  ┌───────────────────────────────────┐
                              │ Google Gemini   │  │ Cloudflare Tunnel (same CF acct)  │
                              │ gemini-3-flash- │  │  public hostnames protected by    │
                              │ preview         │  │  CF Access service-token policy   │
                              └─────────────────┘  └──────────┬────────────────────────┘
                                                              │ outbound-only QUIC (443)
                                                              ▼
                                         ┌──────────────────────────────────┐
                                         │ User's PC (Windows + Docker)     │
                                         │  cloudflared ingress             │
                                         │  → neo4j      (HTTP :7474)       │
                                         │  → qdrant     (HTTP :6333)       │
                                         │  → postgrest  (→ postgres:5432)  │
                                         │  → webdis     (→ redis:6379)     │
                                         │  NO inbound ports, NO firewall   │
                                         │  changes, NO port forwarding     │
                                         └──────────────────────────────────┘
```

Key consequences of the pivot:
- **No AWS CloudFront.** Cloudflare handles cache, TLS, WAF, DDoS, bot management.
- **No AWS WAF.** Cloudflare free tier: Free Managed Ruleset (emergency signatures) + Bot Fight Mode + our own custom rules and rate limits via `cloudflare_ruleset`.
- **No ACM cert anywhere.** Cloudflare Universal SSL is free and auto-renewed.
- **No Route53.** DNS lives in Cloudflare for the delegated `ohi.shiftbloom.studio` zone.
- **No AWS OAC / SigV4 signing.** Lambda Function URL is reachable publicly but enforces a shared-secret header that only Cloudflare knows.
- **Cloudflare is fully Terraformed** via the `cloudflare/cloudflare` provider — zone, DNS, tunnel, tunnel routes, Access applications, WAF rules, rate-limit rules, cache rules.
- **AWS surface drops to:** Lambda, ECR, S3, Secrets Manager, KMS, CloudWatch, Budgets, SNS. That's it.

### 1.1 Hard numbers

| Parameter | Value | Source |
|---|---|---|
| AWS monthly hard cap | €150 | Budget alarm at €100 forecast / €140 actual |
| Expected traffic | < 100 monthly users | Sizing assumption |
| LLM provider (Phase 1) | `gemini-3-flash-preview` | User choice; budget NOT capped in Phase 1 per user directive (see §9.1 R10) |
| Region (workload) | `eu-central-1` (Frankfurt) | Lowest PC↔Lambda RTT |
| Public DNS zone | `ohi.shiftbloom.studio` | Delegated to Cloudflare via NS records at user's current DNS provider |
| Public hostname | `ohi.shiftbloom.studio` (apex of the delegated zone) | |
| Tunnel hostnames | `neo4j/qdrant/pg/redis.ohi.shiftbloom.studio` | Same CF zone |
| Data layer | User's PC (Docker) | Zero RDS / ElastiCache / OpenSearch |
| Estimated total AWS | ~€7–15/month | Well under cap |
| Cloudflare cost | €0 | Free tier |

### 1.2 Failure model

- **PC off → tunnel 503.** Lambda catches `httpx.ReadTimeout` / CF-origin 502, returns structured `{"status":"resting","reason":"origin-unreachable"}` with HTTP 503 + `Retry-After: 300`.
- **Gemini API error** → L7 fast-fail with `{"status":"llm_unavailable"}`.
- **Lambda cold-start** (~1.5–3s) → accepted; user-facing latency budget is dominated by Gemini (~8–15s), so cold-start is noise.
- **Gemini spend runaway (unlikely but real)** → see §9.1 R10. No daily ceiling in Phase 1 per user directive; AWS Budget alarm + Gemini-console-side limit are the safety net.

---

## 2. Terraform module layout (brainstorm Section 2, APPROVED)

### 2.1 Tooling

- **HashiCorp Terraform**, pin `required_version >= 1.10.0, < 2.0.0`.
- Providers:
  - `hashicorp/aws ~> 5.80`
  - `cloudflare/cloudflare ~> 4.40`
- No second AWS region alias (no CloudFront, no ACM).
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
│   │                            # with auth_type=NONE, invoke_mode=RESPONSE_STREAM.
│   │                            # No AWS-side auth; Cloudflare-injected
│   │                            # X-OHI-Edge-Secret header is validated by
│   │                            # FastAPI middleware (see §7.5). URL is discoverable
│   │                            # but unusable without the secret.
│   ├── iam.tf                   # execution role, Secrets-read policy, CloudWatch policy
│   ├── logs.tf                  # aws_cloudwatch_log_group, retention 14d
│   ├── outputs.tf               # function ARN, Function URL, alias ARN
│   ├── variables.tf             # image_tag (input from CI), memory_mb, timeout_s, env_vars
│   ├── versions.tf              # key = "prod/compute/terraform.tfstate"
│   └── terraform.tfvars         # defaults: memory_mb=2048, timeout_s=60
├── cloudflare/                  # CF-managed edge: zone, DNS, WAF, rate limits, cache rules,
│   │                            # tunnel, tunnel routes, Access applications. Replaces the
│   │                            # previous AWS `edge/` layer.
│   ├── main.tf                  # cloudflare_zone "ohi_shiftbloom" (already created manually
│   │                            # before this layer can apply; data source reads it)
│   ├── dns.tf                   # cloudflare_record: CNAME "@" → Lambda Function URL hostname
│   │                            # (proxied=true), plus tunnel CNAMEs auto-created by the tunnel
│   │                            # resource below
│   ├── waf.tf                   # cloudflare_ruleset phase=http_request_firewall_custom
│   │                            # for our own rules; cloudflare_ruleset phase=
│   │                            # http_ratelimit for rate-limit rules. Free-tier
│   │                            # "CF Free Managed Ruleset" + "Bot Fight Mode" are
│   │                            # enabled via account-level settings (NOT custom
│   │                            # managed rulesets — those require Pro plan).
│   ├── cache.tf                 # cloudflare_ruleset "prod-cache" (bypass /api/*, cache /health/*
│   │                            # 60s, cache static assets long)
│   ├── tunnel.tf                # cloudflare_zero_trust_tunnel_cloudflared "ohi-pc" +
│   │                            # cloudflare_zero_trust_tunnel_cloudflared_config with
│   │                            # 4 ingress rules (neo4j, qdrant, pg, redis) +
│   │                            # cloudflare_record proxied CNAMEs for each
│   ├── access.tf                # cloudflare_zero_trust_access_application (×4, one per tunnel
│   │                            # hostname) + cloudflare_zero_trust_access_service_token for
│   │                            # Lambda's tunnel auth; token value written to AWS Secrets
│   │                            # Manager via `secrets/` remote state (see §4.1)
│   ├── edge_secret.tf           # cloudflare_ruleset phase=http_request_transform adds
│   │                            # X-OHI-Edge-Secret header to all proxied requests.
│   │                            # Value comes from a TF input variable marked
│   │                            # `sensitive = true`, supplied by the release workflow
│   │                            # as -var=edge_secret=$(aws secretsmanager get-secret-value).
│   │                            # NOT stored in tfstate in plaintext — the sensitive
│   │                            # flag masks it in plan/apply output. (CF Secrets Store
│   │                            # is not supported by the cloudflare/cloudflare 4.x
│   │                            # provider yet; revisit in Phase 2.)
│   ├── outputs.tf               # tunnel UUID, public hostnames, service token id (NOT secret)
│   ├── variables.tf             # zone_id, zone_name, lambda_function_url (from compute state)
│   ├── versions.tf              # key = "prod/cloudflare/terraform.tfstate"
│   └── terraform.tfvars         # zone_name = "ohi.shiftbloom.studio"
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
       secrets ─┼─► compute ─► cloudflare ─► observability
                │      ▲             │
                └──────┘             │
                      (cloudflare writes the Access service-token
                       value back into Secrets Manager entries
                       created by `secrets/`)
```

`cloudflare` depends on `compute` because it needs the Lambda Function URL hostname for the proxied DNS record. `cloudflare` also reads `secrets/` remote state for ARNs, and writes an `aws_secretsmanager_secret_version` for the CF Access service-token (the only secret value Terraform writes — it's generated inside TF by `cloudflare_zero_trust_access_service_token`, so keeping it in Terraform is unavoidable; see §4.2).

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
| `Layer` | `bootstrap` / `storage` / `secrets` / `compute` / `cloudflare` / `observability` | Per-layer cost attribution |
| `ManagedBy` | `terraform` | Distinguish from hand-created |
| `CostCenter` | `ohi` | Budget enforcement |

### 2.6 Naming

- Prefix all named resources with `ohi-` (e.g., `ohi-api`, `ohi-alerts`, `ohi-verify-rate-limit`).
- No environment suffix (single env).
- Where AWS requires globally-unique names (S3 buckets), append `<aws-account-id>`.

---

## 3. CI/CD (brainstorm Section 3)

### 3.1 Branching + release model (per user directive 2026-04-16)

GitFlow, release-tag-triggered deploys:

```
feature/*  ──PR──►  develop  ──PR──►  main  ──tag v*.*.*──►  prod rollout
            (CI: tests)      (CI: tests)    (CI: build image + TF apply)
```

- `develop` is the integration branch. Merges from `feature/*` run full test suite on CI.
- `main` is the "ready to release" branch. Merges from `develop` require passing CI.
- **A git tag matching `v*.*.*` on `main` triggers prod rollout**: image build → ECR push → `terraform apply` on the `compute/` layer with `image_tag=<tag>`.
- Other layers (`storage/`, `secrets/`, `cloudflare/`, `observability/`) apply via manual `workflow_dispatch` — they change rarely and deserve an explicit human click.
- `bootstrap/` is applied from Fabian's laptop only (§3.5).

### 3.2 Posture

- **GitHub Actions** as the single runner. No self-hosted runners.
- **OIDC federation** (`token.actions.githubusercontent.com`) → `ohi-terraform-apply` IAM role. No long-lived AWS keys anywhere.
- **Cloudflare API token** (scoped) stored as a GitHub Actions secret `CLOUDFLARE_API_TOKEN`. Scopes:
  - `Zone:Read`, `Zone:Edit` (limited to the `ohi.shiftbloom.studio` zone)
  - `Account:Cloudflare Tunnel:Edit`
  - `Account:Access: Apps and Policies:Edit`
- **Plan on PR**, **apply on tag push (compute only) or manual dispatch (other layers)**.
- **GitHub Environment `prod`** exists, but its only purpose is to scope the OIDC role-assumption. It does NOT require a reviewer.
- Concurrency group `terraform-apply` ensures only one apply at a time across all layers.
- Concurrency group `release-rollout` ensures only one tagged deploy in flight at a time.

### 3.3 Workflows

Five workflow files under `.github/workflows/`:

#### 3.3.1 `test.yml` — develop + main PR gate
- **Trigger:** `pull_request` to `develop` or `main`, or `push` to `develop` or `main`.
- **Steps:** ruff/mypy/pytest on `src/api/`, plus frontend + MCP checks where relevant. No AWS/CF touching.

#### 3.3.2 `infra-plan.yml` — PR gate for TF changes
- **Trigger:** `pull_request` targeting `develop` or `main`, paths `infra/terraform/**`.
- **Job matrix:** one job per layer (`storage`, `secrets`, `compute`, `cloudflare`, `observability`). `bootstrap` is excluded from CI.
- **Steps:**
  1. Checkout
  2. Configure AWS creds via OIDC (role `ohi-terraform-apply`, session `gha-plan-<layer>-<pr>`)
  3. Export `CLOUDFLARE_API_TOKEN` from secrets (only needed for `cloudflare` layer plans)
  4. `terraform init` with S3 backend
  5. `terraform fmt -check -recursive`
  6. `terraform validate`
  7. `tflint --minimum-failure-severity=warning`
  8. `checkov -d . --framework terraform --quiet` (annotations, non-blocking)
  9. `terraform plan -no-color -out=tfplan`
  10. Post plan as sticky PR comment
- **Required to merge:** plan jobs green.

#### 3.3.3 `release.yml` — **release-tag triggered rollout** (the auto-deploy path)
- **Trigger:** `push` of tag matching `v*.*.*` (on any branch, but convention is the tag lives on `main`).
- **Guard:** reject if the tag is not semver or if the commit is not on `main`.
- **Jobs (sequential):**
  1. **`test`** — re-run full test suite against the tagged commit.
  2. **`build-and-push-image`**:
     - Checkout at the tag ref
     - OIDC → ECR login
     - `docker buildx build` → tag with `${{ github.ref_name }}` (e.g., `v0.2.0`) AND `prod` (moving)
     - Push both tags to ECR `ohi-api`
  3. **`deploy-compute`**:
     - OIDC, `terraform init` in `infra/terraform/compute/`
     - `terraform apply -auto-approve -var=image_tag=${{ github.ref_name }}`
     - Verify Lambda invocation: `curl https://ohi.shiftbloom.studio/health/deep` with the CF edge secret header, assert 200
     - On failure: roll back by applying with the prior `image_tag` (fetched from ECR's second-newest tag — see §3.5 "Rollback" for runbook)
  4. **`notify`**:
     - Post release notes + deploy status to a GitHub Release (auto-created from the tag)
     - SNS alert (not email spam — one message per release)

#### 3.3.4 `infra-apply.yml` — Manual dispatch for non-compute layers
- **Trigger:** `workflow_dispatch` with inputs:
  - `layer`: enum (`storage` | `secrets` | `cloudflare` | `observability`)
  - `confirm`: text input — must equal `apply` to proceed
- **Job:** OIDC, plan, apply. Posts outputs to Job Summary.

#### 3.3.5 `bootstrap-drift.yml` — nightly drift check for bootstrap
- **Trigger:** `schedule` nightly + `workflow_dispatch`.
- **Steps:** downloads the committed bootstrap state (read-only), `terraform plan` against it, alerts if non-empty. Does not apply. Uses a scoped read-only IAM user (not the apply role) for this plan only.

### 3.4 Lambda container image

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

### 3.5 Image-tag handoff (CI → Terraform)

Images are tagged with the semver release tag (e.g. `v0.2.0`) plus a moving `prod` tag. The release workflow passes `-var=image_tag=${{ github.ref_name }}` to the compute-layer apply. Digest pinning is still mandatory — Terraform only detects the image change because the data source reads the current digest for the supplied tag:

```hcl
# compute/variables.tf
variable "image_tag" {
  type    = string
  default = "prod"   # default for manual applies; release CI overrides to the semver tag
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

**Rollback** (§3.3.3 job 3's failure path, and manual runbook):
1. `aws ecr describe-images --repository-name ohi-api --query 'imageDetails[?imageTags!=null]|sort_by(@,&imagePushedAt)[-2].imageTags[0]'` → previous semver tag.
2. Re-dispatch `infra-apply.yml` with `layer=compute` and override `image_tag` to that value.

### 3.6 Bootstrap is local-only (unchanged)

`bootstrap/` is applied from Fabian's laptop once (Phase I.0). Rationale: it creates the state bucket + OIDC provider + the CI role itself — bootstrapping CI with CI is circular. Destroy is rare and intentional. The lockfile `.terraform.lock.hcl` IS committed so provider versions are reproducible.

`data.aws_ecr_image` is re-read on every plan, so pushing a new image under the `prod` tag changes the digest, which shows as a delta in `terraform plan` and triggers a redeploy on apply. For a pinned deploy, dispatch `infra-apply.yml` with `image_tag=<git-sha>` as an optional input.

Runbook steps captured in `infra/terraform/bootstrap/README.md` (written as part of the implementation plan).

---

## 4. Secrets + env config (brainstorm Section 4)

### 4.1 Secrets Manager entries

Most are created empty by Terraform and seeded out-of-band. The two marked **(TF-managed value)** have values generated inside Terraform (the Cloudflare provider returns them on create) and are written with an `aws_secretsmanager_secret_version` resource — unavoidable because the values don't exist until TF runs.

| Secret name | Purpose | Shape | Consumer | Seeding |
|---|---|---|---|---|
| `ohi/gemini-api-key` | Google Generative AI key | string | Lambda | Manual (AWS CLI) |
| `ohi/internal-bearer-token` | Long random for MCP/CI/benchmark trusted callers | string (≥64 chars) | Lambda (auth) | Manual |
| `ohi/cloudflared-tunnel-token` | Tunnel credential for PC cloudflared container | string | PC (not Lambda) | **TF-managed value** — `cloudflare_zero_trust_tunnel_cloudflared.tunnel_token` |
| `ohi/cf-access-service-token` | CF Access service-token pair for Lambda→tunnel auth | JSON `{client_id, client_secret}` | Lambda | **TF-managed value** — `cloudflare_zero_trust_access_service_token.*` |
| `ohi/cf-edge-secret` | Shared secret that CF adds as header; Lambda middleware rejects requests without it | string | Lambda (verify) + Cloudflare (set) | Manual — generated with `openssl rand -hex 32`, seeded to both AWS Secrets Manager and Cloudflare Transform Rule secret |
| `ohi/labeler-tokens` | Expert + adjudicator labeler tokens (algorithm §11) | JSON `{expert: [...], adjudicator: [...]}` | Lambda | Manual |
| `ohi/pc-origin-credentials` | Basic-auth for PostgREST + WebDIS (defense-in-depth behind CF Access) | JSON `{pg_rest_user, pg_rest_pass, webdis_user, webdis_pass}` | Lambda + PC | Manual |
| `ohi/neo4j-credentials` | Neo4j auth | JSON `{user, password}` | Lambda + PC | Manual |

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
- **Lazy fetch** on first call per secret (not all at cold start — some endpoints don't need Gemini key).
- **In-memory TTL cache**, TTL 10 min. Manual rotation via CLI takes effect within 10 min without redeploy.
- **Decryption errors** → raise `ConfigurationError` at import time; Lambda cold-start fails loudly rather than returning subtly-broken responses.
- **Bootstrap-grace tolerance** for `ohi/cf-access-service-token` and `ohi/cloudflared-tunnel-token`: these two are created empty by the `secrets/` layer and populated by the later `cloudflare/` layer (§2.3). Between the two applies, the Lambda may cold-start with a missing or empty value. The `SecretsLoader` treats `ResourceNotFoundException` or an empty `SecretString` on these **two** specific secrets as non-fatal → the routes that need them return 503 `{"status":"bootstrapping"}`. All other secrets use strict fail-fast.

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
| `OHI_GEMINI_DAILY_CEILING_EUR` | `0` (= unlimited in Phase 1 per user directive; see §9.1 R10) | tfvars (tunable) |
| `OHI_CF_TUNNEL_HOSTNAME_NEO4J` | `neo4j.ohi.shiftbloom.studio` | tfvars |
| `OHI_CF_TUNNEL_HOSTNAME_QDRANT` | `qdrant.ohi.shiftbloom.studio` | tfvars |
| `OHI_CF_TUNNEL_HOSTNAME_PG_REST` | `pg.ohi.shiftbloom.studio` | tfvars |
| `OHI_CF_TUNNEL_HOSTNAME_WEBDIS` | `redis.ohi.shiftbloom.studio` | tfvars |
| `OHI_CF_EDGE_SECRET_ARN` | full ARN to `ohi/cf-edge-secret` | `secrets/` remote state |
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

- Lambda writes **structured JSON** via `structlog` to stdout → CloudWatch Logs group `/aws/lambda/ohi-api`, retention **7 days** (user directive — minimal footprint).
- Log lines carry: `ts`, `level`, `msg`, `request_id`, `route`, `ip_hash`, `pipeline_stage`, `duration_ms`, `status_code`, plus pipeline-specific fields.
- **No CloudFront (CloudFront is gone).** Cloudflare's own "Logs" tab on the dashboard provides edge analytics on the free tier (aggregate metrics, cached-vs-origin ratio, top countries, top status codes). Per-request logs (Logpush) is paid — skipped.

### 5.2 Metric filters

Four CloudWatch Logs metric filters under namespace `OHI/App`, all counting `1` per match:

| Filter | Pattern | Metric | Purpose |
|---|---|---|---|
| pipeline-error | `{ $.level = "ERROR" && $.pipeline_stage = * }` | `PipelineError` | Any L1–L7 error |
| rate-limit-app | `{ $.msg = "rate_limit_triggered" }` | `RateLimitApp` | L1/L2 app-layer rate-limit hits |
| pc-origin-timeout | `{ $.msg = "pc_origin_timeout" }` | `PCOriginTimeout` | Cloudflare tunnel origin unreachable |
| cold-start | `{ $.msg = "lambda_cold_start" }` | `LambdaColdStart` | Cold-start diagnostic |

(Removed `gemini-ceiling-hit` — no ceiling in Phase 1. See §9.1 R10.)

### 5.3 Alarms

| Alarm | Metric | Threshold | Action |
|---|---|---|---|
| Budget-forecast-100€ | AWS/Billing (Budget) | forecast ≥ €100 | SNS → email |
| Budget-actual-140€ | AWS/Billing (Budget) | actual ≥ €140 | SNS → email |
| Lambda-5xx-rate | AWS/Lambda `Errors / Invocations` | > 0.10 for 3×5-min | SNS → email |
| PC-origin-timeout-rate | `OHI/App/PCOriginTimeout` | > 5 in 15 min | SNS → email |

SNS topic `ohi-alerts` has a single email subscription `fabian@shiftbloom.studio`. Cloudflare-side alerting (WAF block spikes, origin health) uses Cloudflare Notifications — free tier allows email notifications, configured in the CF dashboard once (not Terraformed in Phase 1).

### 5.4 Budgets

Two AWS Budgets in `observability/budgets.tf`:

1. **Forecast alarm** — €100/month, forecast exceed → SNS (early warning).
2. **Actual alarm** — €140/month, actual exceed → SNS (hard-stop warning).

No Budget **Action** (e.g., stop Lambda) — too drastic for Phase 1, and Lambda doesn't integrate cleanly with Budget Actions anyway.

### 5.5 Gemini spend posture (Phase 1: no daily ceiling)

Per user directive 2026-04-16: **no Lambda-enforced daily Gemini ceiling in Phase 1.** The algorithm spec's §11 L3 ceiling logic (`gemini_cost_eur` counter in Redis) stays implemented as dead code gated by `OHI_GEMINI_DAILY_CEILING_EUR > 0`; defaults to `0` = disabled.

Safety net in Phase 1 relies on:
1. **AWS Budget alarms** (€100 forecast / €140 actual) as a late-warning tripwire.
2. **Cloudflare rate limits** at edge (100 req/min per IP on `/verify`, 1000 req/hour per IP global).
3. **Google Cloud side** quota: the user sets a hard API-quota cap in the Google Cloud console for the Gemini project (documented in runbook).

See §9.1 R10 for the accepted risk.

### 5.6 CloudWatch dashboard

Single dashboard `ohi-prod`:
- Row 1: Lambda invocations, errors, throttles, duration p50/p95/p99 (Frankfurt).
- Row 2: Metric filters (PipelineError, RateLimitApp, PCOriginTimeout, ColdStart) sparklines.
- Row 3: AWS Cost per day (Cost Explorer CUR query widget). Budgets widget.
- (Cloudflare-side metrics live in the CF dashboard, not mirrored to CloudWatch.)

### 5.7 Cost tracking convention

- Every resource tagged `CostCenter=ohi`.
- Monthly review: Cost Explorer filter `CostCenter=ohi`, group by `Layer`. Target < €50/mo steady state.

---

## 6. PC-side container stack

### 6.0 No inbound ports. No port forwarding. No firewall rules.

This is worth emphasizing because it's the single biggest operational win of the chosen architecture:
- Cloudflare Tunnel is **outbound-only**. The PC's `cloudflared` container initiates a QUIC connection to Cloudflare's edge on port 443 and keeps it open.
- The AVM Fritz.Box at the user's residential connection **does not need any port-forwarding rules**.
- Windows Defender Firewall (or any residential firewall) **does not need inbound exceptions**.
- Data-store containers (Neo4j, Qdrant, Postgres, Redis) and their HTTP proxies (PostgREST, WebDIS) **do not expose any host ports** in prod mode — they're reachable only from inside the Docker network (where `cloudflared` lives).
- The ISP can give you any dynamic IP or even CGNAT — doesn't matter. Tunnel works.

If local dev ports are desired for testing, they live in a separate Compose profile (§6.2).

### 6.1 New file: `docker/compose/pc-data.yml`

Standalone file — does NOT modify the existing `docker-compose.yml` (V1 stack stays functional). Uses Compose **profiles** to distinguish between prod PC hosting and local development.

```bash
# Prod PC hosting (no host ports, only CF Tunnel access):
docker compose -f docker/compose/pc-data.yml --profile pc-prod \
  --env-file docker/compose/.env.pc-data up -d

# Local dev (host ports bound on 127.0.0.1 for direct app connections,
# no cloudflared, no tunnel):
docker compose -f docker/compose/pc-data.yml --profile local-dev \
  --env-file docker/compose/.env.pc-data up -d
```

Implementation approach (exact YAML in the implementation plan; Compose profiles + an override file is the cleanest pattern):

- **Base file `pc-data.yml`** — data services (neo4j, qdrant, postgres, redis), HTTP proxies (postgrest, webdis). No host ports exposed. Services are in profile `pc-prod` only.
- **Override file `pc-data.local-dev.yml`** — extends each service with:
  - `profiles: [local-dev]`
  - `ports: ["127.0.0.1:<port>:<port>"]`
  - `volumes: ["<svc>-data-dev:/data"]` (separate volume names so dev data doesn't touch prod data)
- **`cloudflared` service** — in the base file with `profiles: [pc-prod]` only. Excluded from `local-dev`.

Invocations:
```bash
# Prod (default): only base file, pc-prod profile, no host ports
docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d

# Local dev: both files, local-dev profile, ports on 127.0.0.1, no tunnel
docker compose -f docker/compose/pc-data.yml \
               -f docker/compose/pc-data.local-dev.yml \
               --profile local-dev up -d
```

Pinned images (same across both profiles):
- `neo4j:5-community`
- `qdrant/qdrant:v1.12.0`
- `postgres:16-alpine`
- `postgrest/postgrest:v12.2.0`
- `redis:7-alpine`
- `nicolas/webdis:0.1.23` (HTTP proxy, exposes port 7379)
- `cloudflare/cloudflared:2026.3.0` (prod only)

Named volumes:
- Prod: `neo4j-data`, `qdrant-data`, `pg-data`, `redis-data`
- Dev: `neo4j-data-dev`, `qdrant-data-dev`, `pg-data-dev`, `redis-data-dev`

### 6.2 Local dev vs prod PC — the critical separation

- **Prod PC profile (`pc-prod`)**: Data stores listen only on Docker-internal network; `cloudflared` is the only way in. Volumes persist prod data.
- **Local-dev profile (`local-dev`)**: Data stores bind host ports on `127.0.0.1` (not `0.0.0.0` — localhost only, no LAN exposure); `cloudflared` is NOT started; FastAPI app runs on the host (or in an `api` container pointing at the Docker network). **Uses different Docker volumes from prod** to avoid test data contaminating prod data — volume names are `neo4j-data-dev`, `pg-data-dev`, etc., in the local-dev profile.
- Running both on the same PC **simultaneously** is not supported (volume names could collide depending on Compose version). The dev workflow stops `pc-prod` before starting `local-dev`, or uses a second machine.

### 6.3 DNS rebinding — not a concern here

Local dev uses only `localhost`/`127.0.0.1` and Docker service names (`neo4j`, `qdrant`, etc.). No public hostnames resolve to private IPs, so there's no DNS rebinding attack surface. Prod traffic goes through Cloudflare's edge and tunnel; the PC-side has no public hostname that resolves to its residential IP. **Do not add `*.localtest.me` or `*.nip.io` to the dev setup** — those ARE rebind-vulnerable public names.

### 6.4 Why two HTTP proxies on PC

Cloudflare Tunnel exposes HTTP/HTTPS hostnames to Lambda, but Lambda cannot run `cloudflared access tcp` client. So raw-TCP services (Postgres, Redis) are exposed as HTTP:
- **PostgREST** → Postgres. Lambda builds requests like `GET /feedback_pending?claim_id=eq.X` or `POST /rpc/<fn>`. PostgREST handles row-level security + auth.
- **WebDIS** → Redis. Lambda calls `GET /SET/key/value` and `GET /INCR/key`. Simple subset of Redis operations Lambda actually needs.

Neo4j (HTTP Cypher at :7474) and Qdrant (HTTP at :6333) are natively HTTP — no proxy needed.

### 6.5 `.env.pc-data` template (new file, gitignored)

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

### 6.6 Initial schema seed

A one-shot Phase I.3 step runs the Postgres schema DDL from the algorithm repo (`scripts/db/init.sql`) against the PC postgres container:
```bash
docker compose -f docker/compose/pc-data.yml exec postgres \
  psql -U ohi_app -d ohi_prod -f /docker-entrypoint-initdb.d/init.sql
```

---

## 7. DNS + Cloudflare wiring

### 7.1 One delegated zone, managed by Cloudflare

A single Cloudflare zone `ohi.shiftbloom.studio` is delegated from the user's current DNS provider. Inside that zone, Cloudflare holds **all** OHI records — public API, tunnel hostnames, anything future. The parent zone `shiftbloom.studio` is untouched by this project.

| Zone | Managed by | How to create |
|---|---|---|
| `shiftbloom.studio` | Your current DNS provider | (unchanged) |
| `ohi.shiftbloom.studio` | **Cloudflare** (subdomain zone, free tier) | You add it as a zone in CF dashboard; CF issues NS records; you add NS records at your current provider |

### 7.2 The records you add at your current DNS provider (one-time, Phase I.0)

Cloudflare will show you the exact NS pair when you add `ohi.shiftbloom.studio` as a zone. It's typically four records:

```
ohi.shiftbloom.studio   NS   <assigned-1>.ns.cloudflare.com.
ohi.shiftbloom.studio   NS   <assigned-2>.ns.cloudflare.com.
# (sometimes 4 NS records for anycast)
```

Add those once at your current DNS provider. After DNS propagates (minutes), Cloudflare owns everything under `ohi.shiftbloom.studio`. **You never edit DNS records again** — the Terraform `cloudflare/` layer writes them all automatically.

### 7.3 What Terraform writes inside the Cloudflare zone

Managed by the `cloudflare/` layer:

| Hostname | Type | Target | Proxy |
|---|---|---|---|
| `ohi.shiftbloom.studio` (apex) | CNAME | Lambda Function URL hostname | **Proxied (orange cloud)** — CF handles TLS/WAF/cache |
| `neo4j.ohi.shiftbloom.studio` | CNAME | `<tunnel-uuid>.cfargotunnel.com` | Proxied |
| `qdrant.ohi.shiftbloom.studio` | CNAME | `<tunnel-uuid>.cfargotunnel.com` | Proxied |
| `pg.ohi.shiftbloom.studio` | CNAME | `<tunnel-uuid>.cfargotunnel.com` | Proxied |
| `redis.ohi.shiftbloom.studio` | CNAME | `<tunnel-uuid>.cfargotunnel.com` | Proxied |

The tunnel CNAMEs are auto-generated by the `cloudflare_record`-alongside-`cloudflare_zero_trust_tunnel_cloudflared` resources — you never need to look up a UUID manually.

### 7.4 Cloudflare Access (fully Terraformed per user directive)

For each of the four tunnel hostnames, the `cloudflare/` layer creates:
- A `cloudflare_zero_trust_access_application` with `session_duration = "24h"`.
- A `cloudflare_zero_trust_access_policy` with `decision = "allow"` and `include = [{ service_token = [cloudflare_zero_trust_access_service_token.lambda.id] }]`.
- A single `cloudflare_zero_trust_access_service_token.lambda` whose `client_id` and `client_secret` are written to AWS Secrets Manager entry `ohi/cf-access-service-token` as JSON.

Lambda presents the token on every tunnel request:
```http
CF-Access-Client-Id:     <client_id>
CF-Access-Client-Secret: <client_secret>
```
Cloudflare validates, strips the headers, and proxies to the PC via tunnel. Requests without the token get rejected at CF edge before ever touching the PC.

### 7.5 Public edge: the shared-secret header pattern

`ohi.shiftbloom.studio` is proxied by Cloudflare in front of the Lambda Function URL (which has `auth_type = NONE`). We close the "direct-to-Lambda-URL" bypass by:
- A **Cloudflare Transform Rule** (managed by the `cloudflare/` layer) adds header `X-OHI-Edge-Secret: <value>` to every request going to origin.
- FastAPI middleware (`api/middleware/edge_secret.py`, built by the algorithm sub-project) reads the expected value from Secrets Manager entry `ohi/cf-edge-secret` and rejects any request whose header doesn't match with 403.

**The Lambda Function URL itself is discoverable** (it's a `https://<fn-url-id>.lambda-url.eu-central-1.on.aws/` URL) but unusable without the secret. Rotating the secret:
1. `openssl rand -hex 32 > newsecret.txt`
2. Update in AWS Secrets Manager: `aws secretsmanager put-secret-value --secret-id ohi/cf-edge-secret --secret-string "$(cat newsecret.txt)"`
3. Update in Cloudflare: edit the Transform Rule, paste new value, save.
4. Both take effect within ~1 minute; middleware TTL cache refreshes within 10 min (§4.3).

### 7.6 What you do manually once (vs. what Terraform does)

| Step | Who | When |
|---|---|---|
| Add `ohi.shiftbloom.studio` as zone in CF dashboard | You | Phase I.0 |
| Add NS records at current DNS provider | You | Phase I.0 |
| Create Cloudflare API token (scopes in §3.2); paste as GH secret `CLOUDFLARE_API_TOKEN` | You | Phase I.0 |
| Generate `cf-edge-secret` and seed both sides | You (CLI) | Phase I.1 |
| Everything else (zone records, tunnel, routes, Access apps, policies, WAF rules, cache rules, Transform Rule logic) | Terraform | Every apply |

---

## 8. Phased rollout

Each phase has explicit entry and exit criteria. Phases are ordered — do not skip.

### 8.1 Phase I.0 — Bootstrap (local laptop + Cloudflare setup)

**Goal:** establish AWS bootstrap resources, Cloudflare zone delegation, API token.

**Steps:**
1. AWS account ready, CLI profile `ohi-admin` configured with temporary admin creds.
2. In Cloudflare dashboard: add `ohi.shiftbloom.studio` as a zone (free tier). Copy NS records.
3. At current DNS provider: add the NS records delegating `ohi.shiftbloom.studio` to Cloudflare. Wait for propagation (`dig NS ohi.shiftbloom.studio` shows CF nameservers).
4. In Cloudflare: create an API token with scopes listed in §3.2. Save to a password manager.
5. `cd infra/terraform/bootstrap && terraform init && terraform apply`.
6. Add the API token as GitHub repo secret `CLOUDFLARE_API_TOKEN`. Add `AWS_ROLE_ARN` + `AWS_REGION` as GitHub repo variables.
7. Commit the bootstrap layer's `terraform.lock.hcl` (NOT its `terraform.tfstate`).

**Exit gate:**
- `aws s3 ls s3://ohi-tfstate-<acct>` works.
- `dig NS ohi.shiftbloom.studio @1.1.1.1` returns Cloudflare nameservers.
- `curl -H "Authorization: Bearer $CF_TOKEN" https://api.cloudflare.com/client/v4/user/tokens/verify` returns `status=active`.
- Admin IAM user revoked.

### 8.2 Phase I.1 — Storage + secrets + stub Lambda + Cloudflare edge

**Goal:** public HTTPS endpoint at `https://ohi.shiftbloom.studio` returning a stub response, full Cloudflare edge (DNS + TLS + WAF + rate limits) in place.

**Steps:**
1. Build a stub Lambda image (`docker/lambda/Dockerfile.stub`) that returns `{"status":"bootstrapping"}` HTTP 503. Push to ECR as tag `stub`.
2. Dispatch `infra-apply.yml` with `layer=storage`.
3. Dispatch `infra-apply.yml` with `layer=secrets`.
4. Generate edge secret: `openssl rand -hex 32 > cf-edge-secret.txt`. Seed to AWS: `aws secretsmanager put-secret-value --secret-id ohi/cf-edge-secret --secret-string "$(cat cf-edge-secret.txt)"`.
5. Seed other manual secrets via AWS CLI (Gemini key, internal bearer, labeler tokens, pc-origin-credentials, neo4j-credentials).
6. Dispatch `infra-apply.yml` with `layer=compute` — `image_tag=stub`. Note the Lambda Function URL output.
7. Store the edge-secret value in Cloudflare's dashboard under the zone's Transform Rules (as a rule that sets `X-OHI-Edge-Secret`). Alternative: add it as a tfvar to the cloudflare layer and let TF write the rule.
8. Dispatch `infra-apply.yml` with `layer=cloudflare` — creates DNS, WAF, rate limits, tunnel resource (token not used yet), Access applications, Transform Rule. The tunnel CNAMEs are created but point to an offline tunnel — expected.
9. Dispatch `infra-apply.yml` with `layer=observability`.

**Exit gate:**
- `curl -v https://ohi.shiftbloom.studio/health/live` returns 503 `{"status":"bootstrapping"}` via Cloudflare (check `CF-Ray` response header).
- `curl https://<fn-url>.lambda-url.eu-central-1.on.aws/health/live` (direct) returns 403 (no edge secret header).
- Cloudflare rate-limit test: 200 rapid requests from one IP get 429 after threshold.
- SNS alarm email received for a forced budget-forecast test.

### 8.3 Phase I.2 — Real Lambda image via first release tag

**Goal:** replace stub with real OHI API via the release workflow. Proves the release pipeline works end-to-end.

**Steps:**
1. Merge algorithm Phase 1 remaining tasks (1.9 SSE, 1.10 rate-limit, 1.14 acceptance bench) to `develop` → then PR to `main`.
2. Tag `git tag v0.1.0 && git push origin v0.1.0`.
3. `release.yml` fires automatically: test → build image → push to ECR as `v0.1.0` + `prod` → `terraform apply` on compute with `image_tag=v0.1.0` → health-check assertion.
4. Verify `/health/deep` returns `{"status":"ok", "origin":"unreachable"}` — the Lambda is healthy but the PC tunnel is not yet up.

**Exit gate:**
- GitHub Release page for `v0.1.0` is created with deploy status green.
- `/health/deep` reports `origin.status == "unreachable"` (expected — PC tunnel not up yet).
- Lambda cold-start duration logged; < 4s p99 on first hit after redeploy.

### 8.4 Phase I.3 — PC tunnel up (end-to-end)

**Goal:** PC Docker stack running; CF tunnel connected; Lambda reaches PC through the tunnel; full `/verify` pipeline runs.

**Steps:**
1. On PC: `git pull`, then fill `docker/compose/.env.pc-data` with values from the `ohi/cloudflared-tunnel-token`, `ohi/neo4j-credentials`, and `ohi/pc-origin-credentials` secrets.
2. `docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d`.
3. Verify `cloudflared` logs show "Connection established" to CF edge.
4. Run Postgres schema init: `docker compose -f docker/compose/pc-data.yml --profile pc-prod exec postgres psql -U ohi_app -d ohi_prod -f /schemas/init.sql`.
5. Run ingestion (`gui_ingestion_app`) against the local Neo4j + Qdrant to populate seed data.
6. From Lambda (via `/verify`): `curl -X POST https://ohi.shiftbloom.studio/api/v2/verify -d '{"text":"The Eiffel Tower is in Berlin."}'` — expect a verdict.

**Exit gate:**
- Full verify response with evidence + calibrated verdict.
- PC-off test: `docker compose ... stop cloudflared`; API returns 503 `{"status":"resting"}` within 5s.
- Run ingestion → confirm data present in Qdrant + Neo4j via `/health/deep`.

### 8.5 Phase I.4 — Calibration seeding + public publish

**Goal:** generate initial calibration set from HuggingFace public splits; publish artifacts publicly.

**Steps:**
1. Add workflow `.github/workflows/calibration-seed.yml` — triggered via `workflow_dispatch` only.
2. Workflow runs `scripts/calibration/synthesize_phase2_calibration.py` against the live `https://ohi.shiftbloom.studio/api/v2/verify` API (it's an open endpoint, so CI just calls it like any user would). Internal bearer token used to bypass CF rate limits.
3. Outputs written to public S3 bucket `ohi-artifacts-public-<acct>` (separate from private artifacts bucket) under `calibration/<date>/`. Bucket has a public read-only policy; website-hosted or CloudFront-alias skipped (direct S3 URL is fine for open-source artifact distribution).
4. Workflow commits the calibration metadata summary as a GitHub Release asset attached to the current release tag.

**Exit gate:**
- `https://ohi-artifacts-public-<acct>.s3.eu-central-1.amazonaws.com/calibration/<date>/calibration_set.json` returns HTTP 200 publicly.
- README.md in the bucket root documents the seeding process and data provenance.
- `/health/deep` reports calibration loaded.

### 8.6 Phase I.5 — Alarms + runbooks + cutover

**Goal:** confirm all alarms fire; complete runbooks.

**Steps:**
1. Deliberately trip each alarm (force a 500 error, simulate PC-origin timeout, push a synthetic high-cost day via AWS Cost Explorer manual alarm test).
2. Verify SNS email received for each.
3. Write runbooks: `docs/runbooks/rotate-secret.md`, `rotate-edge-secret.md`, `incident-response-basic.md`, `pc-compose-start.md`, `rollback-deploy.md`, `cloudflare-api-token-rotate.md`.
4. Tag `v0.1.1` if anything changed.

**Exit gate:**
- All alarms test-fired, SNS email received.
- Cost Explorer shows the first week of traffic; cost within forecast.
- All runbooks exist and have been dry-run once.

---

## 9. Risks + deferred items

### 9.1 Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Residential ISP flakiness** on PC side | High | Medium | Lambda 503 w/ Retry-After; Cloudflare origin-unreachable page (free); public messaging that service is part-time |
| R2 | **Cloudflare Tunnel token exfiltration** | Low | High | Token in Secrets Manager only; tunnel routes protected by Access service tokens (double gate); rotation runbook documented |
| R3 | **Lambda cold-start** > 5s causing timeouts | Medium | Low | Image kept < 500MB; boto3 lazy import; only Gemini + secrets fetched lazily; no VPC (no ENI attach penalty) |
| R4 | **Gemini API deprecation** / preview model EOL | Medium | Medium | `OHI_GEMINI_MODEL` env var allows instant switch; algorithm spec already reserves Claude/OpenAI adapter slots |
| R5 | **Lambda Function URL response streaming limits** | Medium | Medium | Explicit test in Phase I.3 on `/verify/stream`; fallback is polling `/verdict/{id}` (algorithm supports this) |
| R6 | **Cloudflare rate-limit mis-tuning** (too tight → legitimate users 429; too loose → abuse) | Medium | Low | Start with conservative defaults (100/min/IP on /verify, 1000/hour/IP global); monitor CF analytics weekly; algorithm also enforces L1 app-layer rate limits |
| R7 | **PC power cut during write** → Postgres/Neo4j corruption | Low | Medium | Out of scope to prevent; documented that periodic PC-local backups are user responsibility (Phase 2 could add S3 backup script) |
| R8 | **Cloudflare account compromise** (lost API token, hijacked DNS) | Low | Critical | API token scoped to single zone + tunnel + Access; token stored only in GH Actions secrets; rotation runbook; MFA mandatory on CF account; no "Global API Key" used anywhere |
| R9 | **Budget alarm triggers at midnight on the 1st** (forecast resets, lag in AWS metrics) | Low | Low | Forecast alarm is informational; actual alarm at €140 is the real gate |
| **R10** | **Gemini spend runaway** (user accepted "unlimited" in Phase 1) | Low–Medium | **High** | **Accepted risk.** Safety nets: (1) Cloudflare rate limits cap request volume; (2) AWS Budget alarm at €100/€140 (covers AWS only — Gemini is a separate bill); (3) user sets a hard quota cap in Google Cloud console for the Gemini project; (4) `OHI_GEMINI_DAILY_CEILING_EUR` env var can be flipped > 0 at any time without code change. **Operator contract:** review Google Cloud Billing weekly for the first month post-launch. |

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

## 10. Cost estimate (revised after Cloudflare-as-edge pivot)

| Line item | Monthly |
|---|---|
| Lambda (~100k invocations, 2GB, avg 3s) | €1–3 |
| S3 (private artifacts + public calibration bucket, < 10GB) | €0.50–2.00 |
| Secrets Manager (8 secrets) | €3.20 (€0.40 × 8) |
| CloudWatch (logs 7d + 4 metric filters + 1 dashboard) | €0.50–2.00 |
| KMS (1 key, < 20k requests) | €1 |
| DynamoDB (state lock, negligible) | ~€0.10 |
| ECR (< 5 images × ~250MB) | €0.30 |
| **AWS subtotal** | **~€7–12** |
| Gemini API (uncapped Phase 1 — see R10) | Unknown; expected €0–20 at <100 users/month |
| Cloudflare free tier (DNS, TLS, CDN, WAF managed ruleset, rate limits, Tunnel, 50 Access seats) | €0 |
| Domain `shiftbloom.studio` | €0 (already owned) |
| **Total (AWS + Gemini at expected usage)** | **~€7–32** |

Comfortably under the €150 cap. The CloudFront+WAF removal saves ~€8–16/month vs the earlier design.

---

## 11. Decisions — log of revisions after first user review

Marked (✅ unchanged) or (🔄 revised 2026-04-16 per user feedback) or (➕ new).

1. ✅ **Email for alarms:** `fabian@shiftbloom.studio`.
2. 🔄 **Gemini daily ceiling:** **unlimited in Phase 1** (was €1/day). See §9.1 R10 for accepted risk and safety nets.
3. 🔄 **Log retention:** **7d Lambda** (was 14d). No more CloudFront S3 (CloudFront is gone).
4. 🔄 **Image tagging:** **semver tag (e.g. `v0.2.0`) + `prod` moving** (was `<git-sha>` + `prod`). Aligns with release-tag deploy model.
5. 🔄 **Image build trigger:** **push of `v*.*.*` tag** (was push to `main` with path filters). GitFlow with explicit release step.
6. 🔄 **CI apply model:** **auto-apply on release tag for `compute/` layer only; manual dispatch for all other layers** (was purely manual).
7. ✅ **GitHub environment `prod`:** exists for OIDC scoping; no required reviewer.
8. 🔄 **DNS:** single Cloudflare-delegated zone **`ohi.shiftbloom.studio`** (was split-brain `shiftbloom.studio` + `tunnel.shiftbloom.studio`). One NS delegation, Terraform handles all records.
9. ✅ **PC HTTP proxies:** PostgREST (for Postgres) + WebDIS (for Redis). Neo4j + Qdrant native HTTP.
10. 🔄 **Cloudflare management:** **fully Terraformed via `cloudflare/cloudflare` provider** (was "UI only, 4 rules"). Covers DNS, WAF, rate limits, cache rules, tunnel, Access apps, Access policies, service tokens, Transform Rules.
11. 🔄 **ECR lifecycle:** **keep last 5 images** (was 10).
12. ✅ **PC compose file:** new standalone `docker/compose/pc-data.yml`, does not modify the existing V1 `docker-compose.yml`.
13. 🔄 **Calibration seeding:** **publicly auditable** — runs via a `workflow_dispatch` GitHub Action, outputs written to a public S3 bucket, artifacts attached to the GitHub Release. No secret values involved.
14. ✅ **KMS:** one CMK `ohi-secrets` created in bootstrap, used for both state bucket and Secrets Manager.
15. 🔄 **Public edge protection (free-tier reality):** Cloudflare Free Managed Ruleset (core emergency signatures, not the paid CF Managed / OWASP Core), Bot Fight Mode, plus our own `cloudflare_ruleset` custom rules and rate-limit rules (100/min/IP on `/verify`, 1000/hour/IP global). This is genuinely sufficient for <100 users/month; upgrade to Pro plan (~€20/mo) revisited only if abuse measurable.
16. ➕ **Public edge choice:** **Cloudflare free tier** (was AWS CloudFront + AWS WAF). Saves ~€8–16/month AWS, consolidates on the CF account already used for Tunnel, removes the ACM-in-us-east-1 quirk entirely.
17. ➕ **Lambda Function URL auth:** `auth_type = NONE` + shared-secret header (`X-OHI-Edge-Secret`) enforced in FastAPI middleware. CF Transform Rule injects the secret; direct-to-URL requests without the header get 403.
18. ➕ **Dev environment:** **no AWS dev env** (cost-ruled-out); local-dev runs via `docker/compose/pc-data.yml --profile local-dev`, exposing host ports on 127.0.0.1 only, with separate Docker volumes from prod.
19. ➕ **Terraform CF provider resources (v4.x names, all verified against provider ~> 4.40):** `cloudflare_zone` (data source — zone created in UI), `cloudflare_record`, `cloudflare_zero_trust_tunnel_cloudflared`, `cloudflare_zero_trust_tunnel_cloudflared_config`, `cloudflare_zero_trust_access_application`, `cloudflare_zero_trust_access_policy`, `cloudflare_zero_trust_access_service_token`, `cloudflare_ruleset` (phases: `http_ratelimit` for rate limits, `http_request_firewall_custom` for WAF custom rules, `http_request_transform` for the edge-secret header injection, `http_request_cache_settings` for cache rules). The legacy `cloudflare_rate_limit` resource is NOT used.
20. ➕ **Google Cloud quota cap:** user sets a per-project hard API quota on the Gemini project in Google Cloud console (documented in a runbook `google-cloud-quota-cap.md`). Acts as the ultimate spend gate outside AWS.

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
2. **`infra/terraform/_shared/`** — module exposing tags + name_prefix.
3. **`infra/terraform/storage/`, `secrets/`, `compute/`, `cloudflare/`, `observability/`** — each applyable via `infra-apply.yml` or (compute only) via `release.yml`.
4. **`docker/lambda/Dockerfile`** + `docker/lambda/Dockerfile.stub`.
5. **`docker/compose/pc-data.yml`** with both `pc-prod` and `local-dev` profiles + `.env.pc-data.example`.
6. **Five `.github/workflows/` files** — `test.yml`, `infra-plan.yml`, `infra-apply.yml`, `release.yml`, `bootstrap-drift.yml`, plus `calibration-seed.yml`.
7. **FastAPI middleware** — `api/middleware/edge_secret.py` (enforces `X-OHI-Edge-Secret` header; reads from Secrets Manager with TTL cache).
8. **Runbooks** — `docs/runbooks/rotate-secret.md`, `rotate-edge-secret.md`, `cloudflare-api-token-rotate.md`, `google-cloud-quota-cap.md`, `pc-compose-start.md`, `incident-response-basic.md`, `rollback-deploy.md`.
9. **Validation scripts** — Phase-gate assertion scripts (`./scripts/validate-phase-i0.sh`, etc.) that the plan's tasks invoke to verify their own exit criteria without human judgment.

Every task in the plan must have:
- Explicit inputs (what files/outputs are required from prior tasks)
- Explicit outputs (what it writes)
- An automated validation step (the task is NOT done until the validation passes)
- No ambiguous "configure X" steps — every step has concrete commands

---

**End of spec.**
