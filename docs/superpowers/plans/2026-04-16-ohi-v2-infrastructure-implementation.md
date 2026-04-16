# OHI v2 Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship all the code and configuration required to deploy OHI v2 to production — Terraform-managed AWS (Lambda + Secrets + S3 + CloudWatch), fully-Terraformed Cloudflare edge (DNS + WAF + Tunnel + Access), GitHub Actions CI/CD (GitFlow + release-tag auto-deploy), PC-side Docker Compose (pc-prod + local-dev profiles), FastAPI edge-secret middleware, runbooks, and phase-validation scripts. After running this plan end-to-end, a human operator can follow the bootstrap runbook to bring prod online under €150/month.

**Architecture:** User → Cloudflare (free tier: DNS + TLS + WAF + rate limits + cache + Transform Rule that adds `X-OHI-Edge-Secret` header) → Lambda Function URL (auth_type=NONE, invoke_mode=RESPONSE_STREAM) → FastAPI on Lambda Web Adapter → (Gemini API) + (Cloudflare Tunnel → PC Docker stack: neo4j / qdrant / postgres+postgrest / redis+webdis). No CloudFront, no AWS WAF, no ACM, no Route53. **No inbound ports** anywhere on the PC side (CF Tunnel is outbound-only).

**Tech Stack:**
- **Terraform** 1.10 with providers `hashicorp/aws ~> 5.80` and `cloudflare/cloudflare ~> 4.40`
- **AWS** services: Lambda (container), Secrets Manager, S3, KMS, CloudWatch, SNS, Budgets, ECR, DynamoDB (state lock), IAM OIDC
- **Cloudflare** (free tier): Zone, DNS, Ruleset (WAF/ratelimit/transform/cache), Zero Trust Tunnel + Access
- **Python** 3.12, **FastAPI**, **AWS Lambda Web Adapter** `0.8.4`
- **Docker Compose** v2 with profiles
- **GitHub Actions** with OIDC federation; no long-lived AWS keys
- **PostgREST** `v12.2.0` and **WebDIS** `0.1.23` as HTTP sidecars for Postgres/Redis

**Companion spec:** [`docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md`](../specs/2026-04-16-ohi-v2-infrastructure-design.md). This plan strictly implements that spec; do not deviate without updating the spec first.

---

## Scope check

This plan covers exactly the infrastructure sub-project (sub-project 2 of 3 in the OHI v2 rewrite). It does NOT cover the algorithm work (already shipped on branch `feat/ohi-v2-foundation`) or the Next.js frontend rewrite (sub-project 3, not started). Single plan is appropriate; the work is cohesive and interdependent.

---

## Branch strategy

Execute this plan on a **new branch** `feat/ohi-v2-infra` cut from `feat/ohi-v2-foundation`:

```bash
# From repo root on local main
git fetch origin
git checkout -b feat/ohi-v2-infra origin/feat/ohi-v2-foundation
```

The spec and plan docs live under `docs/superpowers/` which is gitignored on `feat/*` branches (see `.gitignore`). Before execution begins, copy the two relevant docs into the branch workspace as untracked working copies (they'll stay gitignored):

```bash
# From the worktree / branch checkout
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp /path/to/local/main/docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md \
   docs/superpowers/specs/
cp /path/to/local/main/docs/superpowers/plans/2026-04-16-ohi-v2-infrastructure-implementation.md \
   docs/superpowers/plans/
```

All commits in this plan land on `feat/ohi-v2-infra`. Do NOT push to remote until the user explicitly requests it.

---

## File structure — what this plan creates

### Terraform (`infra/terraform/`)

```
infra/terraform/
├── README.md
├── .tflint.hcl
├── .checkov.yml
├── bootstrap/                   # LOCAL STATE, committed lockfile only
│   ├── main.tf
│   ├── outputs.tf
│   ├── variables.tf
│   ├── versions.tf
│   ├── terraform.tfvars
│   ├── README.md                # cold-start runbook
│   └── .gitignore
├── _shared/                     # consumed as `module.shared` by each layer
│   ├── main.tf
│   ├── outputs.tf
│   ├── variables.tf
│   └── versions.tf
├── storage/                     # S3 artifacts (private + public) buckets
├── secrets/                     # Secrets Manager entries (empty on create)
├── compute/                     # Lambda (container image), IAM, logs, Function URL
├── cloudflare/                  # Zone, DNS, WAF, rate limits, cache, Tunnel, Access
└── observability/               # CloudWatch dashboard, metric filters, alarms, SNS, Budgets
```

Each non-bootstrap layer has the same shape:
```
<layer>/
├── main.tf           # data sources, remote state reads, module "shared"
├── <concern>.tf      # one or more; named for what they provision
├── outputs.tf
├── variables.tf
├── versions.tf       # backend "s3" { key = "prod/<layer>/terraform.tfstate" }
└── terraform.tfvars  # non-secret defaults only
```

### Docker (`docker/`)

```
docker/
├── lambda/
│   ├── Dockerfile                  # real OHI Lambda container
│   └── Dockerfile.stub             # bootstrapping 503 responder
└── compose/
    ├── pc-data.yml                 # base PC-side services (pc-prod profile)
    ├── pc-data.local-dev.yml       # override: host ports on 127.0.0.1, no tunnel
    ├── .env.pc-data.example        # template; real .env is gitignored
    └── init/
        └── 01-ohi-schema.sql       # Postgres schema for spec §12 tables
```

### FastAPI code (`src/api/`)

The `src/api` package layout is **flat**: the `pyproject.toml` declares `packages = ["config", "server", "models", …]` at `src/api/` level, and tests live at **repo-root `tests/api/...`**, not under `src/api/tests/`. Imports are bare: `from config.secrets_loader import …` NOT `from api.config.secrets_loader import …`.

```
src/api/config/
├── infra_env.py                    # NEW — infra env var accessor; does NOT touch settings.py
└── secrets_loader.py               # NEW — SecretsLoader singleton with TTL cache

src/api/server/middleware/
├── __init__.py                     # MODIFIED — also export EdgeSecretMiddleware
└── edge_secret.py                  # NEW — X-OHI-Edge-Secret enforcement

src/api/server/app.py               # MODIFIED — register EdgeSecretMiddleware (env-gated)

tests/api/config/
└── test_secrets_loader.py          # NEW

tests/api/server/middleware/
└── test_edge_secret.py             # NEW

tests/api/server/
└── test_app_edge_secret_wiring.py  # NEW
```

> **Hands-off rule.** Per the handoff checkpoint, `src/api/config/settings.py`, `src/api/config/dependencies.py`, `src/api/adapters/null_graph.py`, the `gui_ingestion_app/*` tree, and any stray `nul` / `api/` artifacts are user WIP and MUST NOT be edited. This plan only creates new files in `src/api/config/` and `src/api/server/middleware/` and modifies `src/api/server/app.py` + `src/api/server/middleware/__init__.py` (both of which are part of the V2 algorithm Phase 1 work already shipped).
>
> **Pytest invocation**: run `pytest` from `$REPO/src/api/` (the pyproject.toml lives there; `testpaths = ["tests"]` resolves to the repo-root `tests/` via a pytest `rootdir` walk-up — confirm during execution that `pytest tests/api/...` from `src/api/` picks up the right files; if it doesn't, invoke pytest from the repo root with `pytest -c src/api/pyproject.toml tests/api/...`).

### GitHub Actions (`.github/workflows/`)

```
.github/workflows/
├── test.yml                    # PR test suite
├── infra-plan.yml              # PR TF plan, matrix per layer
├── infra-apply.yml             # manual workflow_dispatch for non-compute layers
├── release.yml                 # tag v*.*.* → image build + compute apply
├── bootstrap-drift.yml         # nightly drift check on bootstrap
└── calibration-seed.yml        # manual, populates public calibration bucket
```

### Runbooks (`docs/runbooks/`)

```
docs/runbooks/
├── bootstrap-cold-start.md
├── rotate-secret.md
├── rotate-edge-secret.md
├── cloudflare-api-token-rotate.md
├── google-cloud-quota-cap.md
├── pc-compose-start.md
├── incident-response-basic.md
└── rollback-deploy.md
```

### Validation scripts (`scripts/infra/`)

```
scripts/infra/
├── validate-phase-i0.sh
├── validate-phase-i1.sh
├── validate-phase-i2.sh
├── validate-phase-i3.sh
├── validate-phase-i4.sh
└── validate-phase-i5.sh
```

---

## Phase conventions

- Every task uses TDD where applicable (Python code). Terraform tasks substitute `terraform validate` + `tflint` + `checkov` for unit tests; the plan step explicitly runs them.
- **Use absolute paths in Bash commands** because the repo is on Windows + Git Bash and cwd resets between invocations. The absolute repo root is: `C:/Users/Fabia/Documents/shiftbloom/git/open-hallucination-index/`. In plan steps below, the repo root is abbreviated as `$REPO` — export it once at session start: `export REPO=/c/Users/Fabia/Documents/shiftbloom/git/open-hallucination-index`.
- Every task ends with one commit. No task spans commits; no commit spans tasks.
- Every commit message follows Conventional Commits: `feat(infra): …`, `test(infra): …`, `docs(runbook): …`, `chore(ci): …`.
- **No remote pushes until the user explicitly requests it.**
- Skip a task silently if its target file already exists AND its content matches the spec — this can happen if the plan is re-executed after a partial run. Verify via `git diff` before assuming "done".

---

## Phase 0 — Prerequisites + repo skeleton

Goal: cut the branch, create top-level scaffolding, set up lint configs.

### Task 0.1: Cut the feat branch

**Files:** none (git operation only)

- [ ] **Step 1: Cut the branch**

```bash
cd "$REPO" && git fetch origin && git checkout -b feat/ohi-v2-infra origin/feat/ohi-v2-foundation
```

- [ ] **Step 2: Copy spec + plan docs into the branch workspace (untracked)**

The spec + plan live on `main` but `docs/superpowers/` is gitignored on `feat/*`. Copy them as working-copies so the executor can read them from the branch:

```bash
# From the feat branch worktree (wherever $REPO points):
mkdir -p "$REPO/docs/superpowers/specs" "$REPO/docs/superpowers/plans"

# Extract from main's git tree directly (avoids path-resolution on Windows)
cd "$REPO" && git show main:docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md > docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md
cd "$REPO" && git show main:docs/superpowers/plans/2026-04-16-ohi-v2-infrastructure-implementation.md > docs/superpowers/plans/2026-04-16-ohi-v2-infrastructure-implementation.md

# Also grab the algorithm spec for cross-reference
cd "$REPO" && git show main:docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md > docs/superpowers/specs/2026-04-16-ohi-v2-algorithm-design.md 2>/dev/null || true
```

The .gitignore on the branch keeps these untracked; no accidental commit.

- [ ] **Step 3: Verify branch state**

```bash
cd "$REPO" && git status && git log --oneline -5
```
Expected: branch is `feat/ohi-v2-infra`, HEAD = tip of `origin/feat/ohi-v2-foundation`, clean working tree (spec+plan docs untracked under `docs/superpowers/`).

### Task 0.2: Top-level Terraform scaffolding + lint configs

**Files:**
- Create: `infra/terraform/README.md`
- Create: `infra/terraform/.tflint.hcl`
- Create: `infra/terraform/.checkov.yml`

- [ ] **Step 1: Create `infra/terraform/README.md`**

```markdown
# OHI v2 Infrastructure — Terraform root

See the infrastructure spec at `docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md`
and the implementation plan at `docs/superpowers/plans/2026-04-16-ohi-v2-infrastructure-implementation.md`.

## Layered structure

This repository uses **layered root stacks** — each subdirectory is its own
Terraform root module with its own state file. Apply in dependency order.

| Layer | Depends on | Applied by | Cadence |
|---|---|---|---|
| `bootstrap/` | — | Human laptop, once | Extremely rarely (disaster) |
| `_shared/` | — | Consumed as module by all other layers | N/A |
| `storage/` | bootstrap | `.github/workflows/infra-apply.yml` | Rarely |
| `secrets/` | bootstrap | `.github/workflows/infra-apply.yml` | Rarely |
| `compute/` | storage, secrets | `.github/workflows/release.yml` (auto on tag) | Per release |
| `cloudflare/` | compute | `.github/workflows/infra-apply.yml` | Rarely |
| `observability/` | compute, cloudflare | `.github/workflows/infra-apply.yml` | Rarely |

See `bootstrap/README.md` for the one-time cold-start runbook.

## Backends

All non-bootstrap layers use an S3 remote backend with DynamoDB state locking,
provisioned by `bootstrap/`. Each layer's `versions.tf` declares the exact key
under `prod/<layer>/terraform.tfstate`.

## Pre-commit checks

Run from each layer directory before committing TF changes:

```bash
terraform fmt -check
terraform validate
tflint --minimum-failure-severity=warning
checkov -d . --framework terraform --quiet
```
```

- [ ] **Step 2: Create `infra/terraform/.tflint.hcl`**

```hcl
plugin "aws" {
  enabled = true
  version = "0.30.0"
  source  = "github.com/terraform-linters/tflint-ruleset-aws"
}

config {
  call_module_type = "local"
  force            = false
}

rule "terraform_required_version" { enabled = true }
rule "terraform_required_providers" { enabled = true }
rule "terraform_naming_convention" { enabled = true }
rule "terraform_unused_declarations" { enabled = true }
```

- [ ] **Step 3: Create `infra/terraform/.checkov.yml`**

```yaml
framework:
  - terraform
soft-fail: true          # don't block plan; we surface warnings as PR annotations
skip-check:
  - CKV_AWS_18           # S3 access logging — not needed for <100 users, overkill
  - CKV_AWS_21           # S3 versioning — we set it where it matters (state bucket), skip elsewhere
  - CKV2_AWS_6           # S3 MFA delete — single-operator, no
  - CKV_AWS_144          # S3 cross-region replication — out of scope
  - CKV_AWS_158          # CloudWatch log groups KMS encryption — we use the default AWS-managed key
  - CKV_AWS_272          # Lambda code-signing — deferred to Phase 2
```

- [ ] **Step 4: Commit**

```bash
cd "$REPO" && git add infra/terraform/README.md infra/terraform/.tflint.hcl infra/terraform/.checkov.yml && git commit -m "chore(infra): scaffold infra/terraform with README + lint configs"
```

---

## Phase I.0 — Bootstrap

Goal: author the `_shared` module, the `bootstrap/` local-state root module, and the Phase I.0 validation script + runbook. This phase does NOT apply the bootstrap — it only produces the code for a human to later apply from their laptop.

### Task I.0.1: `_shared/` module

**Files:**
- Create: `infra/terraform/_shared/versions.tf`
- Create: `infra/terraform/_shared/variables.tf`
- Create: `infra/terraform/_shared/main.tf`
- Create: `infra/terraform/_shared/outputs.tf`

- [ ] **Step 1: Create `versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0"
}
```

- [ ] **Step 2: Create `variables.tf`**

```hcl
variable "layer" {
  description = "Name of the layer consuming this shared module (for the Layer tag)."
  type        = string
  validation {
    condition     = contains(["bootstrap", "storage", "secrets", "compute", "cloudflare", "observability"], var.layer)
    error_message = "layer must be one of bootstrap, storage, secrets, compute, cloudflare, observability."
  }
}

variable "region" {
  description = "AWS region for workload resources."
  type        = string
  default     = "eu-central-1"
}

variable "project" {
  description = "Project short-name prefix for resources."
  type        = string
  default     = "ohi"
}

variable "environment" {
  description = "Environment short-name. Single-env design = prod."
  type        = string
  default     = "prod"
}
```

- [ ] **Step 3: Create `main.tf`**

```hcl
# Values-only module. No resources here.
```

- [ ] **Step 4: Create `outputs.tf`**

```hcl
output "tags" {
  description = "Default tag map applied via provider default_tags in each layer."
  value = {
    Project     = var.project
    Environment = var.environment
    Layer       = var.layer
    ManagedBy   = "terraform"
    CostCenter  = var.project
  }
}

output "name_prefix" {
  description = "Resource name prefix. Single-env means no env suffix."
  value       = var.project
}

output "region" {
  description = "Workload region."
  value       = var.region
}

output "project" {
  value = var.project
}

output "environment" {
  value = var.environment
}
```

- [ ] **Step 5: Validate**

```bash
cd "$REPO/infra/terraform/_shared" && terraform init -backend=false && terraform validate
```
Expected: `Success! The configuration is valid.`

- [ ] **Step 6: Commit**

```bash
cd "$REPO" && git add infra/terraform/_shared/ && git commit -m "feat(infra): _shared TF module exposing tags + name_prefix"
```

### Task I.0.2: `bootstrap/` versions + variables + .gitignore

**Files:**
- Create: `infra/terraform/bootstrap/versions.tf`
- Create: `infra/terraform/bootstrap/variables.tf`
- Create: `infra/terraform/bootstrap/terraform.tfvars`
- Create: `infra/terraform/bootstrap/.gitignore`

- [ ] **Step 1: Create `versions.tf` (no backend block — local state)**

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = module.shared.tags
  }
}

module "shared" {
  source = "../_shared"
  layer  = "bootstrap"
  region = var.region
}
```

- [ ] **Step 2: Create `variables.tf`**

```hcl
variable "region" {
  description = "AWS region for workload resources (not CloudFront — we don't use CloudFront)."
  type        = string
  default     = "eu-central-1"
}

variable "github_org" {
  description = "GitHub org (owner) for OIDC subject claims."
  type        = string
}

variable "github_repo" {
  description = "GitHub repo name for OIDC subject claims."
  type        = string
}

variable "github_branch_pattern" {
  description = "Branch/tag pattern the OIDC-assumable role accepts. `ref:refs/heads/*` + `ref:refs/tags/v*`."
  type        = list(string)
  default = [
    "repo:%s/%s:ref:refs/heads/main",
    "repo:%s/%s:ref:refs/heads/develop",
    "repo:%s/%s:ref:refs/tags/v*",
    "repo:%s/%s:pull_request",
  ]
}
```

- [ ] **Step 3: Create `terraform.tfvars`**

```hcl
region      = "eu-central-1"
github_org  = "shiftbloom-studio"
github_repo = "open-hallucination-index"
```

- [ ] **Step 4: Create `.gitignore`**

```
# Bootstrap uses LOCAL state — commit lockfile but not state.
*.tfstate
*.tfstate.*
.terraform/
.terraform.tfstate.lock.info
```

- [ ] **Step 5: Commit**

```bash
cd "$REPO" && git add infra/terraform/bootstrap/versions.tf infra/terraform/bootstrap/variables.tf infra/terraform/bootstrap/terraform.tfvars infra/terraform/bootstrap/.gitignore && git commit -m "feat(infra): bootstrap TF versions + variables + tfvars"
```

### Task I.0.3: `bootstrap/` — state bucket + DynamoDB lock table + KMS

**Files:**
- Create: `infra/terraform/bootstrap/main.tf`

- [ ] **Step 1: Create `main.tf` — first half (state bucket + DynamoDB + KMS)**

```hcl
data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  prefix     = module.shared.name_prefix
}

# ---------------------------------------------------------------------------
# KMS key used by the state bucket AND Secrets Manager
# ---------------------------------------------------------------------------
resource "aws_kms_key" "ohi_secrets" {
  description             = "OHI state + secrets KMS CMK"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_alias" "ohi_secrets" {
  name          = "alias/${local.prefix}-secrets"
  target_key_id = aws_kms_key.ohi_secrets.key_id
}

# ---------------------------------------------------------------------------
# Terraform state S3 bucket (SSE-KMS, versioned, block public)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "tfstate" {
  bucket = "${local.prefix}-tfstate-${local.account_id}"
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.ohi_secrets.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# DynamoDB state lock table
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "tfstate_lock" {
  name         = "${local.prefix}-tfstate-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
```

- [ ] **Step 2: Validate**

```bash
cd "$REPO/infra/terraform/bootstrap" && terraform init -backend=false && terraform validate
```
Expected: valid. No apply — apply is human-operated per §3.6 of the spec.

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add infra/terraform/bootstrap/main.tf && git commit -m "feat(infra): bootstrap — KMS key, state bucket, DynamoDB lock table"
```

### Task I.0.4: `bootstrap/` — ECR repo with lifecycle policy

**Files:**
- Modify: `infra/terraform/bootstrap/main.tf` (append)

- [ ] **Step 1: Append ECR repo to `main.tf`**

```hcl
# ---------------------------------------------------------------------------
# ECR repository for the Lambda container image
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "ohi_api" {
  name                 = "${local.prefix}-api"
  image_tag_mutability = "MUTABLE"  # `prod` tag moves across releases

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ohi_secrets.arn
  }
}

resource "aws_ecr_lifecycle_policy" "ohi_api" {
  repository = aws_ecr_repository.ohi_api.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPatternList = ["v*", "prod", "stub"]
          countType     = "imageCountMoreThan"
          countNumber   = 5
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
    ]
  })
}
```

- [ ] **Step 2: Validate**

```bash
cd "$REPO/infra/terraform/bootstrap" && terraform validate
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add infra/terraform/bootstrap/main.tf && git commit -m "feat(infra): bootstrap — ECR repo ohi-api with 5-image lifecycle"
```

### Task I.0.5: `bootstrap/` — GitHub OIDC + apply role + drift read-only role

**Files:**
- Modify: `infra/terraform/bootstrap/main.tf` (append)

- [ ] **Step 1: Append OIDC resources to `main.tf`**

```hcl
# ---------------------------------------------------------------------------
# GitHub OIDC provider (one per account; idempotent)
# ---------------------------------------------------------------------------
# Modern AWS IAM auto-validates the cert chain for token.actions.githubusercontent.com,
# so thumbprint_list is not strictly required, but providing one satisfies older
# providers. Leave as empty; AWS will default to its managed cert.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]  # fallback; AWS ignores it on modern accounts
}

# ---------------------------------------------------------------------------
# Main apply role — assumed by CI for plan/apply. NOT admin.
# ---------------------------------------------------------------------------
locals {
  oidc_sub_patterns = [for pat in var.github_branch_pattern : format(pat, var.github_org, var.github_repo)]
}

data "aws_iam_policy_document" "apply_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = local.oidc_sub_patterns
    }
  }
}

resource "aws_iam_role" "terraform_apply" {
  name               = "${local.prefix}-terraform-apply"
  assume_role_policy = data.aws_iam_policy_document.apply_trust.json
  max_session_duration = 3600
}

# Scoped inline policy — wide enough to manage all layers, narrow enough to not be admin.
data "aws_iam_policy_document" "apply_policy" {
  statement {
    sid = "TerraformStateBucket"
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.tfstate.arn,
      "${aws_s3_bucket.tfstate.arn}/*",
    ]
  }

  statement {
    sid = "TerraformStateLock"
    actions = [
      "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:DescribeTable",
    ]
    resources = [aws_dynamodb_table.tfstate_lock.arn]
  }

  statement {
    sid = "KMSUse"
    actions = [
      "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey",
    ]
    resources = [aws_kms_key.ohi_secrets.arn]
  }

  statement {
    sid    = "LayerManagement"
    effect = "Allow"
    actions = [
      # Lambda
      "lambda:*",
      # IAM (scoped below)
      "iam:GetRole", "iam:PassRole", "iam:CreateRole", "iam:DeleteRole",
      "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:PutRolePolicy",
      "iam:DeleteRolePolicy", "iam:GetRolePolicy", "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies", "iam:CreatePolicy", "iam:DeletePolicy",
      "iam:GetPolicy", "iam:GetPolicyVersion", "iam:ListPolicyVersions",
      "iam:CreatePolicyVersion", "iam:DeletePolicyVersion", "iam:TagRole",
      "iam:UntagRole", "iam:TagPolicy", "iam:UntagPolicy",
      # Secrets Manager
      "secretsmanager:*",
      # S3 artifact buckets
      "s3:*",
      # CloudWatch Logs + Metrics + Dashboard
      "logs:*", "cloudwatch:*",
      # SNS
      "sns:*",
      # Budgets
      "budgets:*",
      # ECR
      "ecr:*",
      # IAM OIDC (read-only; bootstrap owns the provider)
      "iam:GetOpenIDConnectProvider", "iam:ListOpenIDConnectProviders",
    ]
    resources = ["*"]
  }

  # NOTE: `iam:AttachRolePolicy` appears both in the broad allow above AND in
  # this deny. Because Deny always wins, the net effect is: CI can attach any
  # managed policy to roles EXCEPT AdministratorAccess. That's the desired
  # "no admin escalation" guardrail.
  statement {
    sid       = "DenyIAMUserAndAdminEscalation"
    effect    = "Deny"
    actions   = [
      "iam:CreateUser", "iam:DeleteUser", "iam:CreateAccessKey", "iam:DeleteAccessKey",
      "iam:AttachUserPolicy", "iam:PutUserPolicy",
      "iam:AttachRolePolicy",
    ]
    resources = ["*"]
    condition {
      test     = "ArnLike"
      variable = "iam:PolicyARN"
      values   = ["arn:aws:iam::aws:policy/AdministratorAccess"]
    }
  }
}

resource "aws_iam_role_policy" "terraform_apply" {
  name   = "${local.prefix}-terraform-apply-inline"
  role   = aws_iam_role.terraform_apply.id
  policy = data.aws_iam_policy_document.apply_policy.json
}

# ---------------------------------------------------------------------------
# Drift-check read-only role — used by bootstrap-drift.yml nightly workflow.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "drift_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [format("repo:%s/%s:ref:refs/heads/main", var.github_org, var.github_repo)]
    }
  }
}

resource "aws_iam_role" "terraform_drift" {
  name               = "${local.prefix}-terraform-drift"
  assume_role_policy = data.aws_iam_policy_document.drift_trust.json
}

resource "aws_iam_role_policy_attachment" "terraform_drift_readonly" {
  role       = aws_iam_role.terraform_drift.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}
```

- [ ] **Step 2: Validate**

```bash
cd "$REPO/infra/terraform/bootstrap" && terraform validate
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add infra/terraform/bootstrap/main.tf && git commit -m "feat(infra): bootstrap — GitHub OIDC provider + scoped apply role + drift role"
```

### Task I.0.6: `bootstrap/outputs.tf`

**Files:**
- Create: `infra/terraform/bootstrap/outputs.tf`

- [ ] **Step 1: Write `outputs.tf`**

```hcl
output "state_bucket" {
  description = "S3 bucket holding all layer state files."
  value       = aws_s3_bucket.tfstate.bucket
}

output "state_lock_table" {
  description = "DynamoDB table for state locking."
  value       = aws_dynamodb_table.tfstate_lock.name
}

output "kms_key_arn" {
  description = "CMK for state bucket + Secrets Manager."
  value       = aws_kms_key.ohi_secrets.arn
}

output "kms_key_alias" {
  description = "KMS alias that other layers reference."
  value       = aws_kms_alias.ohi_secrets.name
}

output "ecr_repository_url" {
  description = "ECR repo URL for pushing Lambda images."
  value       = aws_ecr_repository.ohi_api.repository_url
}

output "ecr_repository_name" {
  value = aws_ecr_repository.ohi_api.name
}

output "github_oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.github.arn
}

output "terraform_apply_role_arn" {
  description = "Store this in GitHub repo vars as AWS_ROLE_ARN."
  value       = aws_iam_role.terraform_apply.arn
}

output "terraform_drift_role_arn" {
  description = "Store this in GitHub repo vars as AWS_DRIFT_ROLE_ARN."
  value       = aws_iam_role.terraform_drift.arn
}

output "aws_region" {
  description = "Store this in GitHub repo vars as AWS_REGION."
  value       = var.region
}

output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}
```

- [ ] **Step 2: Validate**

```bash
cd "$REPO/infra/terraform/bootstrap" && terraform validate && terraform fmt -check
```

- [ ] **Step 3: tflint**

```bash
cd "$REPO/infra/terraform/bootstrap" && tflint --minimum-failure-severity=warning --config=$REPO/infra/terraform/.tflint.hcl
```
Expected: no errors. Warnings OK.

- [ ] **Step 4: Commit**

```bash
cd "$REPO" && git add infra/terraform/bootstrap/outputs.tf && git commit -m "feat(infra): bootstrap outputs — ARNs + names for CI consumption"
```

### Task I.0.7: `bootstrap/README.md` — cold-start runbook

**Files:**
- Create: `infra/terraform/bootstrap/README.md`
- Create: `docs/runbooks/bootstrap-cold-start.md` (pointer)

- [ ] **Step 1: Write `infra/terraform/bootstrap/README.md`**

```markdown
# Bootstrap — cold-start runbook

The bootstrap layer is applied **from your laptop, exactly once**, to provision
the AWS resources that every other layer depends on:

- S3 state bucket + DynamoDB lock table (for remote state)
- KMS CMK (`alias/ohi-secrets`) used by state + Secrets Manager
- ECR repository (`ohi-api`)
- GitHub OIDC provider + two IAM roles (apply + drift)

It uses **local state**, committed nowhere. The `.terraform.lock.hcl` IS committed
so provider versions stay reproducible.

## Prerequisites

1. AWS account with admin credentials, temporarily available as CLI profile `ohi-admin`.
2. Terraform `>= 1.10.0` installed locally.
3. `aws` CLI configured: `aws sts get-caller-identity --profile ohi-admin` returns your account.
4. **Cloudflare zone `ohi.shiftbloom.studio` must be created and DNS-delegated BEFORE any `cloudflare/` PR is opened.** The `cloudflare/` Terraform layer uses `data "cloudflare_zone"` which fails if the zone doesn't exist, which would break every `infra-plan.yml` CI run. Steps:
   a. In Cloudflare dashboard → Add Site → enter `ohi.shiftbloom.studio` → pick Free plan.
   b. Cloudflare shows 2–4 NS records (e.g. `xxx.ns.cloudflare.com`).
   c. At the DNS provider that hosts `shiftbloom.studio`, add those NS records under the `ohi` subdomain, delegating `ohi.shiftbloom.studio` → Cloudflare.
   d. Wait for propagation: `dig NS ohi.shiftbloom.studio @1.1.1.1` must show CF nameservers.
5. Cloudflare API token with scopes listed in `docs/runbooks/cloudflare-api-token-rotate.md`; store as GitHub repo secret `CLOUDFLARE_API_TOKEN` after bootstrap applies.

## Apply

```bash
export AWS_PROFILE=ohi-admin
cd "$REPO/infra/terraform/bootstrap"

# First-time: generate .terraform.lock.hcl
terraform init

# Review the plan
terraform plan -out=bootstrap.plan

# Apply
terraform apply bootstrap.plan
```

## Post-apply — set GitHub repo variables + secrets

Terraform prints the values. Add these **repo variables** (NOT secrets — they're not sensitive):

| Name | Value | Source |
|---|---|---|
| `AWS_ROLE_ARN` | `output.terraform_apply_role_arn` | bootstrap output |
| `AWS_DRIFT_ROLE_ARN` | `output.terraform_drift_role_arn` | bootstrap output |
| `AWS_REGION` | `output.aws_region` | bootstrap output (`eu-central-1`) |
| `ECR_REPOSITORY_URL` | `output.ecr_repository_url` | bootstrap output |

Create the Cloudflare API token (token scopes in `docs/runbooks/cloudflare-api-token-rotate.md`)
and add as **repo secret**:

| Name | Value |
|---|---|
| `CLOUDFLARE_API_TOKEN` | (from CF dashboard → My Profile → API Tokens) |

## Post-apply — revoke admin profile

Once CI's OIDC-assumed role is working (first workflow run succeeds), revoke the
temporary admin creds:

```bash
aws iam list-access-keys --user-name <your-admin-user>
aws iam delete-access-key --access-key-id <id> --user-name <your-admin-user>
```

## Disaster recovery

If the state bucket is lost, the committed lockfile + bootstrap code is enough
to recreate everything — `terraform init && terraform apply`. Downstream layers
must be re-applied after bootstrap recovers (their state files live in the
state bucket, which is now empty, so they'll try to re-create everything —
dangerous; consult `docs/runbooks/incident-response-basic.md`).

## Drift

The nightly `.github/workflows/bootstrap-drift.yml` uses the `terraform_drift`
role to run `terraform plan` against the committed bootstrap code and alert
on any unexpected change. If you see drift, investigate before blindly applying
local state (state bucket may have been tampered with).
```

- [ ] **Step 2: Create pointer in `docs/runbooks/bootstrap-cold-start.md`**

```markdown
# Bootstrap cold-start

See [`infra/terraform/bootstrap/README.md`](../../infra/terraform/bootstrap/README.md) for the runbook.
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add infra/terraform/bootstrap/README.md docs/runbooks/bootstrap-cold-start.md && git commit -m "docs(runbook): bootstrap cold-start procedure"
```

### Task I.0.8: Phase I.0 validation script

**Files:**
- Create: `scripts/infra/validate-phase-i0.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Validates Phase I.0 exit gate per spec §8.1.
# Usage: scripts/infra/validate-phase-i0.sh
# Exits 0 on success, non-zero with a list of failures on failure.
set -euo pipefail

fail=0
ok()   { echo "  [ok]   $1"; }
bad()  { echo "  [FAIL] $1" >&2; fail=$((fail+1)); }

echo "Phase I.0 validation"
echo "===================="

# 1. Bootstrap TF files exist
if [[ -f "infra/terraform/bootstrap/main.tf" && -f "infra/terraform/bootstrap/outputs.tf" ]]; then
  ok "bootstrap/ TF files present"
else
  bad "bootstrap/ missing main.tf or outputs.tf"
fi

# 2. _shared module
if [[ -d "infra/terraform/_shared" ]]; then
  ok "_shared module present"
else
  bad "_shared module missing"
fi

# 3. bootstrap validates
pushd infra/terraform/bootstrap > /dev/null
if terraform validate -no-color > /tmp/tfval.log 2>&1; then
  ok "bootstrap/ terraform validate passes"
else
  bad "bootstrap/ terraform validate failed: $(cat /tmp/tfval.log)"
fi
popd > /dev/null

# 4. Documentation
[[ -f "infra/terraform/bootstrap/README.md" ]] && ok "bootstrap README present" || bad "bootstrap README missing"
[[ -f "infra/terraform/README.md" ]] && ok "infra/terraform/README.md present" || bad "infra/terraform/README.md missing"

# 5. lint configs
[[ -f "infra/terraform/.tflint.hcl" ]] && ok "tflint config present" || bad "tflint config missing"

# 6. [humans] GH repo vars are set — this we can't check from here; runbook states it.
echo "  [note] GitHub repo vars (AWS_ROLE_ARN, AWS_REGION, etc.) must be set manually per bootstrap/README.md"

if [[ $fail -eq 0 ]]; then
  echo "PASS"
  exit 0
else
  echo "FAIL ($fail issue(s))"
  exit 1
fi
```

- [ ] **Step 2: chmod +x and run against current working tree**

```bash
cd "$REPO" && chmod +x scripts/infra/validate-phase-i0.sh && ./scripts/infra/validate-phase-i0.sh
```
Expected: PASS (apart from the manual-check note).

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add scripts/infra/validate-phase-i0.sh && git commit -m "test(infra): Phase I.0 validation script"
```

---

## Phase I.1 — Storage + Secrets + Compute + Cloudflare + Observability

Goal: author all five non-bootstrap TF layers, the Lambda container Dockerfiles, and the FastAPI edge-secret middleware + SecretsLoader. At the end of this phase, all code is in place for a human to CI-apply the stack.

### Task I.1.1: `storage/` layer

**Files:**
- Create: `infra/terraform/storage/versions.tf`
- Create: `infra/terraform/storage/variables.tf`
- Create: `infra/terraform/storage/main.tf`
- Create: `infra/terraform/storage/outputs.tf`
- Create: `infra/terraform/storage/terraform.tfvars`

- [ ] **Step 1: Create `versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  backend "s3" {
    bucket         = "ohi-tfstate-VAR_AWS_ACCOUNT_ID"  # substituted via backend-config; see CI
    key            = "prod/storage/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "ohi-tfstate-lock"
    encrypt        = true
    kms_key_id     = "alias/ohi-secrets"
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = module.shared.tags
  }
}

module "shared" {
  source = "../_shared"
  layer  = "storage"
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  prefix     = module.shared.name_prefix
}
```

Note: `backend "s3"` values cannot use variables, so the bucket name is passed at `init` time via `-backend-config`. See `infra-plan.yml` and `infra-apply.yml` in Phase I.2.

- [ ] **Step 2: Create `variables.tf`**

```hcl
variable "region" {
  type    = string
  default = "eu-central-1"
}
```

- [ ] **Step 3: Create `main.tf`**

```hcl
# ---------------------------------------------------------------------------
# Private artifacts bucket (NLI heads, calibration, retraining reports)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.prefix}-artifacts-${local.account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-retraining-reports"
    status = "Enabled"
    filter { prefix = "retraining-reports/" }
    expiration { days = 365 }
  }

  rule {
    id     = "expire-old-eval-snapshots"
    status = "Enabled"
    filter { prefix = "eval-snapshots/" }
    expiration { days = 90 }
  }
}

# ---------------------------------------------------------------------------
# Public calibration bucket — open-source transparency
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts_public" {
  bucket = "${local.prefix}-artifacts-public-${local.account_id}"
}

resource "aws_s3_bucket_public_access_block" "artifacts_public" {
  bucket = aws_s3_bucket.artifacts_public.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "artifacts_public" {
  bucket     = aws_s3_bucket.artifacts_public.id
  depends_on = [aws_s3_bucket_public_access_block.artifacts_public]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicRead"
        Effect    = "Allow"
        Principal = "*"
        Action    = ["s3:GetObject"]
        Resource  = ["${aws_s3_bucket.artifacts_public.arn}/*"]
      },
    ]
  })
}

resource "aws_s3_bucket_cors_configuration" "artifacts_public" {
  bucket = aws_s3_bucket.artifacts_public.id

  cors_rule {
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    allowed_headers = ["*"]
    max_age_seconds = 86400
  }
}
```

- [ ] **Step 4: Create `outputs.tf`**

```hcl
output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "artifacts_bucket_arn" {
  value = aws_s3_bucket.artifacts.arn
}

output "artifacts_public_bucket" {
  value = aws_s3_bucket.artifacts_public.bucket
}

output "artifacts_public_url" {
  description = "Base URL for publicly-accessible calibration artifacts."
  value       = "https://${aws_s3_bucket.artifacts_public.bucket}.s3.${var.region}.amazonaws.com"
}
```

- [ ] **Step 5: Create `terraform.tfvars`**

```hcl
region = "eu-central-1"
```

- [ ] **Step 6: Validate**

```bash
cd "$REPO/infra/terraform/storage" && terraform init -backend=false && terraform validate && terraform fmt -check
```

- [ ] **Step 7: Commit**

```bash
cd "$REPO" && git add infra/terraform/storage/ && git commit -m "feat(infra): storage layer — private + public S3 artifacts buckets"
```

### Task I.1.2: `secrets/` layer

**Files:**
- Create: `infra/terraform/secrets/versions.tf`
- Create: `infra/terraform/secrets/variables.tf`
- Create: `infra/terraform/secrets/main.tf`
- Create: `infra/terraform/secrets/outputs.tf`
- Create: `infra/terraform/secrets/terraform.tfvars`

- [ ] **Step 1: Create `versions.tf`** (same shape as storage; backend key = `prod/secrets/terraform.tfstate`)

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  backend "s3" {
    bucket         = "ohi-tfstate-VAR_AWS_ACCOUNT_ID"
    key            = "prod/secrets/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "ohi-tfstate-lock"
    encrypt        = true
    kms_key_id     = "alias/ohi-secrets"
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = module.shared.tags }
}

module "shared" {
  source = "../_shared"
  layer  = "secrets"
  region = var.region
}

data "aws_kms_alias" "ohi_secrets" {
  name = "alias/ohi-secrets"
}
```

- [ ] **Step 2: Create `variables.tf`**

```hcl
variable "region" {
  type    = string
  default = "eu-central-1"
}
```

- [ ] **Step 3: Create `main.tf`** — seven secret entries, all empty (values seeded out-of-band or written by `cloudflare/` layer)

```hcl
locals {
  secret_names = {
    gemini_api_key            = "ohi/gemini-api-key"
    internal_bearer_token     = "ohi/internal-bearer-token"
    cloudflared_tunnel_token  = "ohi/cloudflared-tunnel-token"
    cf_access_service_token   = "ohi/cf-access-service-token"
    cf_edge_secret            = "ohi/cf-edge-secret"
    labeler_tokens            = "ohi/labeler-tokens"
    pc_origin_credentials     = "ohi/pc-origin-credentials"
    neo4j_credentials         = "ohi/neo4j-credentials"
  }
}

resource "aws_secretsmanager_secret" "this" {
  for_each = local.secret_names

  name                    = each.value
  kms_key_id              = data.aws_kms_alias.ohi_secrets.target_key_arn
  recovery_window_in_days = 0   # immediate delete; single-env accepted risk

  tags = {
    SecretRole = each.key
  }
}
```

- [ ] **Step 4: Create `outputs.tf`**

```hcl
output "secret_arns" {
  description = "Map of secret role -> ARN. Consumed by compute/ and cloudflare/ layers."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.arn }
}

output "secret_names" {
  description = "Map of secret role -> name."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.name }
}
```

- [ ] **Step 5: Create `terraform.tfvars`**

```hcl
region = "eu-central-1"
```

- [ ] **Step 6: Validate + commit**

```bash
cd "$REPO/infra/terraform/secrets" && terraform init -backend=false && terraform validate && terraform fmt -check
cd "$REPO" && git add infra/terraform/secrets/ && git commit -m "feat(infra): secrets layer — 8 empty Secrets Manager entries + KMS"
```

### Task I.1.3: `SecretsLoader` Python class (TDD)

Reads secret ARNs from env vars, fetches via boto3 with 10-min TTL cache. Critical secrets (Gemini, internal bearer) fail-fast; bootstrap-grace secrets (cf-access-service-token, cloudflared-tunnel-token) tolerate missing values.

**Files:**
- Create: `src/api/config/infra_env.py`
- Create: `src/api/config/secrets_loader.py`
- Create: `tests/api/config/test_secrets_loader.py`

- [ ] **Step 1: Write the failing test**

Create `$REPO/tests/api/config/test_secrets_loader.py`:

```python
"""Tests for SecretsLoader."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from config.secrets_loader import (
    BootstrapGraceSecret,
    ConfigurationError,
    SecretsLoader,
)


@pytest.fixture
def fake_boto_client():
    """A MagicMock standing in for boto3's SecretsManager client."""
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "abc123"}
    return client


def test_loader_fetches_secret_and_caches(fake_boto_client):
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    v1 = loader.get("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/gemini-api-key-xx")
    v2 = loader.get("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/gemini-api-key-xx")
    assert v1 == "abc123" == v2
    fake_boto_client.get_secret_value.assert_called_once()


def test_loader_json_parses_when_requested(fake_boto_client):
    fake_boto_client.get_secret_value.return_value = {
        "SecretString": json.dumps({"client_id": "cid", "client_secret": "csec"})
    }
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    v = loader.get_json("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/cf-access-token-xx")
    assert v == {"client_id": "cid", "client_secret": "csec"}


def test_loader_raises_on_resource_not_found_for_critical(fake_boto_client):
    err = {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}}
    fake_boto_client.get_secret_value.side_effect = ClientError(err, "GetSecretValue")
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    with pytest.raises(ConfigurationError):
        loader.get("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/gemini-api-key-xx")


def test_loader_tolerates_missing_bootstrap_grace_secret(fake_boto_client):
    err = {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}}
    fake_boto_client.get_secret_value.side_effect = ClientError(err, "GetSecretValue")
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    with pytest.raises(BootstrapGraceSecret):
        loader.get(
            "arn:aws:secretsmanager:eu-central-1:111:secret:ohi/cf-access-service-token-xx",
            bootstrap_grace=True,
        )


def test_loader_tolerates_empty_bootstrap_grace_value(fake_boto_client):
    fake_boto_client.get_secret_value.return_value = {"SecretString": ""}
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    with pytest.raises(BootstrapGraceSecret):
        loader.get(
            "arn:aws:secretsmanager:eu-central-1:111:secret:ohi/cloudflared-tunnel-token-xx",
            bootstrap_grace=True,
        )


def test_loader_cache_expires_after_ttl(fake_boto_client, monkeypatch):
    """After TTL elapses, the next get() re-fetches."""
    from config import secrets_loader as sl
    now = [1000.0]
    monkeypatch.setattr(sl.time, "monotonic", lambda: now[0])
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    loader.get("arn:aws:...:secret:x")
    now[0] = 2000.0  # past TTL
    loader.get("arn:aws:...:secret:x")
    assert fake_boto_client.get_secret_value.call_count == 2
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd "$REPO/src/api" && pytest ../../tests/api/config/test_secrets_loader.py -v
```
Expected: collection error (module doesn't exist yet). If pytest can't find `tests/api/…` from `src/api`, invoke from repo root instead: `cd "$REPO" && pytest -c src/api/pyproject.toml tests/api/config/test_secrets_loader.py -v`.

- [ ] **Step 3: Create `src/api/config/infra_env.py`**

```python
"""Infra environment-variable accessor.

This module is deliberately separate from `settings.py` (which is user WIP and
off-limits). It reads environment variables populated by the Lambda runtime
(set in Terraform via `aws_lambda_function.environment.variables`).

Every accessor raises `KeyError` with a clear message if the var is missing —
fail-fast at Lambda cold start is intentional.
"""
from __future__ import annotations

import os


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise KeyError(
            f"Required environment variable {name} is unset. "
            f"This value is written by Terraform's compute/ layer; check Lambda config."
        )
    return v


# --- Secret ARNs (from Secrets Manager) ---
def gemini_api_key_secret_arn() -> str:
    return _require("OHI_GEMINI_KEY_SECRET_ARN")


def internal_bearer_secret_arn() -> str:
    return _require("OHI_INTERNAL_BEARER_SECRET_ARN")


def edge_secret_arn() -> str:
    return _require("OHI_CF_EDGE_SECRET_ARN")


def cf_access_service_token_secret_arn() -> str:
    return _require("OHI_CF_ACCESS_SERVICE_TOKEN_SECRET_ARN")


def cloudflared_tunnel_token_secret_arn() -> str:
    return _require("OHI_CLOUDFLARED_TUNNEL_TOKEN_SECRET_ARN")


def labeler_tokens_secret_arn() -> str:
    return _require("OHI_LABELER_TOKENS_SECRET_ARN")


def pc_origin_credentials_secret_arn() -> str:
    return _require("OHI_PC_ORIGIN_CREDENTIALS_SECRET_ARN")


def neo4j_credentials_secret_arn() -> str:
    return _require("OHI_NEO4J_CREDENTIALS_SECRET_ARN")


# --- Tunnel hostnames ---
def tunnel_neo4j_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_NEO4J")


def tunnel_qdrant_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_QDRANT")


def tunnel_pg_rest_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_PG_REST")


def tunnel_webdis_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_WEBDIS")


# --- Runtime config ---
def gemini_model() -> str:
    return os.environ.get("OHI_GEMINI_MODEL", "gemini-3-flash-preview")


def gemini_daily_ceiling_eur() -> float:
    return float(os.environ.get("OHI_GEMINI_DAILY_CEILING_EUR", "0"))  # 0 = unlimited
```

- [ ] **Step 4: Create `src/api/config/secrets_loader.py`**

```python
"""Thin wrapper around boto3 SecretsManager with TTL cache + bootstrap-grace.

Two failure modes:
- `ConfigurationError` — critical secret missing; Lambda cold-start fails loudly.
- `BootstrapGraceSecret` — specific secrets that are legitimately empty between
  `secrets/` apply and `cloudflare/` apply (spec §4.3); callers that raise this
  should return 503 `{"status":"bootstrapping"}` rather than 500.

Importable lazily; do NOT instantiate at module import time, because Lambda
cold-start imports happen before AWS_REGION is set in some test harnesses.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError


class ConfigurationError(RuntimeError):
    """Critical secret unavailable."""


class BootstrapGraceSecret(RuntimeError):
    """Bootstrap-grace secret unavailable; caller should 503 not 500."""


@dataclass
class _CacheEntry:
    value: str
    fetched_at: float


class SecretsLoader:
    """Lazy, TTL-cached accessor for AWS Secrets Manager values.

    Usage:
        loader = SecretsLoader()
        key = loader.get(os.environ["OHI_GEMINI_KEY_SECRET_ARN"])
    """

    def __init__(self, client: Any | None = None, ttl_seconds: int = 600) -> None:
        self._client = client or boto3.client("secretsmanager")
        self._ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}

    def get(self, secret_arn: str, *, bootstrap_grace: bool = False) -> str:
        """Return the secret's string value, possibly from cache."""
        cached = self._cache.get(secret_arn)
        if cached is not None and (time.monotonic() - cached.fetched_at) < self._ttl:
            return cached.value
        try:
            resp = self._client.get_secret_value(SecretId=secret_arn)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if bootstrap_grace and code == "ResourceNotFoundException":
                raise BootstrapGraceSecret(
                    f"Secret {secret_arn} not yet populated; bootstrap-grace tolerated."
                ) from exc
            raise ConfigurationError(
                f"Cannot read secret {secret_arn}: {code}"
            ) from exc
        value = resp.get("SecretString") or ""
        if bootstrap_grace and not value:
            raise BootstrapGraceSecret(
                f"Secret {secret_arn} exists but is empty; bootstrap-grace tolerated."
            )
        self._cache[secret_arn] = _CacheEntry(value=value, fetched_at=time.monotonic())
        return value

    def get_json(self, secret_arn: str, *, bootstrap_grace: bool = False) -> Any:
        return json.loads(self.get(secret_arn, bootstrap_grace=bootstrap_grace))


# Module-level singleton for app-wide reuse. Lazily created on first attribute access.
_loader: SecretsLoader | None = None


def get_loader() -> SecretsLoader:
    global _loader
    if _loader is None:
        _loader = SecretsLoader()
    return _loader
```

- [ ] **Step 5: Run the tests, verify they pass**

```bash
cd "$REPO" && pytest -c src/api/pyproject.toml tests/api/config/test_secrets_loader.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 6: Run ruff + mypy**

```bash
cd "$REPO/src/api" && ruff format config/secrets_loader.py config/infra_env.py && ruff check config/secrets_loader.py config/infra_env.py && mypy config/secrets_loader.py config/infra_env.py
```
Expected: no errors. If `mypy` complains about boto3 stubs, add a `# type: ignore[import-not-found]` to the boto3 import — boto3-stubs is not in dev deps by default.

- [ ] **Step 7: Commit**

```bash
cd "$REPO" && git add src/api/config/infra_env.py src/api/config/secrets_loader.py tests/api/config/test_secrets_loader.py && git commit -m "feat(api): SecretsLoader with TTL cache + bootstrap-grace semantics"
```

### Task I.1.4: `EdgeSecretMiddleware` (TDD)

**Files:**
- Create: `src/api/server/middleware/edge_secret.py`
- Create: `tests/api/server/middleware/test_edge_secret.py`

- [ ] **Step 1: Write failing test**

Create `$REPO/tests/api/server/middleware/test_edge_secret.py`:

```python
"""Tests for EdgeSecretMiddleware."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.middleware.edge_secret import EdgeSecretMiddleware


@pytest.fixture
def app_with_middleware():
    app = FastAPI()

    def get_secret() -> str:
        return "correct-secret-abc123"

    app.add_middleware(EdgeSecretMiddleware, get_expected_secret=get_secret)

    @app.get("/hello")
    def hello():
        return {"msg": "hi"}

    return app


def test_rejects_missing_header(app_with_middleware):
    client = TestClient(app_with_middleware)
    r = client.get("/hello")
    assert r.status_code == 403
    assert r.json()["detail"] == "missing_edge_secret"


def test_rejects_wrong_header(app_with_middleware):
    client = TestClient(app_with_middleware)
    r = client.get("/hello", headers={"X-OHI-Edge-Secret": "wrong"})
    assert r.status_code == 403
    assert r.json()["detail"] == "invalid_edge_secret"


def test_accepts_correct_header(app_with_middleware):
    client = TestClient(app_with_middleware)
    r = client.get("/hello", headers={"X-OHI-Edge-Secret": "correct-secret-abc123"})
    assert r.status_code == 200
    assert r.json() == {"msg": "hi"}


def test_health_live_is_exempt(app_with_middleware):
    """`/health/live` must work without the header (Lambda's own readiness probes)."""
    # Add a route mimicking the health router.
    @app_with_middleware.get("/health/live")
    def live():
        return {"status": "live"}

    client = TestClient(app_with_middleware)
    r = client.get("/health/live")
    assert r.status_code == 200


def test_timing_safe_comparison_used(app_with_middleware, monkeypatch):
    """Ensure hmac.compare_digest is used (non-timing-attack-vulnerable)."""
    import hmac
    called = []

    real_compare = hmac.compare_digest

    def spy(a, b):
        called.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr("server.middleware.edge_secret.hmac.compare_digest", spy)

    client = TestClient(app_with_middleware)
    client.get("/hello", headers={"X-OHI-Edge-Secret": "correct-secret-abc123"})
    assert called, "hmac.compare_digest must be used for the comparison"
```

- [ ] **Step 2: Run, verify fails**

```bash
cd "$REPO" && pytest -c src/api/pyproject.toml tests/api/server/middleware/test_edge_secret.py -v
```

Expected: collection error (module missing).

- [ ] **Step 3: Create `src/api/server/middleware/edge_secret.py`**

```python
"""Enforce the X-OHI-Edge-Secret header set by Cloudflare's Transform Rule.

Traffic path: User → CF → (CF adds X-OHI-Edge-Secret: <shared>) → Lambda Function URL.
Lambda Function URL has `auth_type = NONE`, so without this middleware any caller
who knows the Function URL could bypass CF. This middleware closes that gap.

Design:
- Secret value is fetched lazily via a caller-supplied `get_expected_secret` callable.
  In prod, that callable reads from `SecretsLoader.get(infra_env.edge_secret_arn())`.
- `/health/live` is exempt so Lambda's own runtime health checks don't need
  the header. Other /health/* routes still require it.
- Comparison uses `hmac.compare_digest` (constant-time).
"""
from __future__ import annotations

import hmac
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

HEADER_NAME = "X-OHI-Edge-Secret"
EXEMPT_PATHS = frozenset({"/health/live"})

logger = logging.getLogger("ohi.middleware.edge_secret")


class EdgeSecretMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, get_expected_secret: Callable[[], str]) -> None:
        super().__init__(app)
        self._get_expected = get_expected_secret

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        provided = request.headers.get(HEADER_NAME, "")
        if not provided:
            logger.warning("rate_limit_triggered", extra={"reason": "missing_edge_secret", "path": request.url.path})
            return JSONResponse({"detail": "missing_edge_secret"}, status_code=403)

        try:
            expected = self._get_expected()
        except Exception:
            logger.exception("Failed to load edge secret")
            return JSONResponse({"detail": "edge_secret_unavailable"}, status_code=503)

        if not hmac.compare_digest(provided, expected):
            logger.warning("rate_limit_triggered", extra={"reason": "invalid_edge_secret", "path": request.url.path})
            return JSONResponse({"detail": "invalid_edge_secret"}, status_code=403)

        return await call_next(request)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd "$REPO" && pytest -c src/api/pyproject.toml tests/api/server/middleware/test_edge_secret.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: ruff + mypy**

```bash
cd "$REPO/src/api" && ruff format server/middleware/edge_secret.py && ruff check server/middleware/edge_secret.py && mypy server/middleware/edge_secret.py
```

- [ ] **Step 6: Commit**

```bash
cd "$REPO" && git add src/api/server/middleware/edge_secret.py tests/api/server/middleware/test_edge_secret.py && git commit -m "feat(api): EdgeSecretMiddleware enforces CF-injected shared secret"
```

### Task I.1.5: Wire `EdgeSecretMiddleware` into the FastAPI app

**Files:**
- Modify: `src/api/server/middleware/__init__.py`
- Modify: `src/api/server/app.py`

- [ ] **Step 1: Read current state of both files** to preserve whatever's already there

```bash
cat "$REPO/src/api/server/middleware/__init__.py"
cat "$REPO/src/api/server/app.py"
```

- [ ] **Step 2: Update `server/middleware/__init__.py`** — add EdgeSecretMiddleware alongside the existing (flat-import) RetentionMiddleware line

Current file (do NOT change the retention import — keep the bare `server.middleware.retention` path):
```python
from server.middleware.retention import RetentionMiddleware

__all__ = ["RetentionMiddleware"]
```

Updated:
```python
"""Server middleware exports."""
from server.middleware.edge_secret import EdgeSecretMiddleware
from server.middleware.retention import RetentionMiddleware

__all__ = ["EdgeSecretMiddleware", "RetentionMiddleware"]
```

- [ ] **Step 3: Update `server/app.py`** — register the middleware

Find the existing block:
```python
    # TODO(Task 1.10): PerIPTokenBucketMiddleware (per-IP rate limit).
    # TODO(Task 1.10): CostCeilingMiddleware (daily $ ceiling).
    # TODO(Task 1.10): InternalAuthMiddleware (internal bearer token).

    # Retention policy (Task 1.11): raw text is NOT persisted unless the
    # caller explicitly opts in via ?retain=true. See spec §11.
    app.add_middleware(RetentionMiddleware)
```

Add the EdgeSecretMiddleware registration BEFORE RetentionMiddleware (Starlette middleware runs in reverse-add order, and we want the edge check to run first):

```python
    # TODO(Task 1.10): PerIPTokenBucketMiddleware (per-IP rate limit).
    # TODO(Task 1.10): CostCeilingMiddleware (daily $ ceiling).
    # TODO(Task 1.10): InternalAuthMiddleware (internal bearer token).

    # Retention policy (Task 1.11): raw text is NOT persisted unless the
    # caller explicitly opts in via ?retain=true. See spec §11.
    app.add_middleware(RetentionMiddleware)

    # Edge-secret gate (infra sub-project): every request must carry the
    # X-OHI-Edge-Secret header that Cloudflare's Transform Rule injects. See
    # docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md §7.5.
    # Skipped gracefully if the secret ARN env var is unset — keeps local
    # development (where there's no CF in front) working unchanged.
    import os as _os  # local import to avoid polluting module scope

    if _os.environ.get("OHI_CF_EDGE_SECRET_ARN"):
        from config.infra_env import edge_secret_arn
        from config.secrets_loader import get_loader
        from server.middleware.edge_secret import EdgeSecretMiddleware

        app.add_middleware(
            EdgeSecretMiddleware,
            get_expected_secret=lambda: get_loader().get(edge_secret_arn()),
        )
```

- [ ] **Step 4: Add an integration test** that the wiring works end-to-end

Create `$REPO/tests/api/server/test_app_edge_secret_wiring.py`:

```python
"""Integration test: app.py must register EdgeSecretMiddleware when env is set."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_edge_secret_middleware_registered_when_env_set(monkeypatch):
    monkeypatch.setenv("OHI_CF_EDGE_SECRET_ARN", "arn:aws:secretsmanager:eu-central-1:1:secret:ohi/cf-edge-secret-ab")

    # Patch the SecretsLoader so we don't actually call AWS
    fake_loader = MagicMock()
    fake_loader.get.return_value = "test-secret-value"

    with patch("config.secrets_loader.get_loader", return_value=fake_loader):
        from fastapi.testclient import TestClient

        # Re-import to trigger create_app() with the patched env
        import importlib
        from server import app as app_module
        importlib.reload(app_module)

        client = TestClient(app_module.app)
        r_no_header = client.get("/health/ready")
        assert r_no_header.status_code == 403
        r_correct = client.get("/health/ready", headers={"X-OHI-Edge-Secret": "test-secret-value"})
        assert r_correct.status_code in (200, 503)  # whatever /health/ready returns when infra partial

        # /health/live exempt
        r_live = client.get("/health/live")
        assert r_live.status_code in (200, 503)


def test_edge_secret_middleware_not_registered_when_env_unset(monkeypatch):
    monkeypatch.delenv("OHI_CF_EDGE_SECRET_ARN", raising=False)
    import importlib
    from server import app as app_module
    importlib.reload(app_module)

    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)
    r = client.get("/health/ready")
    # Should not be 403 — middleware is not active
    assert r.status_code != 403
```

- [ ] **Step 5: Run tests, verify**

```bash
cd "$REPO" && pytest -c src/api/pyproject.toml tests/api/server/test_app_edge_secret_wiring.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Full test suite regression**

```bash
cd "$REPO" && pytest -c src/api/pyproject.toml tests/api
```
Expected: all tests pass (147 pre-existing + new ones from this phase).

- [ ] **Step 7: Commit**

```bash
cd "$REPO" && git add src/api/server/middleware/__init__.py src/api/server/app.py tests/api/server/test_app_edge_secret_wiring.py && git commit -m "feat(api): wire EdgeSecretMiddleware into create_app() (env-gated)"
```

### Task I.1.6: Lambda Dockerfile — stub

**Files:**
- Create: `docker/lambda/Dockerfile.stub`

- [ ] **Step 1: Write `Dockerfile.stub`**

```dockerfile
# Minimal 503 responder for Phase I.1 bring-up.
# Returns {"status":"bootstrapping"} to every path so we can verify CloudFront + WAF + Function URL wiring before the real app is ready.
FROM public.ecr.aws/lambda/python:3.12

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV AWS_LAMBDA_EXEC_WRAPPER=/opt/extensions/lambda-adapter
ENV PORT=8080
ENV READINESS_CHECK_PATH=/health/live

RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.32.0

RUN mkdir -p ${LAMBDA_TASK_ROOT}/stub
COPY <<'EOF' ${LAMBDA_TASK_ROOT}/stub/app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health/live")
def live():
    return {"status": "bootstrapping"}

@app.middleware("http")
async def bootstrapping_all(request, call_next):
    if request.url.path == "/health/live":
        return await call_next(request)
    return JSONResponse({"status": "bootstrapping"}, status_code=503)
EOF

CMD ["uvicorn", "stub.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Build locally to verify Dockerfile syntax**

```bash
cd "$REPO" && docker build -f docker/lambda/Dockerfile.stub -t ohi-api:stub-local .
```
Expected: successful build. Does NOT push.

- [ ] **Step 3: Optional smoke-test (run locally)**

```bash
docker run -d --rm -p 9000:8080 --name ohi-stub ohi-api:stub-local
sleep 3
curl -s http://127.0.0.1:9000/2015-03-31/functions/function/invocations -d '{"requestContext":{"http":{"method":"GET","path":"/health/live"}},"rawPath":"/health/live","headers":{}}'
docker stop ohi-stub
```
Expected: body containing `"status": "bootstrapping"`.

- [ ] **Step 4: Commit**

```bash
cd "$REPO" && git add docker/lambda/Dockerfile.stub && git commit -m "feat(infra): Lambda stub image — 503 bootstrapping for Phase I.1 bring-up"
```

### Task I.1.7: Lambda Dockerfile — real

**Files:**
- Create: `docker/lambda/Dockerfile`
- Create: `docker/lambda/.dockerignore`

- [ ] **Step 1: Write `docker/lambda/Dockerfile`**

```dockerfile
# Multi-stage build for the OHI v2 Lambda container.
# Stage 1 installs dependencies against a vanilla Python image for speed.
# Stage 2 uses the AWS Lambda Python base, adds the Lambda Web Adapter layer,
# and copies in the app code.

# --- Stage 1: dependency install -------------------------------------------
FROM public.ecr.aws/docker/library/python:3.12-slim AS builder

WORKDIR /build

# IMPORTANT: base pyproject.toml pulls sentence-transformers + torch (~800MB).
# Those belong on the PC/ingestion side, NOT in Lambda. We install a TRIMMED
# runtime set directly instead of `pip install /build`. If you need a new dep
# in the Lambda image, add it here explicitly — the full pyproject is for
# local dev + ingestion, not for Lambda.
RUN pip install --no-cache-dir --target /deps \
    "fastapi>=0.128,<1.0" \
    "uvicorn[standard]>=0.40" \
    "pydantic>=2.12" \
    "pydantic-settings>=2.12" \
    "httpx>=0.28" \
    "openai>=2.14" \
    "neo4j>=6.0" \
    "qdrant-client>=1.16" \
    "redis>=7.1" \
    "pyyaml>=6.0" \
    "orjson>=3.11" \
    "structlog>=24.1" \
    "boto3>=1.34"

# --- Stage 2: runtime ------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.12

# Lambda Web Adapter — intercepts Lambda events, forwards HTTP to PORT=8080
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV AWS_LAMBDA_EXEC_WRAPPER=/opt/extensions/lambda-adapter
ENV PORT=8080
ENV READINESS_CHECK_PATH=/health/ready
ENV PYTHONUNBUFFERED=1

# Copy deps first (cached layer when only app code changes)
COPY --from=builder /deps ${LAMBDA_TASK_ROOT}

# Copy app code. src/api is the *package root* in the flat layout; its immediate
# children (config/, server/, models/, adapters/, pipeline/, interfaces/, services/)
# each become importable top-level packages under ${LAMBDA_TASK_ROOT}.
COPY src/api/ ${LAMBDA_TASK_ROOT}/
# Strip files we don't want in the image (pyproject, README, tests).
RUN rm -f ${LAMBDA_TASK_ROOT}/pyproject.toml ${LAMBDA_TASK_ROOT}/README.md \
 && rm -rf ${LAMBDA_TASK_ROOT}/tests

# Bare imports: `uvicorn server.app:app`, NOT `api.server.app:app`.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

- [ ] **Step 2: Write `.dockerignore`**

```
**/__pycache__/
**/*.pyc
**/.pytest_cache/
**/.mypy_cache/
**/.ruff_cache/
**/.venv/
**/tests/
**/docs/
**/README.md
.git/
```

- [ ] **Step 3: Build locally to verify**

```bash
cd "$REPO" && docker build -f docker/lambda/Dockerfile -t ohi-api:local .
```
Expected: successful build. Image size < 500 MB (check via `docker images ohi-api:local`).

- [ ] **Step 4: Verify AWS_LAMBDA_EXEC_WRAPPER path**

```bash
docker run --rm --entrypoint sh ohi-api:local -c "ls -l /opt/extensions/"
```
Expected: `lambda-adapter` executable present.

- [ ] **Step 5: Commit**

```bash
cd "$REPO" && git add docker/lambda/Dockerfile docker/lambda/.dockerignore && git commit -m "feat(infra): Lambda container image Dockerfile (multi-stage, < 500MB)"
```

### Task I.1.8: `compute/` — versions + variables

**Files:**
- Create: `infra/terraform/compute/versions.tf`
- Create: `infra/terraform/compute/variables.tf`
- Create: `infra/terraform/compute/terraform.tfvars`

- [ ] **Step 1: `versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  backend "s3" {
    bucket         = "ohi-tfstate-VAR_AWS_ACCOUNT_ID"
    key            = "prod/compute/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "ohi-tfstate-lock"
    encrypt        = true
    kms_key_id     = "alias/ohi-secrets"
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = module.shared.tags }
}

module "shared" {
  source = "../_shared"
  layer  = "compute"
  region = var.region
}

data "aws_caller_identity" "current" {}
data "aws_kms_alias" "ohi_secrets" { name = "alias/ohi-secrets" }

data "terraform_remote_state" "secrets" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/secrets/terraform.tfstate"
    region = var.region
  }
}

data "terraform_remote_state" "storage" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/storage/terraform.tfstate"
    region = var.region
  }
}

data "aws_ecr_repository" "api" {
  name = "${module.shared.name_prefix}-api"
}

locals {
  account_id   = data.aws_caller_identity.current.account_id
  prefix       = module.shared.name_prefix
  secret_arns  = data.terraform_remote_state.secrets.outputs.secret_arns
  artifacts_bucket = data.terraform_remote_state.storage.outputs.artifacts_bucket
}
```

- [ ] **Step 2: `variables.tf`**

```hcl
variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "image_tag" {
  description = "ECR image tag to deploy. Defaults to the moving `prod` tag. CI release workflow passes the semver tag."
  type        = string
  default     = "prod"
}

variable "memory_mb" {
  type    = number
  default = 2048
}

variable "timeout_s" {
  type    = number
  default = 60
}

variable "log_retention_days" {
  type    = number
  default = 7
}

variable "tunnel_hostname_neo4j" {
  type    = string
  default = "neo4j.ohi.shiftbloom.studio"
}

variable "tunnel_hostname_qdrant" {
  type    = string
  default = "qdrant.ohi.shiftbloom.studio"
}

variable "tunnel_hostname_pg_rest" {
  type    = string
  default = "pg.ohi.shiftbloom.studio"
}

variable "tunnel_hostname_webdis" {
  type    = string
  default = "redis.ohi.shiftbloom.studio"
}

variable "gemini_model" {
  type    = string
  default = "gemini-3-flash-preview"
}

variable "gemini_daily_ceiling_eur" {
  description = "0 = unlimited (Phase 1 per spec §9.1 R10)"
  type        = number
  default     = 0
}
```

- [ ] **Step 3: `terraform.tfvars`**

```hcl
region             = "eu-central-1"
memory_mb          = 2048
timeout_s          = 60
log_retention_days = 7
```

- [ ] **Step 4: Validate + commit**

```bash
cd "$REPO/infra/terraform/compute" && terraform init -backend=false && terraform validate && terraform fmt -check
cd "$REPO" && git add infra/terraform/compute/versions.tf infra/terraform/compute/variables.tf infra/terraform/compute/terraform.tfvars && git commit -m "feat(infra): compute layer — versions + variables + tfvars"
```

### Task I.1.9: `compute/` — IAM + log group

**Files:**
- Create: `infra/terraform/compute/iam.tf`
- Create: `infra/terraform/compute/logs.tf`

- [ ] **Step 1: `iam.tf`**

```hcl
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "${local.prefix}-api-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# Basic CloudWatch Logs permissions
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Secrets read policy (all 8 secrets, KMS Decrypt scoped)
data "aws_iam_policy_document" "secrets_read" {
  statement {
    sid       = "ReadAllOhiSecrets"
    actions   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
    resources = values(local.secret_arns)
  }
  statement {
    sid       = "DecryptViaSecretsManager"
    actions   = ["kms:Decrypt"]
    resources = [data.aws_kms_alias.ohi_secrets.target_key_arn]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["secretsmanager.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "lambda_secrets" {
  name   = "${local.prefix}-api-secrets"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.secrets_read.json
}

# S3 artifacts bucket read/write (Lambda writes calibration, reads NLI heads)
data "aws_iam_policy_document" "artifacts_rw" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
    resources = [
      "arn:aws:s3:::${local.artifacts_bucket}",
      "arn:aws:s3:::${local.artifacts_bucket}/*",
    ]
  }
}

resource "aws_iam_role_policy" "lambda_artifacts" {
  name   = "${local.prefix}-api-artifacts"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.artifacts_rw.json
}
```

- [ ] **Step 2: `logs.tf`**

```hcl
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.prefix}-api"
  retention_in_days = var.log_retention_days
}
```

- [ ] **Step 3: Validate + commit**

```bash
cd "$REPO/infra/terraform/compute" && terraform validate
cd "$REPO" && git add infra/terraform/compute/iam.tf infra/terraform/compute/logs.tf && git commit -m "feat(infra): compute — Lambda exec role, secrets-read policy, log group 7d"
```

### Task I.1.10: `compute/` — Lambda function + Function URL

**Files:**
- Create: `infra/terraform/compute/lambda.tf`

- [ ] **Step 1: Write `lambda.tf`**

```hcl
data "aws_ecr_image" "api" {
  repository_name = data.aws_ecr_repository.api.name
  image_tag       = var.image_tag
}

resource "aws_lambda_function" "api" {
  function_name = "${local.prefix}-api"
  role          = aws_iam_role.lambda_exec.arn

  package_type = "Image"
  image_uri    = "${data.aws_ecr_repository.api.repository_url}@${data.aws_ecr_image.api.image_digest}"

  memory_size = var.memory_mb
  timeout     = var.timeout_s

  environment {
    variables = {
      OHI_ENV                              = "prod"
      OHI_REGION                           = var.region
      OHI_LOG_LEVEL                        = "INFO"
      OHI_GEMINI_MODEL                     = var.gemini_model
      OHI_GEMINI_DAILY_CEILING_EUR         = tostring(var.gemini_daily_ceiling_eur)
      OHI_CF_TUNNEL_HOSTNAME_NEO4J         = var.tunnel_hostname_neo4j
      OHI_CF_TUNNEL_HOSTNAME_QDRANT        = var.tunnel_hostname_qdrant
      OHI_CF_TUNNEL_HOSTNAME_PG_REST       = var.tunnel_hostname_pg_rest
      OHI_CF_TUNNEL_HOSTNAME_WEBDIS        = var.tunnel_hostname_webdis
      OHI_S3_ARTIFACTS_BUCKET              = local.artifacts_bucket

      # Secret ARNs (values fetched at runtime via SecretsLoader)
      OHI_GEMINI_KEY_SECRET_ARN                     = local.secret_arns["gemini_api_key"]
      OHI_INTERNAL_BEARER_SECRET_ARN                = local.secret_arns["internal_bearer_token"]
      OHI_CF_EDGE_SECRET_ARN                        = local.secret_arns["cf_edge_secret"]
      OHI_CF_ACCESS_SERVICE_TOKEN_SECRET_ARN        = local.secret_arns["cf_access_service_token"]
      OHI_CLOUDFLARED_TUNNEL_TOKEN_SECRET_ARN       = local.secret_arns["cloudflared_tunnel_token"]
      OHI_LABELER_TOKENS_SECRET_ARN                 = local.secret_arns["labeler_tokens"]
      OHI_PC_ORIGIN_CREDENTIALS_SECRET_ARN          = local.secret_arns["pc_origin_credentials"]
      OHI_NEO4J_CREDENTIALS_SECRET_ARN              = local.secret_arns["neo4j_credentials"]
    }
  }

  logging_config {
    log_format = "JSON"
    log_group  = aws_cloudwatch_log_group.api.name
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic,
    aws_iam_role_policy.lambda_secrets,
    aws_iam_role_policy.lambda_artifacts,
    aws_cloudwatch_log_group.api,
  ]
}

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"

  cors {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["*"]
    max_age       = 86400
  }
}
```

- [ ] **Step 2: Validate + commit**

```bash
cd "$REPO/infra/terraform/compute" && terraform validate && terraform fmt -check
cd "$REPO" && git add infra/terraform/compute/lambda.tf && git commit -m "feat(infra): compute — Lambda container + Function URL (auth=NONE, stream)"
```

### Task I.1.11: `compute/` — outputs

**Files:**
- Create: `infra/terraform/compute/outputs.tf`

- [ ] **Step 1: Write**

```hcl
output "function_arn" {
  value = aws_lambda_function.api.arn
}

output "function_name" {
  value = aws_lambda_function.api.function_name
}

output "function_url" {
  description = "Direct Lambda Function URL (not user-facing; CloudFlare proxies it)."
  value       = aws_lambda_function_url.api.function_url
}

output "function_url_hostname" {
  description = "Hostname-only form for the CF CNAME target."
  value       = replace(replace(aws_lambda_function_url.api.function_url, "https://", ""), "/", "")
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.api.name
}
```

- [ ] **Step 2: Validate + tflint + commit**

```bash
cd "$REPO/infra/terraform/compute" && terraform validate && terraform fmt -check && tflint --config=$REPO/infra/terraform/.tflint.hcl
cd "$REPO" && git add infra/terraform/compute/outputs.tf && git commit -m "feat(infra): compute — outputs (function ARN, Function URL, log group)"
```

### Task I.1.12: `cloudflare/` — versions, providers, zone data, DNS apex

**Files:**
- Create: `infra/terraform/cloudflare/versions.tf`
- Create: `infra/terraform/cloudflare/variables.tf`
- Create: `infra/terraform/cloudflare/main.tf`
- Create: `infra/terraform/cloudflare/dns.tf`
- Create: `infra/terraform/cloudflare/terraform.tfvars`

- [ ] **Step 1: `versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.40"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {
    bucket         = "ohi-tfstate-VAR_AWS_ACCOUNT_ID"
    key            = "prod/cloudflare/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "ohi-tfstate-lock"
    encrypt        = true
    kms_key_id     = "alias/ohi-secrets"
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = module.shared.tags }
}

# API token read from env var CLOUDFLARE_API_TOKEN (set in CI secrets).
provider "cloudflare" {}
```

- [ ] **Step 2: `variables.tf`**

```hcl
variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "zone_name" {
  description = "Cloudflare zone managed by this layer (apex of delegated subdomain)."
  type        = string
  default     = "ohi.shiftbloom.studio"
}

variable "cf_account_id" {
  description = "Cloudflare account ID. Find at dash.cloudflare.com → right sidebar."
  type        = string
  sensitive   = false
}

variable "edge_secret" {
  description = "Shared secret that CF Transform Rule injects as X-OHI-Edge-Secret. Pass via -var or TF_VAR_edge_secret; NOT stored in tfvars."
  type        = string
  sensitive   = true
}

variable "rate_limit_verify_per_min" {
  type    = number
  default = 100
}

variable "rate_limit_global_per_hour" {
  type    = number
  default = 1000
}
```

- [ ] **Step 3: `main.tf`**

```hcl
module "shared" {
  source = "../_shared"
  layer  = "cloudflare"
  region = var.region
}

data "aws_caller_identity" "current" {}

data "terraform_remote_state" "secrets" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/secrets/terraform.tfstate"
    region = var.region
  }
}

data "terraform_remote_state" "compute" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/compute/terraform.tfstate"
    region = var.region
  }
}

# Zone is created manually in the CF dashboard (free-tier subdomain zone setup).
# We read it as a data source and manage records/rulesets under it.
data "cloudflare_zone" "this" {
  name = var.zone_name
}

locals {
  zone_id           = data.cloudflare_zone.this.id
  account_id        = var.cf_account_id
  lambda_fn_hostname = data.terraform_remote_state.compute.outputs.function_url_hostname
  secret_arns       = data.terraform_remote_state.secrets.outputs.secret_arns
}
```

- [ ] **Step 4: `dns.tf`**

```hcl
# Apex record — proxied, points to Lambda Function URL.
resource "cloudflare_record" "apex" {
  zone_id = local.zone_id
  name    = "@"
  type    = "CNAME"
  content = local.lambda_fn_hostname
  proxied = true
  ttl     = 1  # "Automatic" for proxied records
  comment = "OHI v2 public endpoint; proxied through CF (WAF, TLS, cache)."
}

# Tunnel hostnames are declared in tunnel.tf next to the tunnel resource.
```

- [ ] **Step 5: `terraform.tfvars`** (non-sensitive only)

```hcl
region      = "eu-central-1"
zone_name   = "ohi.shiftbloom.studio"
# cf_account_id and edge_secret are supplied at apply time via -var
```

- [ ] **Step 6: Validate + commit**

```bash
cd "$REPO/infra/terraform/cloudflare" && terraform init -backend=false && terraform validate && terraform fmt -check
cd "$REPO" && git add infra/terraform/cloudflare/versions.tf infra/terraform/cloudflare/variables.tf infra/terraform/cloudflare/main.tf infra/terraform/cloudflare/dns.tf infra/terraform/cloudflare/terraform.tfvars && git commit -m "feat(infra): cloudflare layer — providers, zone data, apex DNS"
```

### Task I.1.13: `cloudflare/` — WAF + rate-limit rulesets

**Files:**
- Create: `infra/terraform/cloudflare/waf.tf`

- [ ] **Step 1: Write `waf.tf`**

```hcl
# --- Free-tier WAF managed ruleset + custom rules -----------------------------
# The full CF Managed Ruleset / OWASP Core Ruleset are paid. Free tier gives:
#   - Free Managed Ruleset (auto-enabled via zone setting below)
#   - Bot Fight Mode (per-zone setting)
#   - Our own cloudflare_ruleset custom rules

# Enable free-tier managed protection via zone-level settings
resource "cloudflare_zone_settings_override" "this" {
  zone_id = local.zone_id

  settings {
    # Security
    security_level           = "medium"
    challenge_ttl            = 1800
    browser_check            = "on"
    email_obfuscation        = "on"
    server_side_exclude      = "on"
    hotlink_protection       = "off"
    # Caching
    cache_level              = "aggressive"
    browser_cache_ttl        = 14400
    always_online            = "off"
    # SSL/TLS
    ssl                      = "strict"
    automatic_https_rewrites = "on"
    min_tls_version          = "1.2"
    tls_1_3                  = "on"
    opportunistic_encryption = "on"
    always_use_https         = "on"
  }
}

# Custom WAF rules
resource "cloudflare_ruleset" "custom_waf" {
  zone_id     = local.zone_id
  name        = "ohi-prod-waf-custom"
  description = "OHI production custom WAF rules"
  kind        = "zone"
  phase       = "http_request_firewall_custom"

  rules {
    action      = "block"
    description = "Block obvious SSRF attempts (metadata endpoints)"
    expression  = <<-EOT
      (http.request.uri.path contains "/169.254.169.254") or
      (http.request.uri.path contains "/latest/meta-data") or
      (http.request.body.raw contains "169.254.169.254")
    EOT
    enabled = true
  }

  rules {
    action      = "managed_challenge"
    description = "Challenge non-browser User-Agent patterns on /feedback"
    expression  = <<-EOT
      (http.request.uri.path eq "/api/v2/feedback") and
      (http.user_agent eq "" or http.user_agent contains "curl" or http.user_agent contains "wget")
    EOT
    enabled = true
  }
}

# Rate-limit rules (modern 4.x pattern: cloudflare_ruleset phase=http_ratelimit)
resource "cloudflare_ruleset" "rate_limits" {
  zone_id     = local.zone_id
  name        = "ohi-prod-rate-limits"
  description = "Per-route rate limits"
  kind        = "zone"
  phase       = "http_ratelimit"

  rules {
    action      = "block"
    description = "POST /api/v2/verify — ${var.rate_limit_verify_per_min} req/min/IP"
    expression  = "(http.request.uri.path eq \"/api/v2/verify\") and (http.request.method eq \"POST\")"
    ratelimit {
      characteristics     = ["ip.src"]
      period              = 60
      requests_per_period = var.rate_limit_verify_per_min
      mitigation_timeout  = 60
    }
    enabled = true
  }

  rules {
    action      = "block"
    description = "Global per-IP ceiling — ${var.rate_limit_global_per_hour} req/hour/IP"
    expression  = "(http.request.uri.path wildcard \"*\")"
    ratelimit {
      characteristics     = ["ip.src"]
      period              = 3600
      requests_per_period = var.rate_limit_global_per_hour
      mitigation_timeout  = 300
    }
    enabled = true
  }
}
```

- [ ] **Step 2: Validate + commit**

```bash
cd "$REPO/infra/terraform/cloudflare" && terraform validate && terraform fmt -check
cd "$REPO" && git add infra/terraform/cloudflare/waf.tf && git commit -m "feat(infra): cloudflare — WAF custom rules + rate limits (100/min verify, 1000/h global)"
```

### Task I.1.14: `cloudflare/` — cache rules

**Files:**
- Create: `infra/terraform/cloudflare/cache.tf`

- [ ] **Step 1: Write**

```hcl
# Cache rules: bypass everything under /api/, aggressively cache /health/live,
# let defaults handle the rest.
resource "cloudflare_ruleset" "cache" {
  zone_id     = local.zone_id
  name        = "ohi-prod-cache"
  description = "Cache-policy overrides"
  kind        = "zone"
  phase       = "http_request_cache_settings"

  rules {
    action      = "set_cache_settings"
    description = "Bypass cache for all API calls"
    expression  = "(starts_with(http.request.uri.path, \"/api/\"))"
    action_parameters {
      cache = false
    }
    enabled = true
  }

  rules {
    action      = "set_cache_settings"
    description = "Cache /health/live for 60s to absorb liveness-probe floods"
    expression  = "(http.request.uri.path eq \"/health/live\")"
    action_parameters {
      cache = true
      edge_ttl {
        mode    = "override_origin"
        default = 60
      }
      browser_ttl {
        mode    = "override_origin"
        default = 60
      }
    }
    enabled = true
  }
}
```

- [ ] **Step 2: Validate + commit**

```bash
cd "$REPO/infra/terraform/cloudflare" && terraform validate
cd "$REPO" && git add infra/terraform/cloudflare/cache.tf && git commit -m "feat(infra): cloudflare — cache rules (bypass /api/, cache /health/live 60s)"
```

### Task I.1.15: `cloudflare/` — tunnel + tunnel config + tunnel CNAMEs

**Files:**
- Create: `infra/terraform/cloudflare/tunnel.tf`

- [ ] **Step 1: Write**

```hcl
# Tunnel resource — generates the tunnel token we'll store in Secrets Manager.
resource "random_password" "tunnel_secret" {
  length  = 64
  special = false
}

resource "cloudflare_zero_trust_tunnel_cloudflared" "pc" {
  account_id    = local.account_id
  name          = "ohi-pc"
  tunnel_secret = base64encode(random_password.tunnel_secret.result)
  config_src    = "cloudflare"  # we manage config via cloudflare_zero_trust_tunnel_cloudflared_config
}

# Ingress rules routing tunnel requests to PC-side docker services.
resource "cloudflare_zero_trust_tunnel_cloudflared_config" "pc" {
  account_id = local.account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.pc.id

  config {
    ingress_rule {
      hostname = "neo4j.${var.zone_name}"
      service  = "http://neo4j:7474"
    }
    ingress_rule {
      hostname = "qdrant.${var.zone_name}"
      service  = "http://qdrant:6333"
    }
    ingress_rule {
      hostname = "pg.${var.zone_name}"
      service  = "http://postgrest:3000"
    }
    ingress_rule {
      hostname = "redis.${var.zone_name}"
      service  = "http://webdis:7379"
    }
    # Catch-all (required as the last rule)
    ingress_rule {
      service = "http_status:404"
    }
  }
}

# DNS records for each tunneled hostname, proxied.
locals {
  tunnel_hostnames = {
    neo4j  = "neo4j"
    qdrant = "qdrant"
    pg     = "pg"
    redis  = "redis"
  }
}

resource "cloudflare_record" "tunnel" {
  for_each = local.tunnel_hostnames

  zone_id = local.zone_id
  name    = each.value
  type    = "CNAME"
  content = "${cloudflare_zero_trust_tunnel_cloudflared.pc.id}.cfargotunnel.com"
  proxied = true
  ttl     = 1
  comment = "OHI tunnel public hostname — protected by CF Access (see access.tf)"
}

```

> The `random` provider is declared in `versions.tf` (Task I.1.12). No separate `terraform {}` block here.

- [ ] **Step 2: Write back the tunnel token to AWS Secrets Manager**

Append to `tunnel.tf`:

```hcl
# Seed the cloudflared-tunnel-token secret from the CF-generated value.
resource "aws_secretsmanager_secret_version" "cloudflared_tunnel_token" {
  secret_id     = local.secret_arns["cloudflared_tunnel_token"]
  secret_string = cloudflare_zero_trust_tunnel_cloudflared.pc.tunnel_token
}
```

- [ ] **Step 3: Validate + commit**

```bash
cd "$REPO/infra/terraform/cloudflare" && terraform init -backend=false -upgrade && terraform validate && terraform fmt -check
cd "$REPO" && git add infra/terraform/cloudflare/tunnel.tf && git commit -m "feat(infra): cloudflare — tunnel + ingress config + 4 CNAMEs + token writeback"
```

### Task I.1.16: `cloudflare/` — Access applications + policies + service token

**Files:**
- Create: `infra/terraform/cloudflare/access.tf`

- [ ] **Step 1: Write**

```hcl
# Single service token issued to Lambda; reused across all 4 tunnel apps.
resource "cloudflare_zero_trust_access_service_token" "lambda" {
  account_id = local.account_id
  name       = "ohi-lambda-tunnel-service-token"
  duration   = "8760h"   # 1 year; we'll rotate manually in a runbook
}

# Access application + policy per tunnel hostname
locals {
  tunnel_access_apps = {
    neo4j  = "neo4j.${var.zone_name}"
    qdrant = "qdrant.${var.zone_name}"
    pg     = "pg.${var.zone_name}"
    redis  = "redis.${var.zone_name}"
  }
}

resource "cloudflare_zero_trust_access_application" "tunnel" {
  for_each = local.tunnel_access_apps

  account_id                = local.account_id
  name                      = "OHI tunnel — ${each.key}"
  domain                    = each.value
  type                      = "self_hosted"
  session_duration          = "24h"
  auto_redirect_to_identity = false
}

resource "cloudflare_zero_trust_access_policy" "tunnel_service_token" {
  for_each = local.tunnel_access_apps

  account_id     = local.account_id
  application_id = cloudflare_zero_trust_access_application.tunnel[each.key].id
  name           = "Lambda service token — ${each.key}"
  precedence     = 1
  decision       = "non_identity"
  include {
    service_token = [cloudflare_zero_trust_access_service_token.lambda.id]
  }
}

# Write the service token client_id + client_secret to Secrets Manager as JSON.
resource "aws_secretsmanager_secret_version" "cf_access_service_token" {
  secret_id = local.secret_arns["cf_access_service_token"]
  secret_string = jsonencode({
    client_id     = cloudflare_zero_trust_access_service_token.lambda.client_id
    client_secret = cloudflare_zero_trust_access_service_token.lambda.client_secret
  })
}
```

- [ ] **Step 2: Validate + commit**

```bash
cd "$REPO/infra/terraform/cloudflare" && terraform validate
cd "$REPO" && git add infra/terraform/cloudflare/access.tf && git commit -m "feat(infra): cloudflare — Access apps + service-token policy + token writeback"
```

### Task I.1.17: `cloudflare/` — edge_secret Transform Rule

**Files:**
- Create: `infra/terraform/cloudflare/edge_secret.tf`

- [ ] **Step 1: Write**

```hcl
# Cloudflare Transform Rule that injects X-OHI-Edge-Secret on every request to origin.
# Lambda's EdgeSecretMiddleware validates this header.
# The edge-secret VALUE comes from var.edge_secret (supplied by runbook / CI), and
# is ALSO written to AWS Secrets Manager so Lambda can fetch it at runtime.
resource "cloudflare_ruleset" "transform_edge_secret" {
  zone_id     = local.zone_id
  name        = "ohi-prod-transform-edge-secret"
  description = "Add X-OHI-Edge-Secret header to all requests going to origin"
  kind        = "zone"
  phase       = "http_request_transform"

  rules {
    action      = "rewrite"
    description = "Inject edge secret header"
    expression  = "(true)"
    action_parameters {
      headers {
        name      = "X-OHI-Edge-Secret"
        operation = "set"
        value     = var.edge_secret
      }
    }
    enabled = true
  }
}

# Mirror the edge-secret value into AWS Secrets Manager so Lambda can read it.
resource "aws_secretsmanager_secret_version" "cf_edge_secret" {
  secret_id     = local.secret_arns["cf_edge_secret"]
  secret_string = var.edge_secret
}
```

- [ ] **Step 2: Validate + commit**

```bash
cd "$REPO/infra/terraform/cloudflare" && terraform validate
cd "$REPO" && git add infra/terraform/cloudflare/edge_secret.tf && git commit -m "feat(infra): cloudflare — Transform Rule injects X-OHI-Edge-Secret + mirrors to Secrets Manager"
```

### Task I.1.18: `cloudflare/` — outputs

**Files:**
- Create: `infra/terraform/cloudflare/outputs.tf`

- [ ] **Step 1: Write**

```hcl
output "zone_id" {
  value = local.zone_id
}

output "apex_hostname" {
  value = var.zone_name
}

output "tunnel_id" {
  value = cloudflare_zero_trust_tunnel_cloudflared.pc.id
}

output "tunnel_cname_target" {
  value = "${cloudflare_zero_trust_tunnel_cloudflared.pc.id}.cfargotunnel.com"
}

output "service_token_client_id" {
  description = "Non-secret; useful for operator dashboards."
  value       = cloudflare_zero_trust_access_service_token.lambda.client_id
}

output "tunnel_hostnames" {
  value = { for k, v in local.tunnel_hostnames : k => "${v}.${var.zone_name}" }
}
```

- [ ] **Step 2: Validate + tflint + commit**

```bash
cd "$REPO/infra/terraform/cloudflare" && terraform validate && terraform fmt -check && tflint --config=$REPO/infra/terraform/.tflint.hcl
cd "$REPO" && git add infra/terraform/cloudflare/outputs.tf && git commit -m "feat(infra): cloudflare — outputs (zone, tunnel, hostnames)"
```

### Task I.1.19: `observability/` — versions + variables + main + SNS

**Files:**
- Create: `infra/terraform/observability/versions.tf`
- Create: `infra/terraform/observability/variables.tf`
- Create: `infra/terraform/observability/main.tf`
- Create: `infra/terraform/observability/sns.tf`
- Create: `infra/terraform/observability/terraform.tfvars`

- [ ] **Step 1: `versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  backend "s3" {
    bucket         = "ohi-tfstate-VAR_AWS_ACCOUNT_ID"
    key            = "prod/observability/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "ohi-tfstate-lock"
    encrypt        = true
    kms_key_id     = "alias/ohi-secrets"
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = module.shared.tags }
}
```

- [ ] **Step 2: `variables.tf`**

```hcl
variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "alert_email" {
  type    = string
  default = "fabian@shiftbloom.studio"
}

variable "budget_forecast_eur" {
  type    = number
  default = 100
}

variable "budget_actual_eur" {
  type    = number
  default = 140
}
```

- [ ] **Step 3: `main.tf`**

```hcl
module "shared" {
  source = "../_shared"
  layer  = "observability"
  region = var.region
}

data "aws_caller_identity" "current" {}

data "terraform_remote_state" "compute" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/compute/terraform.tfstate"
    region = var.region
  }
}

locals {
  prefix         = module.shared.name_prefix
  log_group_name = data.terraform_remote_state.compute.outputs.log_group_name
  function_name  = data.terraform_remote_state.compute.outputs.function_name
}
```

- [ ] **Step 4: `sns.tf`**

```hcl
resource "aws_sns_topic" "alerts" {
  name = "${local.prefix}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
```

- [ ] **Step 5: `terraform.tfvars`**

```hcl
region              = "eu-central-1"
alert_email         = "fabian@shiftbloom.studio"
budget_forecast_eur = 100
budget_actual_eur   = 140
```

- [ ] **Step 6: Validate + commit**

```bash
cd "$REPO/infra/terraform/observability" && terraform init -backend=false && terraform validate && terraform fmt -check
cd "$REPO" && git add infra/terraform/observability/versions.tf infra/terraform/observability/variables.tf infra/terraform/observability/main.tf infra/terraform/observability/sns.tf infra/terraform/observability/terraform.tfvars && git commit -m "feat(infra): observability layer — versions, SNS topic, alert email subscription"
```

### Task I.1.20: `observability/` — metric filters + alarms

**Files:**
- Create: `infra/terraform/observability/metric_filters.tf`
- Create: `infra/terraform/observability/alarms.tf`

- [ ] **Step 1: `metric_filters.tf`**

```hcl
locals {
  metric_namespace = "OHI/App"

  metric_filters = {
    pipeline_error = {
      pattern = "{ $.level = \"ERROR\" && $.pipeline_stage = * }"
      metric  = "PipelineError"
    }
    rate_limit_triggered = {
      pattern = "{ $.msg = \"rate_limit_triggered\" }"
      metric  = "RateLimitApp"
    }
    pc_origin_timeout = {
      pattern = "{ $.msg = \"pc_origin_timeout\" }"
      metric  = "PCOriginTimeout"
    }
    cold_start = {
      pattern = "{ $.msg = \"lambda_cold_start\" }"
      metric  = "LambdaColdStart"
    }
  }
}

resource "aws_cloudwatch_log_metric_filter" "this" {
  for_each = local.metric_filters

  name           = "${local.prefix}-${each.key}"
  log_group_name = local.log_group_name
  pattern        = each.value.pattern

  metric_transformation {
    name          = each.value.metric
    namespace     = local.metric_namespace
    value         = "1"
    default_value = "0"
  }
}
```

- [ ] **Step 2: `alarms.tf`**

```hcl
# Lambda 5xx rate
resource "aws_cloudwatch_metric_alarm" "lambda_5xx" {
  alarm_name          = "${local.prefix}-lambda-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = 0.1
  alarm_actions       = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "error_rate"
    expression  = "errors / invocations"
    label       = "5xx error rate"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      namespace   = "AWS/Lambda"
      metric_name = "Errors"
      period      = 300
      stat        = "Sum"
      dimensions = { FunctionName = local.function_name }
    }
  }

  metric_query {
    id = "invocations"
    metric {
      namespace   = "AWS/Lambda"
      metric_name = "Invocations"
      period      = 300
      stat        = "Sum"
      dimensions = { FunctionName = local.function_name }
    }
  }
}

# PC origin timeout spike
resource "aws_cloudwatch_metric_alarm" "pc_origin_timeout" {
  alarm_name          = "${local.prefix}-pc-origin-timeout-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = 5
  namespace           = local.metric_namespace
  metric_name         = "PCOriginTimeout"
  period              = 300
  statistic           = "Sum"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
```

- [ ] **Step 3: Validate + commit**

```bash
cd "$REPO/infra/terraform/observability" && terraform validate
cd "$REPO" && git add infra/terraform/observability/metric_filters.tf infra/terraform/observability/alarms.tf && git commit -m "feat(infra): observability — 4 metric filters + Lambda 5xx + PC origin alarms"
```

### Task I.1.21: `observability/` — dashboard + budgets + outputs

**Files:**
- Create: `infra/terraform/observability/dashboard.tf`
- Create: `infra/terraform/observability/budgets.tf`
- Create: `infra/terraform/observability/outputs.tf`

- [ ] **Step 1: `budgets.tf`**

```hcl
resource "aws_budgets_budget" "forecast" {
  name         = "${local.prefix}-budget-forecast"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_forecast_eur)
  limit_unit   = "EUR"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["aws:CostCenter$ohi"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }
}

resource "aws_budgets_budget" "actual" {
  name         = "${local.prefix}-budget-actual"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_actual_eur)
  limit_unit   = "EUR"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["aws:CostCenter$ohi"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }
}
```

- [ ] **Step 2: `dashboard.tf`**

```hcl
resource "aws_cloudwatch_dashboard" "ohi_prod" {
  dashboard_name = "${local.prefix}-prod"
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          region = var.region
          title  = "Lambda: invocations / errors / duration p99"
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", local.function_name, { stat = "Sum" }],
            [".", "Errors", ".", ".", { stat = "Sum" }],
            [".", "Duration", ".", ".", { stat = "p99" }],
          ]
          view   = "timeSeries"
          period = 300
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          region = var.region
          title  = "App: PipelineError / RateLimitApp / PCOriginTimeout / ColdStart"
          metrics = [
            [local.metric_namespace, "PipelineError", { stat = "Sum" }],
            [".", "RateLimitApp", { stat = "Sum" }],
            [".", "PCOriginTimeout", { stat = "Sum" }],
            [".", "LambdaColdStart", { stat = "Sum" }],
          ]
          view   = "timeSeries"
          period = 300
        }
      },
    ]
  })
}
```

- [ ] **Step 3: `outputs.tf`**

```hcl
output "sns_alerts_arn" {
  value = aws_sns_topic.alerts.arn
}

output "dashboard_url" {
  value = "https://console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${local.prefix}-prod"
}
```

- [ ] **Step 4: Validate + tflint + commit**

```bash
cd "$REPO/infra/terraform/observability" && terraform validate && terraform fmt -check && tflint --config=$REPO/infra/terraform/.tflint.hcl
cd "$REPO" && git add infra/terraform/observability/dashboard.tf infra/terraform/observability/budgets.tf infra/terraform/observability/outputs.tf && git commit -m "feat(infra): observability — dashboard + budgets (€100 forecast / €140 actual) + outputs"
```

### Task I.1.22: Phase I.1 validation script

**Files:**
- Create: `scripts/infra/validate-phase-i1.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
# Validates Phase I.1 exit gate per spec §8.2.
# Runs `terraform validate` on every non-bootstrap layer and verifies structure.
set -euo pipefail

fail=0
ok()  { echo "  [ok]   $1"; }
bad() { echo "  [FAIL] $1" >&2; fail=$((fail+1)); }

echo "Phase I.1 validation"
echo "===================="

layers=(storage secrets compute cloudflare observability)

for layer in "${layers[@]}"; do
  dir="infra/terraform/${layer}"
  if [[ ! -d "$dir" ]]; then
    bad "$layer/ directory missing"
    continue
  fi

  pushd "$dir" > /dev/null

  # Terraform validate against backend=false (for CI dry-run)
  if terraform init -backend=false -input=false > /tmp/tfinit.log 2>&1; then
    ok "$layer/ terraform init (backend=false)"
  else
    bad "$layer/ terraform init failed: $(tail -5 /tmp/tfinit.log)"
    popd > /dev/null
    continue
  fi

  if terraform validate > /tmp/tfval.log 2>&1; then
    ok "$layer/ terraform validate"
  else
    bad "$layer/ terraform validate failed: $(cat /tmp/tfval.log)"
  fi

  if terraform fmt -check -recursive > /tmp/tffmt.log 2>&1; then
    ok "$layer/ terraform fmt clean"
  else
    bad "$layer/ formatting issues: $(cat /tmp/tffmt.log)"
  fi

  popd > /dev/null
done

# App-side checks (flat-layout: invoke pytest from repo root against the src/api pyproject)
if pytest -c src/api/pyproject.toml \
    tests/api/config/test_secrets_loader.py \
    tests/api/server/middleware/test_edge_secret.py \
    tests/api/server/test_app_edge_secret_wiring.py \
    -q > /tmp/pytest.log 2>&1; then
  ok "middleware + loader tests pass"
else
  bad "middleware or loader tests failed: $(tail -20 /tmp/pytest.log)"
fi

# Docker check (Dockerfile syntax via docker buildx parse)
if docker buildx build --no-cache --progress=plain -f docker/lambda/Dockerfile --target builder -t ohi-api:plan-check . > /tmp/docker.log 2>&1; then
  ok "Dockerfile builder stage builds"
else
  bad "Dockerfile failed: $(tail -20 /tmp/docker.log)"
fi

if [[ $fail -eq 0 ]]; then echo "PASS"; else echo "FAIL ($fail issue(s))"; exit 1; fi
```

- [ ] **Step 2: chmod + run**

```bash
cd "$REPO" && chmod +x scripts/infra/validate-phase-i1.sh && ./scripts/infra/validate-phase-i1.sh
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add scripts/infra/validate-phase-i1.sh && git commit -m "test(infra): Phase I.1 validation script"
```

---

## Phase I.2 — CI/CD workflows

Goal: author all GitHub Actions workflows. Execution of workflows happens in GitHub's environment, not in this plan — but the plan produces the YAML that will work.

### Task I.2.1: `test.yml` — PR test suite

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Write**

```yaml
name: Test

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

concurrency:
  group: test-${{ github.ref }}
  cancel-in-progress: true

jobs:
  api:
    name: API tests
    runs-on: ubuntu-latest
    defaults:
      run: { working-directory: src/api }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: ruff format --check .
      - run: ruff check .
      - run: mypy .
      - run: pytest --tb=short
```

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add .github/workflows/test.yml && git commit -m "chore(ci): test.yml — PR + push gate for src/api"
```

### Task I.2.2: `infra-plan.yml` — PR plan matrix per layer

**Files:**
- Create: `.github/workflows/infra-plan.yml`

- [ ] **Step 1: Write**

```yaml
name: Infra Plan

on:
  pull_request:
    branches: [main, develop]
    paths: ["infra/terraform/**"]

concurrency:
  group: infra-plan-${{ github.ref }}-${{ matrix.layer || '' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read
  pull-requests: write

jobs:
  plan:
    name: Plan ${{ matrix.layer }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        layer: [storage, secrets, compute, cloudflare, observability]
    defaults:
      run: { working-directory: infra/terraform/${{ matrix.layer }} }
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
          role-session-name: gha-plan-${{ matrix.layer }}-${{ github.event.pull_request.number }}

      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: "1.10.3" }

      - name: Resolve account id
        id: acct
        run: echo "id=$(aws sts get-caller-identity --query Account --output text)" >> $GITHUB_OUTPUT

      - run: |
          terraform init \
            -backend-config="bucket=ohi-tfstate-${{ steps.acct.outputs.id }}"

      - run: terraform fmt -check -recursive
      - run: terraform validate

      - uses: terraform-linters/setup-tflint@v4
        with: { tflint_version: "v0.53.0" }
      - run: tflint --init --config=$GITHUB_WORKSPACE/infra/terraform/.tflint.hcl
      - run: tflint --minimum-failure-severity=warning --config=$GITHUB_WORKSPACE/infra/terraform/.tflint.hcl

      - name: Checkov
        run: |
          pip install checkov==3.2.256
          checkov -d . --framework terraform --quiet --config-file $GITHUB_WORKSPACE/infra/terraform/.checkov.yml || true

      - name: Terraform plan
        id: plan
        env:
          TF_VAR_edge_secret: ${{ secrets.CF_EDGE_SECRET_PLACEHOLDER }}  # placeholder; not used for real apply
          TF_VAR_cf_account_id: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
        run: terraform plan -no-color -out=tfplan

      - name: Render plan text to file
        id: plan_text
        run: |
          terraform show -no-color tfplan | head -c 60000 > $RUNNER_TEMP/plan.txt
          echo "path=$RUNNER_TEMP/plan.txt" >> $GITHUB_OUTPUT

      - name: Comment plan on PR
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          header: plan-${{ matrix.layer }}
          path: ${{ steps.plan_text.outputs.path }}
```

Note: `marocchino/sticky-pull-request-comment@v2` supports a `path:` input that reads a file — that avoids any shell-substitution inside YAML, which YAML does not evaluate. The former `message:` form with `$(...)` would have posted literal shell syntax to the PR.

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add .github/workflows/infra-plan.yml && git commit -m "chore(ci): infra-plan.yml — matrix plan per layer on PR"
```

### Task I.2.3: `infra-apply.yml` — manual dispatch

**Files:**
- Create: `.github/workflows/infra-apply.yml`

- [ ] **Step 1: Write**

```yaml
name: Infra Apply

on:
  workflow_dispatch:
    inputs:
      layer:
        type: choice
        description: Layer to apply
        required: true
        options: [storage, secrets, cloudflare, observability]
      confirm:
        type: string
        description: 'Type "apply" to confirm'
        required: true

concurrency:
  group: infra-apply
  cancel-in-progress: false

permissions:
  id-token: write
  contents: read

jobs:
  apply:
    runs-on: ubuntu-latest
    environment: prod
    defaults:
      run: { working-directory: infra/terraform/${{ inputs.layer }} }
    steps:
      - name: Guard
        if: inputs.confirm != 'apply'
        run: |
          echo "::error::confirm input must equal 'apply'"
          exit 1

      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
          role-session-name: gha-apply-${{ inputs.layer }}

      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: "1.10.3" }

      - name: Resolve account id
        id: acct
        run: echo "id=$(aws sts get-caller-identity --query Account --output text)" >> $GITHUB_OUTPUT

      - run: |
          terraform init \
            -backend-config="bucket=ohi-tfstate-${{ steps.acct.outputs.id }}"

      - name: Plan
        env:
          TF_VAR_edge_secret: ${{ secrets.CF_EDGE_SECRET }}
          TF_VAR_cf_account_id: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
        run: terraform plan -out=tfplan

      - name: Apply
        env:
          TF_VAR_edge_secret: ${{ secrets.CF_EDGE_SECRET }}
          TF_VAR_cf_account_id: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
        run: terraform apply tfplan

      - name: Summary
        if: always()
        run: terraform output -no-color >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add .github/workflows/infra-apply.yml && git commit -m "chore(ci): infra-apply.yml — manual dispatch for non-compute layers"
```

### Task I.2.4: `release.yml` — tag → build + compute apply

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write**

```yaml
name: Release

on:
  push:
    tags: ["v*.*.*"]

concurrency:
  group: release-rollout
  cancel-in-progress: false

permissions:
  id-token: write
  contents: write  # creates/updates GitHub Release

jobs:
  test:
    uses: ./.github/workflows/test.yml

  build-and-push-image:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.tag.outputs.tag }}
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
          role-session-name: gha-release-image-${{ github.ref_name }}

      - id: tag
        run: echo "tag=${GITHUB_REF_NAME}" >> $GITHUB_OUTPUT

      - uses: aws-actions/amazon-ecr-login@v2
        id: ecr

      - uses: docker/setup-buildx-action@v3

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/lambda/Dockerfile
          push: true
          tags: |
            ${{ vars.ECR_REPOSITORY_URL }}:${{ steps.tag.outputs.tag }}
            ${{ vars.ECR_REPOSITORY_URL }}:prod
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-compute:
    needs: build-and-push-image
    runs-on: ubuntu-latest
    environment: prod
    defaults:
      run: { working-directory: infra/terraform/compute }
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
          role-session-name: gha-release-compute-${{ github.ref_name }}

      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: "1.10.3" }

      - run: |
          ACCT=$(aws sts get-caller-identity --query Account --output text)
          terraform init -backend-config="bucket=ohi-tfstate-${ACCT}"

      - name: Apply compute with image_tag
        run: |
          terraform apply -auto-approve \
            -var="image_tag=${{ needs.build-and-push-image.outputs.image_tag }}"

      - name: Health check
        run: |
          sleep 10
          code=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "X-OHI-Edge-Secret: ${{ secrets.CF_EDGE_SECRET }}" \
            https://ohi.shiftbloom.studio/health/live)
          if [[ "$code" -ne 200 && "$code" -ne 503 ]]; then
            echo "::error::Health check returned $code"
            exit 1
          fi
          echo "Health check: HTTP $code (200 or 503=bootstrapping both acceptable)"

      - name: Rollback on failure
        if: failure()
        run: |
          PREV=$(aws ecr describe-images --repository-name ohi-api \
            --query 'reverse(sort_by(imageDetails[?imageTags!=null] | [?not_null(imageTags)],&imagePushedAt))[1].imageTags[0]' \
            --output text)
          echo "::warning::Rolling back to $PREV"
          terraform apply -auto-approve -var="image_tag=${PREV}"

  notify:
    needs: deploy-compute
    runs-on: ubuntu-latest
    if: always()
    steps:
      - uses: actions/checkout@v4
      - uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ github.ref_name }}
          name: OHI ${{ github.ref_name }}
          body: |
            Automated release.

            **Deploy status:** ${{ needs.deploy-compute.result }}
            **Image tag:** `${{ github.ref_name }}`

            See spec `docs/superpowers/specs/2026-04-16-ohi-v2-infrastructure-design.md` §3.3.3.
          generate_release_notes: true
```

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add .github/workflows/release.yml && git commit -m "chore(ci): release.yml — tag v*.*.* triggers image build + compute apply + health check"
```

### Task I.2.5: `bootstrap-drift.yml` — nightly drift check

**Files:**
- Create: `.github/workflows/bootstrap-drift.yml`

- [ ] **Step 1: Write**

```yaml
name: Bootstrap Drift

on:
  schedule:
    - cron: "0 3 * * *"   # 03:00 UTC nightly
  workflow_dispatch:

permissions:
  id-token: write
  contents: read

jobs:
  drift:
    runs-on: ubuntu-latest
    defaults:
      run: { working-directory: infra/terraform/bootstrap }
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_DRIFT_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
          role-session-name: gha-bootstrap-drift

      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: "1.10.3" }

      - name: Init (bootstrap uses local state; we init with no backend)
        run: terraform init -backend=false

      - name: Plan
        id: plan
        run: |
          set +e
          terraform plan -detailed-exitcode -no-color > plan.txt 2>&1
          code=$?
          echo "exitcode=$code" >> $GITHUB_OUTPUT
          cat plan.txt
          # 0 = no change, 1 = error, 2 = changes detected
          if [[ $code -eq 2 ]]; then echo "::warning::Drift detected"; fi
          if [[ $code -eq 1 ]]; then echo "::error::Plan failed"; exit 1; fi

      - name: Fail if drift
        if: steps.plan.outputs.exitcode == '2'
        run: exit 1
```

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add .github/workflows/bootstrap-drift.yml && git commit -m "chore(ci): bootstrap-drift.yml — nightly plan against read-only role"
```

### Task I.2.6: Phase I.2 validation script

**Files:**
- Create: `scripts/infra/validate-phase-i2.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
set -euo pipefail

fail=0
ok()  { echo "  [ok]   $1"; }
bad() { echo "  [FAIL] $1" >&2; fail=$((fail+1)); }

echo "Phase I.2 validation"

for wf in test infra-plan infra-apply release bootstrap-drift; do
  if [[ -f ".github/workflows/${wf}.yml" ]]; then
    ok "${wf}.yml present"
  else
    bad "${wf}.yml missing"
  fi
done

# actionlint (if installed)
if command -v actionlint > /dev/null; then
  if actionlint .github/workflows/*.yml > /tmp/actlint.log 2>&1; then
    ok "actionlint passes"
  else
    bad "actionlint errors: $(cat /tmp/actlint.log)"
  fi
else
  echo "  [note] actionlint not installed; skipping YAML lint"
fi

if [[ $fail -eq 0 ]]; then echo "PASS"; else echo "FAIL ($fail)"; exit 1; fi
```

- [ ] **Step 2: chmod + run**

```bash
cd "$REPO" && chmod +x scripts/infra/validate-phase-i2.sh && ./scripts/infra/validate-phase-i2.sh
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add scripts/infra/validate-phase-i2.sh && git commit -m "test(infra): Phase I.2 validation script"
```

---

## Phase I.3 — PC-side Docker stack

Goal: author the Docker Compose base + override files, the env template, and the Postgres schema seed. Execution on the PC happens per the runbook.

### Task I.3.1: `docker/compose/pc-data.yml` — base, pc-prod profile

**Files:**
- Create: `docker/compose/pc-data.yml`

- [ ] **Step 1: Write**

```yaml
# Base compose for OHI v2 PC-side data stack (prod profile).
# - No host ports exposed — only the cloudflared container reaches services.
# - Services run in the `pc-prod` profile; local-dev profile is in
#   pc-data.local-dev.yml (host ports bound on 127.0.0.1, no tunnel).
#
# Start: docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d
# Stop:  docker compose -f docker/compose/pc-data.yml --profile pc-prod down

name: ohi-pc-data

services:
  neo4j:
    image: neo4j:5-community
    profiles: [pc-prod, local-dev]
    restart: unless-stopped
    environment:
      NEO4J_AUTH: ${NEO4J_AUTH}
      NEO4J_server_memory_heap_initial__size: 1G
      NEO4J_server_memory_heap_max__size: 2G
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs

  qdrant:
    image: qdrant/qdrant:v1.12.0
    profiles: [pc-prod, local-dev]
    restart: unless-stopped
    volumes:
      - qdrant-data:/qdrant/storage

  postgres:
    image: postgres:16-alpine
    profiles: [pc-prod, local-dev]
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - pg-data:/var/lib/postgresql/data
      - ./init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 5s
      retries: 10

  postgrest:
    image: postgrest/postgrest:v12.2.0
    profiles: [pc-prod, local-dev]
    restart: unless-stopped
    environment:
      PGRST_DB_URI: ${PGRST_DB_URI}
      PGRST_DB_SCHEMA: public
      PGRST_JWT_SECRET: ${PGRST_JWT_SECRET}
      PGRST_OPENAPI_SERVER_PROXY_URI: https://pg.ohi.shiftbloom.studio
    depends_on:
      postgres: { condition: service_healthy }

  redis:
    image: redis:7-alpine
    profiles: [pc-prod, local-dev]
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

  webdis:
    image: nicolas/webdis:0.1.23
    profiles: [pc-prod, local-dev]
    restart: unless-stopped
    environment:
      WEBDIS_REDIS_HOST: redis
      WEBDIS_HTTP_AUTH: ${WEBDIS_HTTP_AUTH}
    depends_on: [redis]

  cloudflared:
    image: cloudflare/cloudflared:2026.3.0
    profiles: [pc-prod]
    restart: unless-stopped
    command: tunnel --no-autoupdate run
    environment:
      TUNNEL_TOKEN: ${TUNNEL_TOKEN}
    depends_on: [neo4j, qdrant, postgrest, webdis]

volumes:
  neo4j-data:
  neo4j-logs:
  qdrant-data:
  pg-data:
  redis-data:
```

- [ ] **Step 2: Validate syntax**

```bash
cd "$REPO" && docker compose -f docker/compose/pc-data.yml --profile pc-prod config --quiet
```
Expected: no output (valid).

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add docker/compose/pc-data.yml && git commit -m "feat(infra): docker/compose/pc-data.yml — PC prod stack (no host ports, CF tunnel only)"
```

### Task I.3.2: `docker/compose/pc-data.local-dev.yml` — override

**Files:**
- Create: `docker/compose/pc-data.local-dev.yml`

- [ ] **Step 1: Write**

```yaml
# Local-dev override for pc-data.yml. Exposes host ports on 127.0.0.1 only
# and uses separate volumes so dev data never touches prod data.
#
# Start:
#   docker compose -f docker/compose/pc-data.yml \
#                  -f docker/compose/pc-data.local-dev.yml \
#                  --profile local-dev up -d

services:
  neo4j:
    ports:
      - "127.0.0.1:7474:7474"
      - "127.0.0.1:7687:7687"
    volumes:
      - neo4j-data-dev:/data
      - neo4j-logs-dev:/logs

  qdrant:
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
    volumes:
      - qdrant-data-dev:/qdrant/storage

  postgres:
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - pg-data-dev:/var/lib/postgresql/data
      - ./init:/docker-entrypoint-initdb.d:ro

  postgrest:
    ports:
      - "127.0.0.1:3000:3000"

  redis:
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis-data-dev:/data

  webdis:
    ports:
      - "127.0.0.1:7379:7379"

volumes:
  neo4j-data-dev:
  neo4j-logs-dev:
  qdrant-data-dev:
  pg-data-dev:
  redis-data-dev:
```

- [ ] **Step 2: Validate**

```bash
cd "$REPO" && docker compose -f docker/compose/pc-data.yml -f docker/compose/pc-data.local-dev.yml --profile local-dev config --quiet
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add docker/compose/pc-data.local-dev.yml && git commit -m "feat(infra): pc-data.local-dev.yml — 127.0.0.1 host ports + dev volumes"
```

### Task I.3.3: `.env.pc-data.example`

**Files:**
- Create: `docker/compose/.env.pc-data.example`

- [ ] **Step 1: Write**

```dotenv
# Copy to docker/compose/.env.pc-data and fill in real values.
# This file is gitignored; .example is committed.

# ---- Cloudflare Tunnel (pc-prod profile only) ----
# Obtain via: aws secretsmanager get-secret-value --secret-id ohi/cloudflared-tunnel-token
TUNNEL_TOKEN=

# ---- Neo4j ----
# Format: neo4j/<strong-password>. Store in AWS secret ohi/neo4j-credentials.
NEO4J_AUTH=neo4j/CHANGEME

# ---- Postgres ----
POSTGRES_USER=ohi_app
POSTGRES_PASSWORD=CHANGEME
POSTGRES_DB=ohi_prod

# ---- PostgREST ----
PGRST_DB_URI=postgres://ohi_app:CHANGEME@postgres:5432/ohi_prod
PGRST_JWT_SECRET=CHANGEME-at-least-32-chars-of-random

# ---- WebDIS (basic auth over HTTP) ----
# Format: user:password — the webdis image reads this env var directly.
WEBDIS_HTTP_AUTH=lambda:CHANGEME
```

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add docker/compose/.env.pc-data.example && git commit -m "docs(infra): pc-data .env template"
```

### Task I.3.4: Postgres schema seed

**Files:**
- Create: `docker/compose/init/01-ohi-schema.sql`

- [ ] **Step 1: Write SQL matching algorithm spec §12**

```sql
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
```

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add docker/compose/init/01-ohi-schema.sql && git commit -m "feat(infra): Postgres schema seed matching algorithm spec §12"
```

### Task I.3.5: PC compose runbook + validation script

**Files:**
- Create: `docs/runbooks/pc-compose-start.md`
- Create: `scripts/infra/validate-phase-i3.sh`

- [ ] **Step 1: Write runbook**

```markdown
# PC Compose Start / Stop Runbook

## Prereqs

- Docker Desktop (Windows) or Docker Engine (Linux) installed.
- Cloudflare Tunnel created (see `cloudflare-api-token-rotate.md` or `tunnel.tf`) —
  Terraform writes the `TUNNEL_TOKEN` value to AWS Secrets Manager entry
  `ohi/cloudflared-tunnel-token` on first `cloudflare/` layer apply.

## First-time setup

1. Copy env template:
   ```bash
   cp docker/compose/.env.pc-data.example docker/compose/.env.pc-data
   ```
2. Fill all values in `.env.pc-data`. Sources:
   - `TUNNEL_TOKEN` → `aws secretsmanager get-secret-value --secret-id ohi/cloudflared-tunnel-token --query SecretString --output text`
   - `NEO4J_AUTH` → pick strong password, store in `ohi/neo4j-credentials`
   - Postgres + PostgREST JWT → pick strong passwords, store in `ohi/pc-origin-credentials`
   - `WEBDIS_HTTP_AUTH` → pick strong pass, store in `ohi/pc-origin-credentials`
3. Start the stack:
   ```bash
   docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d
   ```
4. Verify:
   ```bash
   docker compose -f docker/compose/pc-data.yml ps
   docker compose -f docker/compose/pc-data.yml logs cloudflared | grep "Connection established"
   ```
   You should see the tunnel come up and 4 ingress rules registered.

## Day-to-day

Start: `docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d`
Stop:  `docker compose -f docker/compose/pc-data.yml --profile pc-prod down`
Logs:  `docker compose -f docker/compose/pc-data.yml logs -f cloudflared`

## Local-dev mode (for feature work, NOT prod)

```bash
docker compose -f docker/compose/pc-data.yml \
               -f docker/compose/pc-data.local-dev.yml \
               --profile local-dev up -d
```
- Host ports bound on 127.0.0.1 only.
- `cloudflared` NOT started.
- Separate volumes (`*-dev`) — dev data doesn't touch prod data.

Stop prod before starting local-dev (volumes are different but the container
names could collide if you forgot `--profile`).

## When to restart

- PC reboot: `cloudflared` auto-restarts via `restart: unless-stopped`.
- After rotating `TUNNEL_TOKEN`: edit `.env.pc-data`, then
  `docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d --force-recreate cloudflared`.
- After Neo4j password rotation: `up -d --force-recreate neo4j` and bolt clients
  will need to re-auth.
```

- [ ] **Step 2: Write `validate-phase-i3.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

fail=0
ok()  { echo "  [ok]   $1"; }
bad() { echo "  [FAIL] $1" >&2; fail=$((fail+1)); }

echo "Phase I.3 validation"

[[ -f "docker/compose/pc-data.yml" ]] && ok "pc-data.yml present" || bad "missing"
[[ -f "docker/compose/pc-data.local-dev.yml" ]] && ok "pc-data.local-dev.yml present" || bad "missing"
[[ -f "docker/compose/.env.pc-data.example" ]] && ok "env template present" || bad "missing"
[[ -f "docker/compose/init/01-ohi-schema.sql" ]] && ok "schema seed present" || bad "missing"
[[ -f "docs/runbooks/pc-compose-start.md" ]] && ok "runbook present" || bad "missing"

if docker compose -f docker/compose/pc-data.yml --profile pc-prod config --quiet > /tmp/c.log 2>&1; then
  ok "pc-prod profile parses"
else
  bad "pc-prod parse failed: $(cat /tmp/c.log)"
fi

if docker compose -f docker/compose/pc-data.yml -f docker/compose/pc-data.local-dev.yml --profile local-dev config --quiet > /tmp/c.log 2>&1; then
  ok "local-dev profile parses"
else
  bad "local-dev parse failed: $(cat /tmp/c.log)"
fi

[[ $fail -eq 0 ]] && echo "PASS" || { echo "FAIL ($fail)"; exit 1; }
```

- [ ] **Step 3: chmod + run + commit**

```bash
cd "$REPO" && chmod +x scripts/infra/validate-phase-i3.sh && ./scripts/infra/validate-phase-i3.sh
cd "$REPO" && git add docs/runbooks/pc-compose-start.md scripts/infra/validate-phase-i3.sh && git commit -m "docs(runbook): PC compose start + Phase I.3 validation script"
```

---

## Phase I.4 — Calibration seeding workflow

Goal: publish initial calibration artifacts publicly, on a manual trigger.

### Task I.4.1: `calibration-seed.yml` workflow

**Files:**
- Create: `.github/workflows/calibration-seed.yml`

- [ ] **Step 1: Write**

```yaml
name: Calibration Seed

on:
  workflow_dispatch:
    inputs:
      domain:
        type: choice
        description: HuggingFace domain to seed
        required: true
        default: all
        options: [all, fever, anli, scifact, pubmedqa-nli, multifc, liar, climatefever, covid-fact]

concurrency:
  group: calibration-seed
  cancel-in-progress: false

permissions:
  id-token: write
  contents: write

jobs:
  seed:
    runs-on: ubuntu-latest
    environment: prod
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
          role-session-name: gha-calibration-seed

      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }

      - run: pip install -e src/api[dev]

      - name: Run synthesis
        env:
          OHI_API_BASE: https://ohi.shiftbloom.studio
          OHI_INTERNAL_BEARER: ${{ secrets.OHI_INTERNAL_BEARER }}
          OHI_CF_EDGE_SECRET: ${{ secrets.CF_EDGE_SECRET }}
          SEED_DOMAIN: ${{ inputs.domain }}
        run: |
          # This script lives in the algorithm repo (Task 2.9).
          # Expected interface:
          #   python -m scripts.calibration.synthesize_phase2_calibration --domain $SEED_DOMAIN --out /tmp/calibration_${SEED_DOMAIN}.json
          python -m scripts.calibration.synthesize_phase2_calibration \
            --domain "${SEED_DOMAIN}" \
            --out "/tmp/calibration_${SEED_DOMAIN}.json"

      - name: Upload to public bucket
        run: |
          ACCT=$(aws sts get-caller-identity --query Account --output text)
          BUCKET="ohi-artifacts-public-${ACCT}"
          DATE=$(date -u +%Y-%m-%d)
          aws s3 cp "/tmp/calibration_${SEED_DOMAIN}.json" \
            "s3://${BUCKET}/calibration/${DATE}/calibration_${SEED_DOMAIN}.json" \
            --content-type application/json \
            --cache-control "max-age=86400, public"
          echo "CALIBRATION_URL=https://${BUCKET}.s3.${{ vars.AWS_REGION }}.amazonaws.com/calibration/${DATE}/calibration_${SEED_DOMAIN}.json" >> $GITHUB_ENV

      - name: Attach to latest release
        if: success()
        run: |
          gh release upload "$(gh release view --json tagName --jq .tagName)" \
            "/tmp/calibration_${SEED_DOMAIN}.json" --clobber || echo "::warning::No release to attach to"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Summary
        run: |
          echo "### Calibration seeded" >> $GITHUB_STEP_SUMMARY
          echo "- Domain: \`${SEED_DOMAIN}\`" >> $GITHUB_STEP_SUMMARY
          echo "- URL: ${CALIBRATION_URL}" >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 2: Commit**

```bash
cd "$REPO" && git add .github/workflows/calibration-seed.yml && git commit -m "chore(ci): calibration-seed.yml — manual publish to public S3 + GH Release"
```

### Task I.4.2: Phase I.4 validation script

**Files:**
- Create: `scripts/infra/validate-phase-i4.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
set -euo pipefail

fail=0
ok()  { echo "  [ok]   $1"; }
bad() { echo "  [FAIL] $1" >&2; fail=$((fail+1)); }

echo "Phase I.4 validation"

[[ -f ".github/workflows/calibration-seed.yml" ]] && ok "calibration-seed.yml present" || bad "missing"

# Verify storage layer declares the public bucket
if grep -q "artifacts_public" infra/terraform/storage/main.tf; then
  ok "public artifacts bucket declared in storage/"
else
  bad "public bucket missing from storage/"
fi

[[ $fail -eq 0 ]] && echo "PASS" || { echo "FAIL ($fail)"; exit 1; }
```

- [ ] **Step 2: chmod + run + commit**

```bash
cd "$REPO" && chmod +x scripts/infra/validate-phase-i4.sh && ./scripts/infra/validate-phase-i4.sh
cd "$REPO" && git add scripts/infra/validate-phase-i4.sh && git commit -m "test(infra): Phase I.4 validation script"
```

---

## Phase I.5 — Runbooks + final validation

Goal: write remaining runbooks, author the alarm-test script, and the final validation pass.

### Task I.5.1: Secret rotation runbooks (batch)

**Files:**
- Create: `docs/runbooks/rotate-secret.md`
- Create: `docs/runbooks/rotate-edge-secret.md`
- Create: `docs/runbooks/cloudflare-api-token-rotate.md`
- Create: `docs/runbooks/google-cloud-quota-cap.md`

- [ ] **Step 1: `rotate-secret.md`**

```markdown
# Rotate a Secret (generic procedure)

Applies to: gemini-api-key, internal-bearer-token, labeler-tokens,
pc-origin-credentials, neo4j-credentials.

## Generic steps

1. Generate a new value:
   ```bash
   openssl rand -base64 64   # for opaque tokens
   ```
   For Gemini: generate in Google Cloud Console → APIs → Credentials.

2. Update AWS Secrets Manager:
   ```bash
   aws secretsmanager put-secret-value \
     --secret-id ohi/<secret-name> \
     --secret-string "<new-value>"
   ```

3. Lambda SecretsLoader TTL is 10 min; within 10 min every Lambda cold start
   picks up the new value. Force immediate pickup:
   ```bash
   aws lambda update-function-configuration --function-name ohi-api --description "rotate-$(date -u +%s)"
   ```
   (An arbitrary config change forces a new version → cold start → fresh read.)

4. Update any external consumers (MCP server, CI), then revoke the old value.

## BACKUP BEFORE ROTATING

`recovery_window_in_days = 0` on all secrets (§4.2 of spec). `put-secret-value`
REPLACES the value — there is NO recover flow. Always save the old value to
your password manager before `put`.
```

- [ ] **Step 2: `rotate-edge-secret.md`**

```markdown
# Rotate the CF Edge Secret

The edge secret is enforced on BOTH sides: AWS Secrets Manager (`ohi/cf-edge-secret`)
for Lambda middleware, and Cloudflare Transform Rule for header injection.

## Procedure

1. Generate a new value:
   ```bash
   openssl rand -hex 32 > /tmp/new-edge-secret.txt
   ```

2. Update Secrets Manager:
   ```bash
   aws secretsmanager put-secret-value --secret-id ohi/cf-edge-secret \
     --secret-string "$(cat /tmp/new-edge-secret.txt)"
   ```

3. Update Cloudflare — via `workflow_dispatch` on `infra-apply.yml` with
   layer=cloudflare, after setting the `CF_EDGE_SECRET` GH secret to the new value.

4. Wait 10 min (Lambda TTL). During the overlap, SOME requests may 403 if
   Cloudflare's new header arrives at a Lambda that still has the old cached
   value. Mitigate by doing steps 2 and 3 within ~30 seconds.

5. Verify: `curl -i https://ohi.shiftbloom.studio/health/live` should return 200.
   Direct-to-Function-URL should return 403.
```

- [ ] **Step 3: `cloudflare-api-token-rotate.md`**

```markdown
# Cloudflare API Token — Scopes + Rotation

## Required scopes (on the CF dashboard → My Profile → API Tokens → Create Token)

### Account-scoped:
- `Account:Cloudflare Tunnel:Edit`
- `Account:Access: Apps and Policies:Edit`
- `Account:Account Settings:Read`

### Zone-scoped (to `ohi.shiftbloom.studio`):
- `Zone:Read`
- `Zone:Zone Settings:Edit`
- `Zone:DNS:Edit`
- `Zone:Zone WAF:Edit`
- `Zone:Page Rules:Edit` (for Transform Rules)

## Create the token

1. Create token in CF dashboard.
2. Save the token value (shown ONCE).
3. Update GitHub repo secret `CLOUDFLARE_API_TOKEN`:
   ```bash
   gh secret set CLOUDFLARE_API_TOKEN < /tmp/token.txt
   ```
4. Verify:
   ```bash
   curl -H "Authorization: Bearer $(cat /tmp/token.txt)" \
        https://api.cloudflare.com/client/v4/user/tokens/verify
   ```
   Expect `"status":"active"`.

## Rotation

Same as creation, then revoke the old token from the CF dashboard.
Never commit the token to any file. Store exclusively in GH secrets.
```

- [ ] **Step 4: `google-cloud-quota-cap.md`**

```markdown
# Google Cloud — Gemini API Quota Cap

Backstops our Phase 1 "unlimited Gemini" posture (spec §9.1 R10).

## One-time setup

1. Google Cloud Console → IAM & Admin → Quotas & System Limits.
2. Filter: `Generative Language API` in the project that issued
   `ohi/gemini-api-key`.
3. For each of the quotas below, click "Edit quotas" and set a hard cap.
   Pick values that match the monthly budget you're willing to spend:

| Quota | Suggested cap |
|---|---|
| `Generate content API requests per day` | 5000 |
| `Input tokens per minute` | 500000 |
| `Output tokens per minute` | 50000 |

4. Click "Submit request" (Google may auto-approve; sometimes takes minutes).

## Verification

```bash
gcloud alpha services quota list \
  --service=generativelanguage.googleapis.com \
  --consumer=projects/<project-number>
```

## Operator contract

Review Google Cloud Billing weekly for the first month post-launch to spot
unexpected spend before it becomes material.
```

- [ ] **Step 5: Commit all four**

```bash
cd "$REPO" && git add docs/runbooks/rotate-secret.md docs/runbooks/rotate-edge-secret.md docs/runbooks/cloudflare-api-token-rotate.md docs/runbooks/google-cloud-quota-cap.md && git commit -m "docs(runbook): secret rotation + CF API token + Gemini quota cap"
```

### Task I.5.2: Incident response + rollback runbooks

**Files:**
- Create: `docs/runbooks/incident-response-basic.md`
- Create: `docs/runbooks/rollback-deploy.md`

- [ ] **Step 1: `incident-response-basic.md`**

```markdown
# Incident Response — Basic

## Triage flowchart

1. **User report / alarm email arrives.** Check what triggered:
   - Budget alarm → §Budget
   - Lambda 5xx rate → §Lambda errors
   - PC origin timeout → §PC unreachable
   - WAF block spike → §WAF (via CF dashboard, not SNS)

2. **Verify scope.** Curl each tier:
   ```bash
   curl -i https://ohi.shiftbloom.studio/health/live       # CF + Lambda public path
   curl -i -H "X-OHI-Edge-Secret: $(aws secretsmanager get-secret-value --secret-id ohi/cf-edge-secret --query SecretString --output text)" \
         https://<lambda-fn-url>/health/live               # Direct Lambda (should 200)
   ```

3. **Check CloudWatch dashboard:** `ohi-prod` in `eu-central-1`.

## § PC unreachable

- SSH or RDP to the PC.
- `docker compose -f docker/compose/pc-data.yml --profile pc-prod ps`
- If `cloudflared` is `Exited`: `docker compose ... up -d cloudflared` and
  check logs for auth errors (token may have been rotated without updating `.env.pc-data`).
- If ISP is down: nothing to do. Cloudflare will return 502 until it's back.

## § Lambda errors

- CloudWatch Logs: `/aws/lambda/ohi-api`, filter on `level = ERROR`.
- Top cause: missing/malformed secret (check SecretsLoader log).
- Second: Gemini API key invalid (quota or revocation).
- Rollback: see `rollback-deploy.md`.

## § Budget alarm

- Don't panic. Check Cost Explorer filter `CostCenter=ohi`.
- Most likely culprit: Lambda duration spike (cold-start run) or S3 egress
  from public calibration bucket (popular artifact).
- If runaway, STOP the Lambda:
  ```bash
  aws lambda put-function-concurrency --function-name ohi-api --reserved-concurrent-executions 0
  ```
  Reset when root cause known.

## § WAF / abuse

- Cloudflare dashboard → Security → Events. Filter to last 1h.
- Add a custom block rule if a specific IP range or pattern is hammering.
- Raise a temporary `managed_challenge` rule on `/api/v2/verify` if signal is mixed.
```

- [ ] **Step 2: `rollback-deploy.md`**

```markdown
# Rollback a deploy

If `release.yml` deploys a bad image, rollback automatically via the `if: failure()`
step. If it didn't (e.g., the image works but the behavior is wrong), manual rollback:

## Option A — redeploy prior image via infra-apply.yml

1. Find the previous semver tag:
   ```bash
   aws ecr describe-images --repository-name ohi-api \
     --query 'reverse(sort_by(imageDetails[?imageTags!=null],&imagePushedAt))[*].imageTags[0]' \
     --output table
   ```
2. Dispatch `infra-apply.yml` with:
   - `layer` = `compute` — **BUT this workflow doesn't list `compute`.** Use option B.

## Option B — workflow_dispatch on release.yml with a manual tag

release.yml triggers on tag push. To redeploy a prior version:

```bash
# Delete the bad tag locally and remotely (ONLY if you own the tag)
git tag -d v0.2.1
git push origin :refs/tags/v0.2.1   # delete remote (CAREFUL — ensure no-one else used it)

# Re-tag a known-good commit
git tag v0.2.1-rollback <commit-sha-that-was-last-known-good>
git push origin v0.2.1-rollback
```

release.yml fires on the new tag push; Lambda picks up the corresponding image.

## Option C — direct Lambda update (emergency only)

```bash
# Get the prior digest
PREV_DIGEST=$(aws ecr describe-images --repository-name ohi-api \
  --query 'reverse(sort_by(imageDetails[?imageTags!=null],&imagePushedAt))[1].imageDigest' \
  --output text)
REPO_URL=$(aws ecr describe-repositories --repository-names ohi-api --query 'repositories[0].repositoryUri' --output text)

aws lambda update-function-code \
  --function-name ohi-api \
  --image-uri "${REPO_URL}@${PREV_DIGEST}"
```

Note: this puts Terraform state out-of-sync. Run `terraform plan` on compute
afterward and re-apply with the correct `image_tag` to re-sync.
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add docs/runbooks/incident-response-basic.md docs/runbooks/rollback-deploy.md && git commit -m "docs(runbook): incident response + rollback procedures"
```

### Task I.5.3: Phase I.5 final validation script

**Files:**
- Create: `scripts/infra/validate-phase-i5.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
# Final validation — assembles all prior phases' checks + verifies runbooks present.
set -euo pipefail

fail=0
ok()  { echo "  [ok]   $1"; }
bad() { echo "  [FAIL] $1" >&2; fail=$((fail+1)); }

echo "Phase I.5 final validation"
echo "=========================="

# Prior phases pass
for phase in 0 1 2 3 4; do
  if ./scripts/infra/validate-phase-i${phase}.sh > /tmp/p${phase}.log 2>&1; then
    ok "phase ${phase} validation passes"
  else
    bad "phase ${phase} validation FAILED: $(tail -5 /tmp/p${phase}.log)"
  fi
done

# All runbooks present
runbooks=(bootstrap-cold-start rotate-secret rotate-edge-secret cloudflare-api-token-rotate google-cloud-quota-cap pc-compose-start incident-response-basic rollback-deploy)
for r in "${runbooks[@]}"; do
  [[ -f "docs/runbooks/${r}.md" ]] && ok "runbook ${r}.md" || bad "runbook ${r}.md missing"
done

# Final ruff/mypy sweep on new API code (flat layout)
pushd src/api > /dev/null
if ruff check config/secrets_loader.py config/infra_env.py server/middleware/edge_secret.py > /tmp/ruff.log 2>&1; then
  ok "ruff clean on new files"
else
  bad "ruff errors: $(cat /tmp/ruff.log)"
fi

if mypy config/secrets_loader.py config/infra_env.py server/middleware/edge_secret.py > /tmp/mypy.log 2>&1; then
  ok "mypy clean on new files"
else
  bad "mypy errors: $(cat /tmp/mypy.log)"
fi
popd > /dev/null

if [[ $fail -eq 0 ]]; then
  echo
  echo "=============================="
  echo "PHASE I.5: ALL PHASES PASS"
  echo "=============================="
  echo
  echo "Next:"
  echo "  1. Push the branch and open a PR."
  echo "  2. CI runs: test.yml, infra-plan.yml (per layer)."
  echo "  3. On merge + tag push (v0.1.0), release.yml fires."
  echo "  4. For manual bring-up on a cold AWS account:"
  echo "     follow docs/runbooks/bootstrap-cold-start.md"
  exit 0
else
  echo "FAIL ($fail issue(s))"
  exit 1
fi
```

- [ ] **Step 2: chmod + run**

```bash
cd "$REPO" && chmod +x scripts/infra/validate-phase-i5.sh && ./scripts/infra/validate-phase-i5.sh
```
Expected: all phases PASS.

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add scripts/infra/validate-phase-i5.sh && git commit -m "test(infra): Phase I.5 final cross-phase validation script"
```

---

## Closing checklist

- [ ] All tasks completed and each produced a single commit.
- [ ] `./scripts/infra/validate-phase-i5.sh` passes.
- [ ] `feat/ohi-v2-infra` branch is ready for PR review.
- [ ] Do NOT push to remote until user confirms.
- [ ] For actual production bring-up, hand over to the human operator with
      `docs/runbooks/bootstrap-cold-start.md`.

## Out of scope for this plan (explicit)

- Actually applying Terraform to AWS or creating real AWS/Cloudflare resources.
- Actually creating the Cloudflare zone or API token (runbook documents it).
- Seeding production data into Neo4j, Qdrant, Postgres.
- Frontend rewrite (separate sub-project 3).
- Anything described as "Phase 2" or "deferred" in the spec's §9.2.

**End of plan.**


