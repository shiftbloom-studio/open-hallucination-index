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
