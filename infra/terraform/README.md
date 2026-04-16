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
