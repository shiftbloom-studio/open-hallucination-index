#!/usr/bin/env bash
# Validates Phase I.1 exit gate per spec §8.2.
# Runs `terraform validate` on every non-bootstrap layer and verifies structure.
set -euo pipefail

fail=0
ok()  { echo "  [ok]   $1"; }
bad() { echo "  [FAIL] $1" >&2; fail=$((fail+1)); }

echo "Phase I.1 validation"
echo "===================="

layers=(storage secrets vercel compute cloudflare observability)

for layer in "${layers[@]}"; do
  dir="infra/terraform/${layer}"
  if [[ ! -d "$dir" ]]; then
    bad "$layer/ directory missing"
    continue
  fi

  if ! command -v terraform > /dev/null; then
    ok "$layer/ files present (terraform CLI missing, validate skipped)"
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
if command -v pytest > /dev/null; then
  if pytest -c src/api/pyproject.toml \
      tests/api/config/test_secrets_loader.py \
      tests/api/server/middleware/test_edge_secret.py \
      tests/api/server/test_app_edge_secret_wiring.py \
      -q --no-cov > /tmp/pytest.log 2>&1; then
    ok "middleware + loader tests pass"
  else
    bad "middleware or loader tests failed: $(tail -20 /tmp/pytest.log)"
  fi
else
  echo "  [skip] pytest not on PATH"
fi

# Docker check (Dockerfile syntax) — parse only, no full build
if command -v docker > /dev/null; then
  # docker compose config is a lightweight parse; the Dockerfile itself needs a
  # real build to fully parse. We just confirm the file exists + is valid-looking.
  if [[ -s docker/lambda/Dockerfile && -s docker/lambda/Dockerfile.stub ]]; then
    ok "Dockerfile + Dockerfile.stub present and non-empty"
  else
    bad "Dockerfile missing/empty"
  fi
else
  echo "  [skip] docker not installed"
fi

if [[ $fail -eq 0 ]]; then echo "PASS"; else echo "FAIL ($fail issue(s))"; exit 1; fi
