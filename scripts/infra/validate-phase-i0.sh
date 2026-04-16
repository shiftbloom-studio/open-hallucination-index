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

# 3. bootstrap validates (only if terraform is available)
if command -v terraform > /dev/null; then
  pushd infra/terraform/bootstrap > /dev/null
  if terraform validate -no-color > /tmp/tfval.log 2>&1; then
    ok "bootstrap/ terraform validate passes"
  else
    # try init-without-backend first then validate
    terraform init -backend=false -input=false > /dev/null 2>&1 || true
    if terraform validate -no-color > /tmp/tfval.log 2>&1; then
      ok "bootstrap/ terraform validate passes (after init)"
    else
      bad "bootstrap/ terraform validate failed: $(cat /tmp/tfval.log)"
    fi
  fi
  popd > /dev/null
else
  echo "  [skip] terraform not installed; skipping validate"
fi

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
