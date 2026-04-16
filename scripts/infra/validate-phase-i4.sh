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
