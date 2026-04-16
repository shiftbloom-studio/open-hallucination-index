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
if command -v ruff > /dev/null; then
  pushd src/api > /dev/null
  if ruff check config/secrets_loader.py config/infra_env.py server/middleware/edge_secret.py > /tmp/ruff.log 2>&1; then
    ok "ruff clean on new files"
  else
    bad "ruff errors: $(cat /tmp/ruff.log)"
  fi

  if mypy --explicit-package-bases config/secrets_loader.py config/infra_env.py server/middleware/edge_secret.py > /tmp/mypy.log 2>&1; then
    ok "mypy clean on new files"
  else
    bad "mypy errors: $(cat /tmp/mypy.log)"
  fi
  popd > /dev/null
else
  echo "  [skip] ruff/mypy not on PATH"
fi

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
