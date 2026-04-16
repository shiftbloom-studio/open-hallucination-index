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
