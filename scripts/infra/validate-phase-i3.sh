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

if command -v docker > /dev/null; then
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
else
  echo "  [skip] docker not installed"
fi

[[ $fail -eq 0 ]] && echo "PASS" || { echo "FAIL ($fail)"; exit 1; }
