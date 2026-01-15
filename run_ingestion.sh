#!/bin/bash
# =============================================================================
# OHI Ingestion Runner Script
# =============================================================================
# Runs the Wikipedia ingestion pipeline from the src/ingestion package.
#
# Usage:
#   ./run_ingestion.sh                     # Run with defaults
#   ./run_ingestion.sh --limit 10000
#   ./run_ingestion.sh --help
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INGESTION_DIR="$ROOT_DIR/src/ingestion"

if [ ! -d "$INGESTION_DIR" ]; then
  echo "❌ Ingestion directory not found: $INGESTION_DIR"
  exit 1
fi

cd "$INGESTION_DIR"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "🚀 Starting ingestion..."
python -m ingestion "$@"
