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

# Check if .venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo "⚠️  No virtual environment found. Creating one..."
    # Check for python3 or python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    else
        PYTHON_CMD=python
    fi
    $PYTHON_CMD -m venv .venv
    echo "✅ Virtual environment created."
fi

if [ -d ".venv" ]; then
  if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
  elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  fi
fi

# Install dependencies if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    # Silence output unless error, but better to show it for first run
    echo "📦 Ensuring dependencies are installed..."
    pip install .
fi

# Go up to 'src' directory so python can find the 'ingestion' package
cd ..

echo "🚀 Starting ingestion..."
python -m src\\ingestion "$@"
