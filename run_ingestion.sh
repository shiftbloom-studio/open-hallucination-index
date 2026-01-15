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

# Determine Python executable absolute path
# We use $(pwd) to get absolute path before we cd .. later
if [ -f "$(pwd)/.venv/Scripts/python.exe" ]; then
    VENV_PYTHON="$(pwd)/.venv/Scripts/python.exe"
elif [ -f "$(pwd)/.venv/bin/python" ]; then
    VENV_PYTHON="$(pwd)/.venv/bin/python"
else
    echo "⚠️  Could not find venv python, using system python"
    VENV_PYTHON="python"
fi

# Install dependencies if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo "📦 Ensuring dependencies are installed..."
    "$VENV_PYTHON" -m pip install .
fi

# Go up to 'src' directory so python can find the 'ingestion' package
cd ..

# Add current directory to PYTHONPATH explicitly
# This ensures that 'import ingestion' works even if pip install didn't fully register it globally
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "🚀 Starting ingestion..."
"$VENV_PYTHON" -m ingestion "$@"
