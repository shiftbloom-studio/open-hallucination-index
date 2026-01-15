#!/bin/bash
# =============================================================================
# OHI Benchmark - Quick Test
# =============================================================================
# Quick functionality test to verify evaluators are working.
# Runs minimal samples to check system health.
#
# Usage: ./run_benchmark_test.sh
# =============================================================================

set -e

CONTAINER="ohi-benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/test_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Quick Test                                             ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "❌ Container ${CONTAINER} is not running."
    echo "   Start with: docker compose -f docker/compose/docker-compose.yml up -d"
    exit 1
fi

echo "Running quick evaluator test..."
echo ""

docker exec ${CONTAINER} python /app/benchmark/test_evaluators_quick.py

echo ""
echo "✅ Quick test complete!"
