#!/bin/bash
# =============================================================================
# OHI Benchmark - Quick Evaluator Test
# =============================================================================
# Tests all evaluators: OHI-Local, OHI-Max, GraphRAG, VectorRAG
# Runs minimal samples (3 claims each) to verify system health.
#
# Usage: ./run_benchmark_test.sh
# =============================================================================

set -e

CONTAINER="ohi-benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/test_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Evaluator Quick Test                                   ║"
echo "║ Testing: OHI-Local, OHI-Max, GraphRAG, VectorRAG                       ║"
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
