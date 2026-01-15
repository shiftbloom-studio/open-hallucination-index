#!/bin/bash
# =============================================================================
# OHI Benchmark - Standard Run
# =============================================================================
# Standard benchmark comparing OHI and VectorRAG with moderate sample sizes.
# Use this for regular performance checks.
#
# Usage: ./run_benchmark.sh
# =============================================================================

set -e

CONTAINER="ohi-benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/standard_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Standard Run                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "❌ Container ${CONTAINER} is not running."
    echo "   Start with: docker compose -f docker/compose/docker-compose.yml up -d"
    exit 1
fi

echo "Output directory: ${OUTPUT_DIR}"
echo ""

docker exec ${CONTAINER} python -m benchmark.comparison_benchmark \
    --evaluators ohi,vector_rag \
    --metrics hallucination,truthfulqa,latency \
    --truthfulqa-max 50 \
    --factscore-max 20 \
    --output-dir "${OUTPUT_DIR}" \
    --chart-dpi 150 \
    --concurrency 5 \
    --verbose

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ Benchmark Complete                                                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ./benchmark_results/standard_${TIMESTAMP}"
