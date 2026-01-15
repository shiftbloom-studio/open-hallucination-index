#!/bin/bash
# =============================================================================
# OHI Benchmark - Detailed Strategy Comparison
# =============================================================================
# Tests OHI with ALL verification strategies and compares them:
#   - vector_semantic   : Pure vector similarity search
#   - graph_exact       : Knowledge graph exact matching
#   - hybrid            : Graph + vector parallel
#   - cascading         : Graph first, vector fallback
#   - mcp_enhanced      : Model Context Protocol sources
#   - adaptive          : Tiered retrieval with early-exit
#
# Use this to find the optimal strategy for different use cases.
#
# Usage: ./run_benchmark_ohi_detailed.sh
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/ohi_strategies_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Strategy Comparison                                    ║"
echo "║                                                                        ║"
echo "║ Comparing all OHI verification strategies                              ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check if running inside Docker or locally
if [ -f /.dockerenv ]; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="docker exec ohi-benchmark python"
fi

# Strategies to test
STRATEGIES=("vector_semantic" "graph_exact" "hybrid" "cascading" "mcp_enhanced" "adaptive")

echo "Strategies to test:"
for strategy in "${STRATEGIES[@]}"; do
    echo "  - ${strategy}"
done
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running OHI Strategy Benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use the legacy benchmark runner which supports strategy comparison
$PYTHON_CMD -m benchmark \
    --all-strategies \
    --output-dir "${OUTPUT_DIR}" \
    --concurrency 5 \
    --bootstrap 1000 \
    --verbose

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ Strategy Comparison Complete                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Analysis includes:"
echo "  - Per-strategy accuracy metrics"
echo "  - Latency comparison (P50, P90, P99)"
echo "  - Statistical significance tests"
echo "  - Strategy recommendation"
