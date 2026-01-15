#!/bin/bash
# =============================================================================
# OHI Benchmark - Detailed OHI Analysis
# =============================================================================
# Comprehensive OHI benchmark including:
#   1. All verification strategies comparison
#   2. Cache performance testing (cold vs warm)
#   3. Full metrics (Hallucination, TruthfulQA, FActScore, Latency)
#
# Strategies tested:
#   - vector_semantic   : Pure vector similarity search
#   - graph_exact       : Knowledge graph exact matching
#   - hybrid            : Graph + vector parallel
#   - cascading         : Graph first, vector fallback
#   - mcp_enhanced      : Model Context Protocol sources
#   - adaptive          : Tiered retrieval with early-exit
#
# Usage: ./run_benchmark_ohi_detailed.sh
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/ohi_detailed_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Detailed Analysis                                      ║"
echo "║                                                                        ║"
echo "║ Tests: All Strategies + Cache Performance                              ║"
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

mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# PHASE 1: Strategy Comparison
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 1: OHI Strategy Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON_CMD -m benchmark.comparison_benchmark \
    --evaluators ohi \
    --ohi-all-strategies \
    --ohi-strategies vector_semantic,graph_exact,hybrid,cascading,mcp_enhanced,adaptive \
    --metrics hallucination,truthfulqa,factscore,latency \
    --truthfulqa-max 100 \
    --factscore-max 50 \
    --output-dir "${OUTPUT_DIR}/strategies" \
    --chart-dpi 200 \
    --concurrency 5 \
    --verbose

# =============================================================================
# PHASE 2: Cache Performance Testing
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 2: Cache Performance Testing (Cold vs Warm)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON_CMD -m benchmark.comparison_benchmark \
    --evaluators ohi \
    --cache-testing \
    --redis-host redis \
    --metrics hallucination,latency \
    --truthfulqa-max 50 \
    --output-dir "${OUTPUT_DIR}/cache" \
    --chart-dpi 200 \
    --concurrency 5 \
    --verbose

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ Detailed OHI Analysis Complete                                         ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Analysis includes:"
echo "  📊 strategies/     - All 6 verification strategies compared"
echo "  ⚡ cache/          - Cache cold vs warm performance"
echo ""
echo "Key output files:"
echo "  - comparison_dashboard.png   Combined visualization"
echo "  - *_report.json              Raw benchmark data"
echo ""
