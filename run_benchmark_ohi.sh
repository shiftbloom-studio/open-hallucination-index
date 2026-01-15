#!/bin/bash
# =============================================================================
# OHI Benchmark - OHI Only
# =============================================================================
# Benchmarks only the OHI system with full metrics and chart generation.
# Use this to evaluate OHI performance without comparing to baselines.
#
# Usage: ./run_benchmark_ohi.sh
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/ohi_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - OHI System Only                                        ║"
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running OHI Benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON_CMD -m benchmark.comparison_benchmark \
    --evaluators ohi \
    --metrics hallucination,truthfulqa,factscore,latency \
    --truthfulqa-max 100 \
    --factscore-max 50 \
    --output-dir "${OUTPUT_DIR}" \
    --chart-dpi 200 \
    --concurrency 5 \
    --verbose

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark Complete                                                 ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  - comparison_report.json (raw data)"
echo "  - comparison_report.md (markdown summary)"
echo "  - charts/ (visualization PNGs)"
