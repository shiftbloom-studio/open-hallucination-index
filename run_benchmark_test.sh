#!/bin/bash
# =============================================================================
# OHI Benchmark - Quick Test
# =============================================================================
# Tests all evaluators (OHI, GPT-4, VectorRAG) with minimal data
# to verify functionality before running full benchmarks.
#
# Usage: ./run_benchmark_test.sh
# =============================================================================

set -e

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Quick Functionality Test                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running inside Docker or locally
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container..."
    PYTHON_CMD="python"
else
    echo "Running via Docker exec..."
    PYTHON_CMD="docker exec ohi-benchmark python"
fi

# Test individual evaluators first
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Testing individual evaluators..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON_CMD -m benchmark.test_evaluators_quick

# Run minimal comparison benchmark
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Running minimal comparison benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON_CMD -m benchmark.comparison_benchmark \
    --evaluators ohi,vector_rag \
    --truthfulqa-max 3 \
    --factscore-max 2 \
    --output-dir /app/benchmark_results/test_run \
    --quiet

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ Quick Test Complete                                                    ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: benchmark_results/test_run/"
