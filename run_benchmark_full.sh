#!/bin/bash
# =============================================================================
# OHI Benchmark - Full Comparison
# =============================================================================
# Complete benchmark comparing all systems:
#   - OHI (Open Hallucination Index) - Our hybrid verification system
#   - GPT-4 (OpenAI) - Direct LLM verification baseline
#   - VectorRAG - Simple vector similarity baseline (using public Wikipedia API)
#
# NOTE: GPT-4 requires OPENAI_API_KEY with sufficient quota.
#       VectorRAG uses public Wikipedia API for fair comparison.
#
# Usage: ./run_benchmark_full.sh
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/full_comparison_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Full System Comparison                                 ║"
echo "║                                                                        ║"
echo "║ Comparing: OHI vs GPT-4 vs VectorRAG                                   ║"
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

# Check GPT-4 availability
echo "Checking evaluator availability..."
$PYTHON_CMD -c "
from benchmark.comparison_config import ComparisonBenchmarkConfig
config = ComparisonBenchmarkConfig.from_env()

print('  OHI: ✓ Ready')

if config.openai.is_configured:
    print('  GPT-4: ✓ API key configured')
else:
    print('  GPT-4: ⚠ No API key (will be skipped)')

print('  VectorRAG: ✓ Using Wikipedia API (fair comparison)')
"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Full Benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Try with all evaluators, but gracefully handle GPT-4 absence
$PYTHON_CMD -m benchmark.comparison_benchmark \
    --evaluators ohi,gpt4,vector_rag \
    --metrics hallucination,truthfulqa,factscore,latency \
    --truthfulqa-max 200 \
    --factscore-max 100 \
    --output-dir "${OUTPUT_DIR}" \
    --chart-dpi 200 \
    --concurrency 5 \
    --verbose

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ Full Benchmark Complete                                                ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - comparison_report.json    Raw benchmark data"
echo "  - comparison_report.md      Markdown summary with tables"
echo "  - charts/                   Visualization charts"
echo ""
echo "Metrics compared:"
echo "  - Hallucination Detection Rate"
echo "  - TruthfulQA Accuracy"
echo "  - FActScore (atomic fact precision)"
echo "  - Latency (P50, P90, P99)"
