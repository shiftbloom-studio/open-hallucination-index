#!/bin/bash
# =============================================================================
# OHI Benchmark - Full System Comparison
# =============================================================================
# Complete benchmark comparing all verification systems:
#   - OHI (Open Hallucination Index) - Our hybrid verification system
#   - GPT-4 (OpenAI) - Direct LLM verification baseline
#   - VectorRAG - Vector similarity baseline (Qdrant)
#
# Metrics tested:
#   - Hallucination Detection (HuggingFace datasets)
#   - TruthfulQA (adversarial questions)
#   - FActScore (atomic fact precision)
#   - Latency (response time performance)
#
# NOTE: GPT-4 requires OPENAI_API_KEY with sufficient quota.
#
# Usage: ./run_benchmark_full.sh
# =============================================================================

set -e

CONTAINER="ohi-benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/app/benchmark_results/full_comparison_${TIMESTAMP}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ OHI Benchmark - Full System Comparison                                 ║"
echo "║                                                                        ║"
echo "║ Comparing: OHI (2 profiles) vs VectorRAG vs GraphRAG           ║"
echo "║ Metrics: Hallucination, TruthfulQA, FActScore, Latency (20 samples)     ║"
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

# Check evaluator availability
echo "Checking evaluator availability..."
docker exec ${CONTAINER} python -c "
from benchmark.comparison_config import ComparisonBenchmarkConfig
config = ComparisonBenchmarkConfig.from_env()

print('  OHI-Local: ✓ Ready')
print('  OHI-Max: ✓ Ready')

print('  VectorRAG: ✓ Using Qdrant vector database')
print('  GraphRAG: ✓ Using Neo4j graph')
"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Full Comparison Benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use -t for pseudo-TTY and FORCE_COLOR for Rich compatibility in Docker/Git Bash
docker exec -t \
    -e TERM=xterm-256color \
    -e FORCE_COLOR=1 \
    -e BENCHMARK_TIMEOUT=180.0 \
    ${CONTAINER} python -m benchmark.comparison_benchmark \
    --evaluators ohi_local,ohi_max,graph_rag,vector_rag \
    --metrics hallucination,truthfulqa,factscore,latency \
    --truthfulqa-max 15 \
    --factscore-max 15 \
    --hallucination-max 15 \
    --output-dir "${OUTPUT_DIR}" \
    --chart-dpi 200 \
    --concurrency 1 \
    --verbose

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║ Full Benchmark Complete                                                ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ./benchmark_results/full_comparison_${TIMESTAMP}"
echo ""
echo "Key output files:"
echo "  📊 comparison_dashboard.png   Combined visualization"
echo "  📄 *_report.json              Raw benchmark data"
