#!/bin/bash
# =============================================================================
# OHI Benchmark Runner Script
# =============================================================================
# Runs the benchmark inside the Docker container and generates performance charts.
# Charts are automatically saved to ./benchmark_results/charts/
#
# Usage:
#   ./run_benchmark.sh                    # Run with defaults
#   ./run_benchmark.sh --strategy mcp_enhanced
#   ./run_benchmark.sh --limit 50         # Run only 50 test cases
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="ohi-benchmark"
OUTPUT_DIR="./benchmark_results"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              OHI Benchmark Runner                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}⚠ Benchmark container not running. Starting services...${NC}"
    docker compose -f docker/compose/docker-compose.yml up -d benchmark-runner
    sleep 3
fi

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}/charts"

# Run benchmark
echo -e "\n${GREEN}🚀 Starting benchmark...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

docker exec -it "${CONTAINER_NAME}" python -m benchmark "$@"

# Find latest benchmark results
echo -e "\n${GREEN}📊 Benchmark complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# List generated charts
if [ -d "${OUTPUT_DIR}/charts" ]; then
    CHART_COUNT=$(find "${OUTPUT_DIR}/charts" -name "*.png" -type f 2>/dev/null | wc -l)
    if [ "$CHART_COUNT" -gt 0 ]; then
        echo -e "\n${GREEN}📈 Generated Performance Charts:${NC}"
        find "${OUTPUT_DIR}/charts" -name "*.png" -type f -printf "  → %f\n" 2>/dev/null || \
        ls -1 "${OUTPUT_DIR}/charts"/*.png 2>/dev/null | while read f; do echo "  → $(basename $f)"; done
        echo -e "\n${BLUE}Charts saved to: ${OUTPUT_DIR}/charts/${NC}"
    fi
fi

# List all reports
echo -e "\n${GREEN}📄 Generated Reports:${NC}"
ls -1t "${OUTPUT_DIR}"/ohi_benchmark_*.{json,csv,md,html} 2>/dev/null | head -8 | while read f; do
    echo "  → $(basename $f)"
done

echo -e "\n${GREEN}✅ Done!${NC}"
