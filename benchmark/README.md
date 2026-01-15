# OHI Benchmark Suite

Research-grade benchmark for evaluating hallucination detection performance
of the **Open Hallucination Index (OHI)** API against VectorRAG and GraphRAG systems.

## Features

### 🧪 Multi-Strategy Comparison
- **Vector Semantic**: Pure vector similarity search
- **Graph Exact**: Knowledge graph exact matching
- **Hybrid**: Graph + vector parallel
- **Cascading**: Graph first, vector fallback
- **MCP Enhanced**: Model Context Protocol-augmented verification
- **Adaptive**: Tiered retrieval with early-exit heuristics

### 📊 Research-Grade Statistical Analysis
- **Bootstrap Confidence Intervals**: For all metrics with configurable iterations
- **DeLong Test**: Statistical comparison of ROC-AUC between strategies
- **McNemar's Test**: Paired classifier comparison with continuity correction
- **Wilson Score Interval**: For proportion estimation

### 📈 Comprehensive Metrics
- **Classification**: Accuracy, Precision, Recall, F1, MCC
- **Calibration**: Brier Score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
- **Curve Analysis**: ROC-AUC, PR-AUC, optimal threshold detection
- **Latency**: P50, P90, P95, P99 percentiles with confidence intervals
- **Custom**: Hallucination Pass Rate (HPR) for AI safety evaluation

### 📝 Multi-Format Reporting
- **Console**: Rich terminal output with tables and panels
- **Markdown**: Publication-ready reports
- **JSON**: Machine-readable structured data
- **CSV**: For spreadsheet analysis

## Installation

```bash
# From the project root
pip install -e benchmark/

# Or with development dependencies
pip install -e "benchmark/[dev]"
```

## Usage

### Command Line

```bash
# Run with default settings
python -m benchmark

# Specify strategies
python -m benchmark --strategies vector_semantic,mcp_enhanced

# Custom configuration
python -m benchmark --threshold 0.7 --concurrency 3 --verbose

# Full options
python -m benchmark --help
```

### Programmatic API

```python
import asyncio
from benchmark import OHIBenchmarkRunner, get_config

async def run_benchmark():
    config = get_config().with_overrides(
        strategies=["vector_semantic", "mcp_enhanced"],
        threshold=0.5,
        concurrency=10,
    )
    
    async with OHIBenchmarkRunner(config=config) as runner:
        report = await runner.run_benchmark()
        print(f"Completed: {report.total_cases} cases")
        
asyncio.run(run_benchmark())
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OHI_API_HOST` | API host | `localhost` |
| `OHI_API_PORT` | API port | `8080` |
| `BENCHMARK_DATASET` | Path to CSV dataset | `benchmark_dataset.csv` |
| `BENCHMARK_OUTPUT_DIR` | Output directory | `benchmark_results` |
| `BENCHMARK_CONCURRENCY` | Parallel requests | `3` |
| `BENCHMARK_THRESHOLD` | Decision threshold | `0.7` |
| `BENCHMARK_WARMUP` | Warmup request count | `5` |
| `BENCHMARK_TIMEOUT` | Request timeout (seconds) | `120` |
| `BENCHMARK_BOOTSTRAP_ITERATIONS` | Bootstrap iterations | `1000` |
| `BENCHMARK_CONFIDENCE_LEVEL` | Confidence level | `0.95` |

### CLI Options

```
Options:
  -s, --strategies TEXT    Comma-separated strategies (default: vector_semantic,mcp_enhanced)
  --all-strategies         Test all available strategies
  -t, --threshold FLOAT    Decision threshold (default: 0.7)
  -c, --concurrency INT    Parallel requests (default: 3)
  -w, --warmup INT         Warmup requests (default: 5)
  --timeout FLOAT          Request timeout in seconds (default: 120)
  -d, --dataset PATH       Path to benchmark dataset CSV
  -o, --output-dir PATH    Output directory for reports
  --bootstrap INT          Bootstrap iterations (default: 1000)
  --confidence FLOAT       Confidence level (default: 0.95)
  --formats TEXT           Output formats (default: csv,json,markdown,html)
  -v, --verbose            Enable verbose logging
  --dry-run                Validate configuration only
  --version                Show version
```

## Dataset Format

The benchmark expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer | Unique case identifier |
| `text` | string | The claim text to verify |
| `label` | boolean | Ground truth label (true = factual) |
| `domain` | string | Domain category (e.g., science, medical) |
| `difficulty` | string | Difficulty level (easy, medium, hard, critical) |
| `notes` | string | Optional notes |
| `hallucination_type` | string | Optional hallucination pattern |

## Output

### Directory Structure

```
benchmark_results/
├── 20241201_120000/
│   ├── report.json
│   ├── report.md
│   ├── results.csv
│   └── console.html
```

### Report Contents

Each report includes:
- **Summary Statistics**: Overall performance metrics
- **Strategy Comparison**: Head-to-head with statistical significance
- **Stratified Analysis**: By domain and difficulty
- **Calibration Analysis**: Reliability diagrams
- **Error Analysis**: Misclassification patterns

## Module Structure

```
benchmark/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── config.py            # Configuration management
├── models.py            # Data structures
├── metrics.py           # Metrics computation
├── runner.py            # Main orchestration
├── analysis/
│   └── statistical.py   # Statistical tests
├── reporters/
│   ├── base.py          # Abstract reporter
│   ├── console.py       # Rich console output
│   ├── markdown.py      # Markdown reports
│   ├── json_reporter.py # JSON export
│   └── csv_reporter.py  # CSV export
└── pyproject.toml       # Package configuration
```

## API Reference

### Core Classes

- `OHIBenchmarkRunner`: Main benchmark orchestrator
- `BenchmarkConfig`: Configuration container
- `BenchmarkReport`: Results container

### Models

- `BenchmarkCase`: Individual test case
- `ResultMetric`: Single verification result
- `StrategyReport`: Per-strategy results

### Metrics

- `ConfusionMatrix`: Classification metrics with MCC
- `CalibrationMetrics`: Brier, ECE, MCE
- `ROCAnalysis`: ROC curve with AUC
- `LatencyStats`: Timing statistics

### Statistical Functions

- `mcnemar_test()`: Paired classifier comparison
- `delong_test()`: AUC comparison
- `bootstrap_ci()`: Bootstrap confidence intervals
- `wilson_ci()`: Wilson score interval

## License

MIT License - see [LICENSE](../LICENSE) for details.
