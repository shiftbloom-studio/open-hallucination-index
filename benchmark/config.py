"""
Benchmark Configuration
=======================

Centralized configuration for the OHI benchmark suite.
Supports environment variables, CLI arguments, and programmatic configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# =============================================================================
# Default Values
# =============================================================================

DEFAULT_API_HOST = "localhost"
DEFAULT_API_PORT = "8080"
DEFAULT_CONCURRENCY = 3
DEFAULT_THRESHOLD = 0.7
DEFAULT_WARMUP = 5
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class BenchmarkConfig:
    """
    Immutable configuration for benchmark execution.

    Attributes:
        api_host: OHI API host.
        api_port: OHI API port.
        dataset_path: Path to the benchmark CSV dataset.
        output_dir: Directory for benchmark reports.
        strategies: List of verification strategies to test.
        threshold: Decision threshold for classification.
        concurrency: Number of parallel API requests.
        warmup_requests: Number of warmup requests before benchmarking.
        timeout_seconds: Request timeout in seconds.
        bootstrap_iterations: Number of bootstrap iterations for CI.
        confidence_level: Confidence level for intervals (0.0 to 1.0).
        verbose: Enable verbose logging.
        use_cache: Whether to use API cache during benchmark.
        target_sources: Number of evidence sources to request.
        output_formats: Report formats to generate.
    """

    # API Configuration
    api_host: str = DEFAULT_API_HOST
    api_port: str = DEFAULT_API_PORT
    api_key: str | None = None

    # Dataset
    dataset_path: Path = field(default_factory=lambda: Path("benchmark/benchmark_dataset.csv"))
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))

    # Benchmark Parameters
    strategies: list[str] = field(
        default_factory=lambda: ["vector_semantic", "mcp_enhanced"]
    )
    threshold: float = DEFAULT_THRESHOLD
    concurrency: int = DEFAULT_CONCURRENCY
    warmup_requests: int = DEFAULT_WARMUP
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    # Statistical Analysis
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL

    # Execution Options
    verbose: bool = False
    use_cache: bool = False
    target_sources: int = 10
    no_progress: bool = False

    # Output Configuration
    output_formats: list[Literal["csv", "json", "markdown", "html"]] = field(
        default_factory=lambda: ["csv", "json", "markdown", "html"]
    )

    @property
    def api_base_url(self) -> str:
        """Full API base URL."""
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def api_verify_url(self) -> str:
        """API verify endpoint URL."""
        return f"{self.api_base_url}/api/v1/verify"

    @property
    def api_batch_url(self) -> str:
        """API batch verify endpoint URL."""
        return f"{self.api_base_url}/api/v1/verify/batch"

    @property
    def api_health_url(self) -> str:
        """API health check endpoint URL."""
        return f"{self.api_base_url}/health"

    def with_overrides(self, **kwargs) -> "BenchmarkConfig":
        """Create a new config with specified overrides."""
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return BenchmarkConfig(**current)


def get_config() -> BenchmarkConfig:
    """
    Load configuration from environment variables.

    Environment Variables:
        OHI_API_HOST: API host (default: localhost)
        OHI_API_PORT: API port (default: 8080)
        BENCHMARK_DATASET: Path to CSV dataset
        BENCHMARK_OUTPUT_DIR: Output directory for reports
        BENCHMARK_CONCURRENCY: Parallel requests
        BENCHMARK_THRESHOLD: Decision threshold
        BENCHMARK_WARMUP: Warmup request count
        BENCHMARK_TIMEOUT: Request timeout in seconds
        BENCHMARK_BOOTSTRAP_ITERATIONS: Bootstrap iterations
        BENCHMARK_CONFIDENCE_LEVEL: Confidence level for CIs

    Returns:
        BenchmarkConfig with values from environment.
    """
    def _parse_bool(value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    api_key_env = os.getenv("OHI_API_KEY") or os.getenv("API_API_KEY")

    return BenchmarkConfig(
        api_host=os.getenv("OHI_API_HOST", DEFAULT_API_HOST),
        api_port=os.getenv("OHI_API_PORT", DEFAULT_API_PORT),
        api_key=api_key_env,
        dataset_path=Path(os.getenv("BENCHMARK_DATASET", "benchmark/benchmark_dataset.csv")),
        output_dir=Path(os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results")),
        concurrency=int(os.getenv("BENCHMARK_CONCURRENCY", str(DEFAULT_CONCURRENCY))),
        threshold=float(os.getenv("BENCHMARK_THRESHOLD", str(DEFAULT_THRESHOLD))),
        warmup_requests=int(os.getenv("BENCHMARK_WARMUP", str(DEFAULT_WARMUP))),
        no_progress=_parse_bool(os.getenv("BENCHMARK_NO_PROGRESS"), False),
        timeout_seconds=float(
            os.getenv("BENCHMARK_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
        ),
        bootstrap_iterations=int(
            os.getenv("BENCHMARK_BOOTSTRAP_ITERATIONS", str(DEFAULT_BOOTSTRAP_ITERATIONS))
        ),
        confidence_level=float(
            os.getenv("BENCHMARK_CONFIDENCE_LEVEL", str(DEFAULT_CONFIDENCE_LEVEL))
        ),
    )
