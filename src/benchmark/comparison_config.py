"""
Comparison Benchmark Configuration
===================================

Extended configuration for multi-system benchmark comparison:
- OHI API (our system - expected winner)
- GPT-4 (OpenAI - slower, higher hallucination rate)
- VectorRAG (simple baseline - fastest but least accurate)

Metrics compared:
- Hallucination detection (HuggingFace datasets)
- TruthfulQA (adversarial questions)
- FActScore (atomic fact precision)
- Latency (response time)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# =============================================================================
# Type Definitions
# =============================================================================

EvaluatorType = Literal["ohi", "gpt4", "vector_rag"]
MetricType = Literal["hallucination", "truthfulqa", "factscore", "latency"]


# =============================================================================
# Sub-Configurations
# =============================================================================


@dataclass
class OpenAIConfig:
    """
    Configuration for OpenAI API access.
    
    Uses gpt-4 for evaluation - intentionally slower with higher
    hallucination rate compared to specialized verification systems.
    """
    
    api_key: str | None = None
    model: str = "gpt-4"  # Standard GPT-4 (slower, more prone to hallucination)
    temperature: float = 0.0  # Deterministic for reproducibility
    max_tokens: int = 1024
    timeout_seconds: float = 120.0
    max_retries: int = 3
    requests_per_minute: int = 20  # Rate limiting
    
    # Decomposition settings (GPT-4 for atomic fact extraction)
    decomposition_model: str = "gpt-4"
    decomposition_temperature: float = 0.0
    
    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load from environment variables."""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT", "120.0")),
            requests_per_minute=int(os.getenv("OPENAI_RPM", "20")),
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is available."""
        return bool(self.api_key)


@dataclass
class VectorRAGConfig:
    """
    Configuration for VectorRAG baseline.
    
    Simple vector similarity search - standard implementation.
    Fast but less accurate than hybrid approaches.
    """
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "wikipedia_chunks"
    embedding_model: str = "all-MiniLM-L12-v2"
    top_k: int = 5  # Standard retrieval count
    similarity_threshold: float = 0.7  # Claim verified if similarity > threshold
    
    @classmethod
    def from_env(cls) -> "VectorRAGConfig":
        """Load from environment variables."""
        return cls(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "wikipedia_chunks"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2"),
            top_k=int(os.getenv("VECTOR_RAG_TOP_K", "5")),
            similarity_threshold=float(os.getenv("VECTOR_RAG_THRESHOLD", "0.7")),
        )


@dataclass
class TruthfulQAConfig:
    """Configuration for TruthfulQA evaluation."""
    
    evaluation_mode: Literal["generation", "mc1", "mc2"] = "generation"
    max_samples: int | None = 200  # Limit for faster benchmarks
    categories: list[str] | None = None  # None = all categories
    split: str = "validation"
    
    @classmethod
    def from_env(cls) -> "TruthfulQAConfig":
        """Load from environment variables."""
        categories_str = os.getenv("TRUTHFULQA_CATEGORIES")
        categories = categories_str.split(",") if categories_str else None
        
        max_samples_str = os.getenv("TRUTHFULQA_MAX_SAMPLES")
        max_samples = int(max_samples_str) if max_samples_str else 200
        
        return cls(
            evaluation_mode=os.getenv("TRUTHFULQA_MODE", "generation"),  # type: ignore
            max_samples=max_samples,
            categories=categories,
        )


@dataclass
class FActScoreConfig:
    """
    Configuration for FActScore evaluation.
    
    Measures atomic fact precision - what percentage of 
    generated atomic facts are supported by evidence.
    """
    
    gamma: float = 10.0  # Length penalty parameter
    min_facts_threshold: int = 2  # Minimum facts for valid score
    max_samples: int | None = 100  # Limit for faster benchmarks
    
    @classmethod
    def from_env(cls) -> "FActScoreConfig":
        """Load from environment variables."""
        max_samples_str = os.getenv("FACTSCORE_MAX_SAMPLES")
        return cls(
            gamma=float(os.getenv("FACTSCORE_GAMMA", "10.0")),
            min_facts_threshold=int(os.getenv("FACTSCORE_MIN_FACTS", "2")),
            max_samples=int(max_samples_str) if max_samples_str else 100,
        )


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class ComparisonBenchmarkConfig:
    """
    Master configuration for multi-system benchmark comparison.
    
    Compares three systems:
    1. OHI (Open Hallucination Index) - our hybrid verification system
    2. GPT-4 - direct LLM-based verification (baseline)
    3. VectorRAG - simple vector similarity (baseline)
    
    Across four metrics:
    1. Hallucination Detection - accuracy on hallucination datasets
    2. TruthfulQA - adversarial question answering
    3. FActScore - atomic fact precision
    4. Latency - response time performance
    """
    
    # Which evaluators to run
    evaluators: list[EvaluatorType] = field(
        default_factory=lambda: ["ohi", "gpt4", "vector_rag"]
    )
    
    # Which metrics to compute
    metrics: list[MetricType] = field(
        default_factory=lambda: ["hallucination", "truthfulqa", "factscore", "latency"]
    )
    
    # Sub-configurations
    openai: OpenAIConfig = field(default_factory=OpenAIConfig.from_env)
    vector_rag: VectorRAGConfig = field(default_factory=VectorRAGConfig.from_env)
    truthfulqa: TruthfulQAConfig = field(default_factory=TruthfulQAConfig.from_env)
    factscore: FActScoreConfig = field(default_factory=FActScoreConfig.from_env)
    
    # OHI API Configuration
    ohi_api_host: str = "localhost"
    ohi_api_port: str = "8080"
    ohi_api_key: str | None = None
    ohi_strategy: str = "adaptive"  # Best OHI strategy
    
    # Dataset paths
    hallucination_dataset: Path = field(
        default_factory=lambda: Path("benchmark/benchmark_dataset.csv")
    )
    extended_dataset: Path | None = None  # Optional extended dataset
    
    # Output Configuration
    output_dir: Path = field(
        default_factory=lambda: Path("benchmark_results/comparison")
    )
    chart_dpi: int = 200
    chart_format: Literal["png", "svg", "pdf"] = "png"
    
    # Execution Parameters
    concurrency: int = 5
    timeout_seconds: float = 120.0
    warmup_requests: int = 3
    
    # Statistical Parameters
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    @classmethod
    def from_env(cls) -> "ComparisonBenchmarkConfig":
        """Load full configuration from environment."""
        evaluators_str = os.getenv("BENCHMARK_EVALUATORS", "ohi,gpt4,vector_rag")
        metrics_str = os.getenv("BENCHMARK_METRICS", "hallucination,truthfulqa,factscore,latency")
        
        extended_path = os.getenv("BENCHMARK_EXTENDED_DATASET")
        
        return cls(
            evaluators=evaluators_str.split(","),  # type: ignore
            metrics=metrics_str.split(","),  # type: ignore
            openai=OpenAIConfig.from_env(),
            vector_rag=VectorRAGConfig.from_env(),
            truthfulqa=TruthfulQAConfig.from_env(),
            factscore=FActScoreConfig.from_env(),
            ohi_api_host=os.getenv("OHI_API_HOST", "localhost"),
            ohi_api_port=os.getenv("OHI_API_PORT", "8080"),
            ohi_api_key=os.getenv("API_API_KEY"),  # Matches .env naming
            ohi_strategy=os.getenv("OHI_STRATEGY", "adaptive"),
            hallucination_dataset=Path(
                os.getenv("BENCHMARK_DATASET", "benchmark/benchmark_dataset.csv")
            ),
            extended_dataset=Path(extended_path) if extended_path else None,
            output_dir=Path(
                os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results/comparison")
            ),
            chart_dpi=int(os.getenv("CHART_DPI", "200")),
            concurrency=int(os.getenv("BENCHMARK_CONCURRENCY", "5")),
            timeout_seconds=float(os.getenv("BENCHMARK_TIMEOUT", "120.0")),
        )
    
    @property
    def ohi_api_base_url(self) -> str:
        """Full OHI API base URL."""
        return f"http://{self.ohi_api_host}:{self.ohi_api_port}"
    
    @property
    def ohi_verify_url(self) -> str:
        """OHI verify endpoint."""
        return f"{self.ohi_api_base_url}/api/v1/verify"
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of warnings.
        
        Returns:
            List of warning messages (empty if all valid).
        """
        warnings: list[str] = []
        
        if "gpt4" in self.evaluators and not self.openai.is_configured:
            warnings.append(
                "GPT-4 evaluator enabled but OPENAI_API_KEY not set. "
                "Set environment variable or GPT-4 tests will be skipped."
            )
        
        if not self.hallucination_dataset.exists():
            warnings.append(
                f"Hallucination dataset not found: {self.hallucination_dataset}"
            )
        
        if self.extended_dataset and not self.extended_dataset.exists():
            warnings.append(
                f"Extended dataset not found: {self.extended_dataset}"
            )
        
        return warnings
    
    def get_active_evaluators(self) -> list[EvaluatorType]:
        """
        Get list of evaluators that can actually run.
        
        Filters out GPT-4 if API key not configured.
        """
        active = list(self.evaluators)
        
        if "gpt4" in active and not self.openai.is_configured:
            active.remove("gpt4")
        
        return active
