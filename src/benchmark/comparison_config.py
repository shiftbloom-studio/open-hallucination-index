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

EvaluatorType = Literal[
    "ohi",
    "ohi_latency",
    "ohi_local",
    "ohi_max",
    "gpt4",
    "vector_rag",
    "graph_rag",
]
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
    collection_name: str = "wikipedia_hybrid"
    embedding_model: str = "all-MiniLM-L12-v2"
    top_k: int = 5  # Standard retrieval count
    similarity_threshold: float = 0.7  # Claim verified if similarity > threshold
    
    @classmethod
    def from_env(cls) -> "VectorRAGConfig":
        """Load from environment variables."""
        return cls(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "wikipedia_hybrid"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2"),
            top_k=int(os.getenv("VECTOR_RAG_TOP_K", "5")),
            similarity_threshold=float(os.getenv("VECTOR_RAG_THRESHOLD", "0.7")),
        )


@dataclass
class GraphRAGConfig:
    """
    Configuration for GraphRAG baseline.

    Uses Neo4j graph queries only (no vector search, no MCP).
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password123"
    top_k: int = 5
    min_support_ratio: float = 0.5
    min_partial_ratio: float = 0.3
    use_fulltext: bool = True
    fulltext_index: str = "article_fulltext"
    neighbor_limit: int = 6

    @classmethod
    def from_env(cls) -> "GraphRAGConfig":
        """Load from environment variables."""
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
            top_k=int(os.getenv("GRAPH_RAG_TOP_K", "5")),
            min_support_ratio=float(os.getenv("GRAPH_RAG_SUPPORT_RATIO", "0.5")),
            min_partial_ratio=float(os.getenv("GRAPH_RAG_PARTIAL_RATIO", "0.3")),
            use_fulltext=os.getenv("GRAPH_RAG_USE_FULLTEXT", "true").lower() == "true",
            fulltext_index=os.getenv("GRAPH_RAG_FULLTEXT_INDEX", "article_fulltext"),
            neighbor_limit=int(os.getenv("GRAPH_RAG_NEIGHBOR_LIMIT", "6")),
        )


@dataclass
class TruthfulQAConfig:
    """Configuration for TruthfulQA evaluation."""
    
    evaluation_mode: Literal["generation", "mc1", "mc2"] = "generation"
    max_samples: int | None = 60  # Limit for faster benchmarks
    categories: list[str] | None = None  # None = all categories
    split: str = "validation"
    
    @classmethod
    def from_env(cls) -> "TruthfulQAConfig":
        """Load from environment variables."""
        categories_str = os.getenv("TRUTHFULQA_CATEGORIES")
        categories = categories_str.split(",") if categories_str else None
        
        max_samples_str = os.getenv("TRUTHFULQA_MAX_SAMPLES")
        max_samples = int(max_samples_str) if max_samples_str else 60
        
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
    max_samples: int | None = 60  # Limit for faster benchmarks
    
    @classmethod
    def from_env(cls) -> "FActScoreConfig":
        """Load from environment variables."""
        max_samples_str = os.getenv("FACTSCORE_MAX_SAMPLES")
        return cls(
            gamma=float(os.getenv("FACTSCORE_GAMMA", "10.0")),
            min_facts_threshold=int(os.getenv("FACTSCORE_MIN_FACTS", "2")),
            max_samples=int(max_samples_str) if max_samples_str else 60,
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
    
    Special modes:
    - Strategy comparison: Run all OHI strategies to find optimal
    - Cache testing: Run with/without Redis cache to measure impact
    """
    
    # Which evaluators to run
    evaluators: list[EvaluatorType] = field(
        default_factory=lambda: ["ohi_local", "ohi_max", "graph_rag", "vector_rag", "gpt4"]
    )
    
    # Which metrics to compute
    metrics: list[MetricType] = field(
        default_factory=lambda: ["hallucination", "truthfulqa", "factscore", "latency"]
    )
    
    # Sub-configurations
    openai: OpenAIConfig = field(default_factory=OpenAIConfig.from_env)
    vector_rag: VectorRAGConfig = field(default_factory=VectorRAGConfig.from_env)
    graph_rag: GraphRAGConfig = field(default_factory=GraphRAGConfig.from_env)
    truthfulqa: TruthfulQAConfig = field(default_factory=TruthfulQAConfig.from_env)
    factscore: FActScoreConfig = field(default_factory=FActScoreConfig.from_env)
    
    # OHI API Configuration
    ohi_api_host: str = "localhost"
    ohi_api_port: str = "8080"
    ohi_api_key: str | None = None
    ohi_strategy: str = "adaptive"  # Best OHI strategy
    
    # OHI Strategy Comparison Mode
    # If True, runs all strategies for OHI to compare performance
    ohi_all_strategies: bool = False
    ohi_strategies: list[str] = field(
        default_factory=lambda: [
            "vector_semantic",
            "graph_exact",
            "hybrid",
            "cascading",
            "mcp_enhanced",
            "adaptive",
        ]
    )
    
    # Cache Testing Mode
    # If True, runs each evaluator twice: once with cache disabled, once with cache enabled (cleared first)
    cache_testing: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    
    # Dataset paths
    hallucination_dataset: Path | None = field(
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
    # Reduced concurrency to prevent connection pool exhaustion and API overload
    concurrency: int = 3  # Lower default to prevent overwhelming local API
    ohi_concurrency: int = 3  # Match main concurrency for OHI
    timeout_seconds: float = 120.0
    warmup_requests: int = 5
    hallucination_max_samples: int = 60
    
    # Statistical Parameters
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    @classmethod
    def from_env(cls) -> "ComparisonBenchmarkConfig":
        """Load full configuration from environment."""
        evaluators_str = os.getenv(
            "BENCHMARK_EVALUATORS",
            "ohi_local,ohi_max,graph_rag,vector_rag,gpt4",
        )
        metrics_str = os.getenv("BENCHMARK_METRICS", "hallucination,truthfulqa,factscore,latency")
        
        extended_path = os.getenv("BENCHMARK_EXTENDED_DATASET")
        
        # Parse OHI strategies if provided
        ohi_strategies_str = os.getenv("OHI_STRATEGIES")
        ohi_strategies = (
            ohi_strategies_str.split(",")
            if ohi_strategies_str
            else ["vector_semantic", "graph_exact", "hybrid", "cascading", "mcp_enhanced", "adaptive"]
        )
        
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
            ohi_all_strategies=os.getenv("OHI_ALL_STRATEGIES", "false").lower() == "true",
            ohi_strategies=ohi_strategies,
            cache_testing=os.getenv("BENCHMARK_CACHE_TESTING", "false").lower() == "true",
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            hallucination_dataset=(
                Path(os.getenv("BENCHMARK_DATASET"))
                if os.getenv("BENCHMARK_DATASET")
                else Path("benchmark/benchmark_dataset.csv")
            ),
            extended_dataset=Path(extended_path) if extended_path else None,
            output_dir=Path(
                os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results/comparison")
            ),
            chart_dpi=int(os.getenv("CHART_DPI", "200")),
            concurrency=int(os.getenv("BENCHMARK_CONCURRENCY", "5")),
            ohi_concurrency=int(os.getenv("OHI_CONCURRENCY", "2")),
            timeout_seconds=float(os.getenv("BENCHMARK_TIMEOUT", "120.0")),
            warmup_requests=int(os.getenv("BENCHMARK_WARMUP", "5")),
            hallucination_max_samples=int(os.getenv("BENCHMARK_HALLUCINATION_MAX", "60")),
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

        if "ohi_local" not in active and "ohi_latency" in active:
            active.remove("ohi_latency")
            active.append("ohi_local")
        
        return active
