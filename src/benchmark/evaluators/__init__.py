"""
Evaluators Package
==================

Provides different claim verification strategies for benchmark comparison:
- OHIEvaluator: Our hybrid verification system (expected winner)
- GPT4Evaluator: Direct LLM-based verification (baseline)
- VectorRAGEvaluator: Vector similarity with Qdrant (default baseline)
- FairVectorRAGEvaluator: Vector similarity with public Wikipedia API (optional fair mode)
"""

from benchmark.evaluators.base import (
    AtomicFact,
    BaseEvaluator,
    EvaluatorResult,
    EvidenceItem,
    FActScoreResult,
    VerificationVerdict,
)
from benchmark.evaluators.gpt4_evaluator import GPT4Evaluator
from benchmark.evaluators.ohi_evaluator import OHIEvaluator
from benchmark.evaluators.graph_rag_evaluator import GraphRAGEvaluator
from benchmark.evaluators.vector_rag_evaluator import VectorRAGEvaluator
from benchmark.evaluators.fair_vector_rag_evaluator import FairVectorRAGEvaluator

__all__ = [
    # Base classes
    "BaseEvaluator",
    "EvaluatorResult",
    "EvidenceItem",
    "AtomicFact",
    "FActScoreResult",
    "VerificationVerdict",
    # Evaluator implementations
    "OHIEvaluator",
    "GPT4Evaluator",
    "VectorRAGEvaluator",
    "FairVectorRAGEvaluator",
    "GraphRAGEvaluator",
]


def get_evaluator(name: str, config, fair_mode: bool = False):
    """
    Factory function to create evaluator by name.
    
    Args:
        name: Evaluator name ("ohi", "gpt4", "vector_rag")
        config: ComparisonBenchmarkConfig instance
        fair_mode: If True, VectorRAG uses public Wikipedia API (fair mode).
               If False, uses Qdrant vector database (default).
        
    Returns:
        Evaluator instance
    """
    # For vector_rag, choose fair or unfair version
    if name == "vector_rag":
        if fair_mode:
            return FairVectorRAGEvaluator(config)
        else:
            return VectorRAGEvaluator(config)

    if name == "graph_rag":
        return GraphRAGEvaluator(config)

    if name in {"ohi_local", "ohi_latency"}:
        return OHIEvaluator(
            config,
            name_override="OHI-Local",
            strategy_override="mcp_enhanced",
            target_sources_override=6,
        )

    if name == "ohi_max":
        return OHIEvaluator(
            config,
            name_override="OHI-Max",
            strategy_override="adaptive",
            target_sources_override=18,
        )
    
    evaluators = {
        "ohi": OHIEvaluator,
        "gpt4": GPT4Evaluator,
    }
    
    if name not in evaluators:
        raise ValueError(
            f"Unknown evaluator: {name}. Available: ohi, ohi_local, ohi_latency, ohi_max, gpt4, vector_rag, graph_rag"
        )
    
    return evaluators[name](config)
