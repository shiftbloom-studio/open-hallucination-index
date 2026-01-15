"""
Evaluators Package
==================

Provides different claim verification strategies for benchmark comparison:
- OHIEvaluator: Our hybrid verification system (expected winner)
- GPT4Evaluator: Direct LLM-based verification (baseline)
- VectorRAGEvaluator: Vector similarity with local DB (uses our data - less fair)
- FairVectorRAGEvaluator: Vector similarity with public Wikipedia API (fair comparison)
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
]


def get_evaluator(name: str, config, fair_mode: bool = True):
    """
    Factory function to create evaluator by name.
    
    Args:
        name: Evaluator name ("ohi", "gpt4", "vector_rag")
        config: ComparisonBenchmarkConfig instance
        fair_mode: If True, VectorRAG uses public Wikipedia API (fair comparison).
                   If False, uses our local Qdrant database (unfair advantage for VectorRAG).
        
    Returns:
        Evaluator instance
    """
    # For vector_rag, choose fair or unfair version
    if name == "vector_rag":
        if fair_mode:
            return FairVectorRAGEvaluator(config)
        else:
            return VectorRAGEvaluator(config)
    
    evaluators = {
        "ohi": OHIEvaluator,
        "gpt4": GPT4Evaluator,
    }
    
    if name not in evaluators:
        raise ValueError(f"Unknown evaluator: {name}. Available: ohi, gpt4, vector_rag")
    
    return evaluators[name](config)
