"""
Evaluators Package
==================

Provides different claim verification strategies for benchmark comparison:
- OHIEvaluator: Our hybrid verification system (expected winner)
- GPT4Evaluator: Direct LLM-based verification (baseline)
- VectorRAGEvaluator: Simple vector similarity (baseline)
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
]


def get_evaluator(name: str, config):
    """
    Factory function to create evaluator by name.
    
    Args:
        name: Evaluator name ("ohi", "gpt4", "vector_rag")
        config: ComparisonBenchmarkConfig instance
        
    Returns:
        Evaluator instance
    """
    evaluators = {
        "ohi": OHIEvaluator,
        "gpt4": GPT4Evaluator,
        "vector_rag": VectorRAGEvaluator,
    }
    
    if name not in evaluators:
        raise ValueError(f"Unknown evaluator: {name}. Available: {list(evaluators.keys())}")
    
    return evaluators[name](config)
