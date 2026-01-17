"""
Evaluators Package
==================

Provides different claim verification strategies for benchmark comparison:
- OHIEvaluator: Our hybrid verification system (expected winner)
- GPT4Evaluator: Direct LLM-based verification (baseline)
- VectorRAGEvaluator: Vector similarity with Qdrant (baseline)
- GraphRAGEvaluator: Graph-based verification with Neo4j
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
    "GraphRAGEvaluator",
]


def get_evaluator(name: str, config):
    """
    Factory function to create evaluator by name.
    
    Args:
        name: Evaluator name (ohi, ohi_local, ohi_max, gpt4, vector_rag, graph_rag)
        config: ComparisonBenchmarkConfig instance
        
    Returns:
        Evaluator instance
        
    OHI Tiers:
        - ohi_local: Only local sources (Neo4j + Qdrant) - tier="local"
        - ohi: Default tier with MCP fallback - tier="default"
        - ohi_max: All sources for maximum coverage - tier="max"
    """
    if name == "vector_rag":
        return VectorRAGEvaluator(config)

    if name == "graph_rag":
        return GraphRAGEvaluator(config)

    if name == "ohi_local":
        return OHIEvaluator(
            config,
            name_override="OHI-Local",
            strategy_override="adaptive",
            tier="local",  # Only local sources (no MCP)
            target_sources_override=5,
        )

    if name == "ohi_max":
        return OHIEvaluator(
            config,
            name_override="OHI-Max",
            strategy_override="mcp_enhanced",
            tier="max",  # All sources including all MCP
            target_sources_override=20,
        )
    
    if name == "ohi":
        return OHIEvaluator(
            config,
            name_override="OHI",
            strategy_override="adaptive",
            tier="default",  # Local first, MCP fallback
            target_sources_override=8,
        )
    
    if name == "gpt4":
        return GPT4Evaluator(config)
    
    raise ValueError(
        f"Unknown evaluator: {name}. Available: ohi, ohi_local, ohi_max, gpt4, vector_rag, graph_rag"
    )
