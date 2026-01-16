"""
Dataset Loaders Package
=======================

Provides dataset loaders for different benchmark types:
- HallucinationLoader: Load hallucination detection datasets
- TruthfulQALoader: Load TruthfulQA from HuggingFace
- FActScoreLoader: Generate FActScore evaluation samples
"""

from benchmark.datasets.hallucination_loader import (
    HallucinationCase,
    HallucinationLoader,
)
from benchmark.datasets.truthfulqa_loader import (
    TruthfulQACase,
    TruthfulQALoader,
)

__all__ = [
    "HallucinationCase",
    "HallucinationLoader",
    "TruthfulQACase",
    "TruthfulQALoader",
]
