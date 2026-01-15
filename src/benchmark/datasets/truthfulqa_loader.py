"""
TruthfulQA Dataset Loader
=========================

Load TruthfulQA dataset from HuggingFace for evaluating
truthfulness in question answering.

TruthfulQA contains adversarial questions designed to
elicit false answers from language models.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TruthfulQACase:
    """
    A single TruthfulQA test case.
    
    Attributes:
        id: Unique identifier
        question: The adversarial question
        category: Question category (Law, Health, Misconceptions, etc.)
        question_type: "Adversarial" or "Non-Adversarial"
        best_answer: The single best truthful answer
        correct_answers: All truthful/correct answers
        incorrect_answers: All false/incorrect answers
        source: Source URL for the question
        
        # For multiple choice evaluation
        mc1_choices: Choices for MC1 (single correct answer)
        mc1_labels: Labels for MC1 (one 1, rest 0s)
        mc2_choices: Choices for MC2 (multiple correct possible)
        mc2_labels: Labels for MC2
    """
    
    id: int
    question: str
    category: str
    question_type: str = "Adversarial"
    best_answer: str = ""
    correct_answers: list[str] = field(default_factory=list)
    incorrect_answers: list[str] = field(default_factory=list)
    source: str = ""
    
    # Multiple choice data
    mc1_choices: list[str] = field(default_factory=list)
    mc1_labels: list[int] = field(default_factory=list)
    mc2_choices: list[str] = field(default_factory=list)
    mc2_labels: list[int] = field(default_factory=list)
    
    @property
    def has_mc_data(self) -> bool:
        """Whether MC data is available."""
        return bool(self.mc1_choices)
    
    def get_claims_for_verification(self) -> list[tuple[str, bool]]:
        """
        Generate claim/label pairs for verification testing.
        
        Returns:
            List of (claim_text, is_correct) tuples
        """
        claims: list[tuple[str, bool]] = []
        
        # Add correct answers as true claims
        for answer in self.correct_answers:
            claim = f"For the question '{self.question[:50]}...', the answer is: {answer}"
            claims.append((claim, True))
        
        # Add incorrect answers as false claims
        for answer in self.incorrect_answers[:3]:  # Limit false answers
            claim = f"For the question '{self.question[:50]}...', the answer is: {answer}"
            claims.append((claim, False))
        
        return claims


@dataclass
class TruthfulQADataset:
    """Collection of TruthfulQA test cases."""
    
    cases: list[TruthfulQACase] = field(default_factory=list)
    config: str = "generation"  # "generation" or "multiple_choice"
    
    @property
    def total(self) -> int:
        return len(self.cases)
    
    @property
    def categories(self) -> set[str]:
        return {c.category for c in self.cases}
    
    def filter_by_category(self, category: str) -> TruthfulQADataset:
        """Return subset filtered by category."""
        filtered = [c for c in self.cases if c.category == category]
        return TruthfulQADataset(cases=filtered, config=self.config)
    
    def sample(self, n: int, seed: int = 42) -> TruthfulQADataset:
        """Return random sample of n cases."""
        import random
        random.seed(seed)
        sampled = random.sample(self.cases, min(n, len(self.cases)))
        return TruthfulQADataset(cases=sampled, config=self.config)
    
    def get_all_claims(self) -> list[tuple[str, bool]]:
        """
        Get all claims from all cases for verification.
        
        Returns:
            List of (claim_text, is_correct) tuples
        """
        all_claims: list[tuple[str, bool]] = []
        for case in self.cases:
            all_claims.extend(case.get_claims_for_verification())
        return all_claims


class TruthfulQALoader:
    """
    Load TruthfulQA dataset from HuggingFace.
    
    Supports two configurations:
    - generation: For open-ended answer evaluation
    - multiple_choice: For MC1/MC2 accuracy evaluation
    """
    
    DATASET_NAME = "truthfulqa/truthful_qa"
    
    def load(
        self,
        config: str = "generation",
        split: str = "validation",
        max_samples: int | None = None,
        categories: list[str] | None = None,
    ) -> TruthfulQADataset:
        """
        Load TruthfulQA dataset.
        
        Args:
            config: "generation" or "multiple_choice"
            split: Dataset split (only "validation" available)
            max_samples: Maximum samples to load
            categories: Filter to specific categories
            
        Returns:
            TruthfulQADataset with loaded cases
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets library not installed. "
                "Install with: pip install datasets"
            ) from e
        
        dataset = load_dataset(self.DATASET_NAME, config, split=split)
        
        cases: list[TruthfulQACase] = []
        
        for i, entry in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            category = entry.get("category", "Unknown")
            
            # Filter by category if specified
            if categories and category not in categories:
                continue
            
            if config == "generation":
                case = self._parse_generation_entry(i, entry)
            else:
                case = self._parse_mc_entry(i, entry)
            
            cases.append(case)
        
        return TruthfulQADataset(cases=cases, config=config)
    
    def _parse_generation_entry(self, idx: int, entry: dict) -> TruthfulQACase:
        """Parse a generation config entry."""
        return TruthfulQACase(
            id=idx + 1,
            question=entry.get("question", ""),
            category=entry.get("category", "Unknown"),
            question_type=entry.get("type", "Adversarial"),
            best_answer=entry.get("best_answer", ""),
            correct_answers=entry.get("correct_answers", []),
            incorrect_answers=entry.get("incorrect_answers", []),
            source=entry.get("source", ""),
        )
    
    def _parse_mc_entry(self, idx: int, entry: dict) -> TruthfulQACase:
        """Parse a multiple_choice config entry."""
        mc1 = entry.get("mc1_targets", {})
        mc2 = entry.get("mc2_targets", {})
        
        return TruthfulQACase(
            id=idx + 1,
            question=entry.get("question", ""),
            category="",  # MC config doesn't have category
            mc1_choices=mc1.get("choices", []),
            mc1_labels=mc1.get("labels", []),
            mc2_choices=mc2.get("choices", []),
            mc2_labels=mc2.get("labels", []),
        )
    
    def load_for_verification(
        self,
        max_samples: int | None = 200,
        categories: list[str] | None = None,
    ) -> list[tuple[str, bool]]:
        """
        Load TruthfulQA as claim/label pairs for verification.
        
        Converts questions + answers into verifiable claims.
        
        Args:
            max_samples: Maximum questions to load
            categories: Filter to specific categories
            
        Returns:
            List of (claim_text, is_correct) tuples
        """
        dataset = self.load(
            config="generation",
            max_samples=max_samples,
            categories=categories,
        )
        
        return dataset.get_all_claims()
