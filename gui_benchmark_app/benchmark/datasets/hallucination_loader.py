"""
Hallucination Dataset Loader
============================

Load hallucination detection benchmark datasets from:
- Local CSV files (benchmark_dataset.csv)
- Extended datasets from HuggingFace (aporia-ai/rag_hallucinations)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HallucinationCase:
    """
    A single hallucination detection test case.
    
    Attributes:
        id: Unique identifier
        text: The claim to verify
        label: True if factual, False if hallucination
        domain: Subject domain (general, technical, medical, etc.)
        difficulty: Difficulty level (easy, medium, hard, critical)
        notes: Additional notes about the case
        hallucination_type: Type of hallucination (for false cases)
        source: Dataset source (csv, aporia, etc.)
    """
    
    id: int
    text: str
    label: bool  # True = factual, False = hallucination
    domain: str = "general"
    difficulty: str = "medium"
    notes: str = ""
    hallucination_type: str | None = None
    source: str = "csv"
    
    @property
    def is_factual(self) -> bool:
        """Whether this case represents a factual claim."""
        return self.label
    
    @property
    def is_hallucination(self) -> bool:
        """Whether this case represents a hallucinated claim."""
        return not self.label


@dataclass
class HallucinationDataset:
    """Collection of hallucination test cases with metadata."""
    
    cases: list[HallucinationCase] = field(default_factory=list)
    source_path: str = ""
    
    @property
    def total(self) -> int:
        return len(self.cases)
    
    @property
    def factual_count(self) -> int:
        return sum(1 for c in self.cases if c.is_factual)
    
    @property
    def hallucination_count(self) -> int:
        return sum(1 for c in self.cases if c.is_hallucination)
    
    @property
    def domains(self) -> set[str]:
        return {c.domain for c in self.cases}
    
    @property
    def difficulties(self) -> set[str]:
        return {c.difficulty for c in self.cases}
    
    def filter_by_domain(self, domain: str) -> "HallucinationDataset":
        """Return subset filtered by domain."""
        filtered = [c for c in self.cases if c.domain == domain]
        return HallucinationDataset(cases=filtered, source_path=self.source_path)
    
    def filter_by_difficulty(self, difficulty: str) -> "HallucinationDataset":
        """Return subset filtered by difficulty."""
        filtered = [c for c in self.cases if c.difficulty == difficulty]
        return HallucinationDataset(cases=filtered, source_path=self.source_path)
    
    def sample(self, n: int, seed: int = 42) -> "HallucinationDataset":
        """Return random sample of n cases."""
        import random
        random.seed(seed)
        sampled = random.sample(self.cases, min(n, len(self.cases)))
        return HallucinationDataset(cases=sampled, source_path=self.source_path)


class HallucinationLoader:
    """
    Load hallucination detection datasets.
    
    Supports:
    - Local CSV files (OHI benchmark format)
    - Extended datasets from HuggingFace
    """
    
    def __init__(self, dataset_path: Path | str | None = None):
        """
        Initialize loader.
        
        Args:
            dataset_path: Path to CSV dataset file.
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
    
    def load_csv(self, path: Path | str | None = None) -> HallucinationDataset:
        """
        Load dataset from CSV file.
        
        Expected CSV format:
            id,domain,difficulty,label,text,notes,hallucination_type
        
        Args:
            path: Path to CSV file (overrides init path)
            
        Returns:
            HallucinationDataset with loaded cases
        """
        csv_path = Path(path) if path else self.dataset_path
        if not csv_path or not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        cases: list[HallucinationCase] = []
        
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse label
                label_str = str(row.get("label", "")).strip().lower()
                label = label_str in ("true", "1", "yes")
                
                case = HallucinationCase(
                    id=int(row.get("id", len(cases) + 1)),
                    text=row.get("text", "").strip(),
                    label=label,
                    domain=row.get("domain", "general").strip().lower(),
                    difficulty=row.get("difficulty", "medium").strip().lower(),
                    notes=row.get("notes", ""),
                    hallucination_type=row.get("hallucination_type") or None,
                    source="csv",
                )
                
                if case.text:  # Skip empty entries
                    cases.append(case)
        
        return HallucinationDataset(cases=cases, source_path=str(csv_path))
    
    def load_from_huggingface(
        self,
        dataset_name: str = "aporia-ai/rag_hallucinations",
        split: str = "train",
        max_samples: int | None = None,
    ) -> HallucinationDataset:
        """
        Load hallucination dataset from HuggingFace.
        
        Supports:
        - aporia-ai/rag_hallucinations (context, question, answer, hallucination)
        - SridharKumarKannam/neural-bridge-rag-hallucination
        - Jerry999/rag-hallucination  
        - muntasir2179/rag-hallucination-combined-dataset-v1
        - neural-bridge/rag-hallucination-dataset-1000
        - cemuluoglakci/hallucination_evaluation
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            max_samples: Maximum number of samples
            
        Returns:
            HallucinationDataset with loaded cases
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets library not installed. "
                "Install with: pip install datasets"
            ) from e
        
        # Try loading the dataset with different splits if the requested split doesn't exist
        try:
            dataset = load_dataset(dataset_name, split=split)
        except (ValueError, KeyError):
            # Try alternative splits
            for alt_split in ["train", "test", "validation"]:
                try:
                    dataset = load_dataset(dataset_name, split=alt_split)
                    break
                except (ValueError, KeyError):
                    continue
            else:
                # Load without specifying split
                dataset_dict = load_dataset(dataset_name)
                # Use the first available split
                dataset = dataset_dict[list(dataset_dict.keys())[0]]
        
        cases: list[HallucinationCase] = []
        
        for i, entry in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            case = self._transform_entry_to_case(entry, i + 1, dataset_name)
            if case:
                cases.append(case)
        
        return HallucinationDataset(cases=cases, source_path=dataset_name)
    
    def _transform_entry_to_case(
        self,
        entry: dict,
        case_id: int,
        dataset_name: str,
    ) -> HallucinationCase | None:
        """
        Transform a HuggingFace dataset entry to HallucinationCase.
        
        Handles multiple dataset formats with intelligent field mapping.
        """
        text = ""
        label = True  # Default to factual
        context = ""
        question = ""
        hallucination_type = None
        
        # Extract text content (try multiple field names)
        for field in ["answer", "text", "claim", "statement", "response", "output"]:
            if field in entry and entry[field]:
                text = str(entry[field]).strip()
                break
        
        # Extract context if available
        for field in ["context", "document", "passage", "source"]:
            if field in entry and entry[field]:
                context = str(entry[field]).strip()
                break
        
        # Extract question if available
        for field in ["question", "query", "input", "prompt"]:
            if field in entry and entry[field]:
                question = str(entry[field]).strip()
                break
        
        # Determine label (hallucination vs factual)
        # Try multiple label field names and formats
        if "is_hallucination" in entry:
            # Boolean: True = hallucination, so label (factual) = not is_hallucination
            label = not bool(entry["is_hallucination"])
        elif "hallucination" in entry:
            val = str(entry["hallucination"]).lower().strip()
            # Handle different string formats
            if val in ("true", "1", "yes", "hallucination"):
                label = False  # It's a hallucination
            elif val in ("false", "0", "no", "faithful", "factual"):
                label = True  # It's factual
            else:
                # Try to interpret as boolean
                label = val != "hallucination"
        elif "label" in entry:
            val = str(entry["label"]).lower().strip()
            # Different datasets use different conventions
            if val in ("hallucination", "false", "0", "no"):
                label = False
            elif val in ("factual", "true", "1", "yes", "faithful"):
                label = True
            else:
                # Try numeric interpretation
                try:
                    label = bool(int(val))
                except (ValueError, TypeError):
                    label = True
        elif "is_factual" in entry:
            label = bool(entry["is_factual"])
        elif "is_correct" in entry:
            label = bool(entry["is_correct"])
        elif "hallucinated" in entry:
            label = not bool(entry["hallucinated"])
        
        # Skip if no text content
        if not text:
            return None
        
        # Determine hallucination type if it's a hallucination
        if not label:
            hallucination_type = entry.get("hallucination_type") or entry.get("error_type") or "rag_hallucination"
        
        # Build notes with available metadata
        notes_parts = []
        if question:
            notes_parts.append(f"Q: {question[:80]}")
        if "category" in entry:
            notes_parts.append(f"Category: {entry['category']}")
        if "source" in entry and isinstance(entry["source"], str):
            notes_parts.append(f"Src: {entry['source'][:40]}")
        notes = " | ".join(notes_parts) if notes_parts else ""
        
        return HallucinationCase(
            id=case_id,
            text=text,
            label=label,
            domain=self._classify_domain(text, context),
            difficulty=self._estimate_difficulty(text),
            notes=notes,
            hallucination_type=hallucination_type,
            source=dataset_name,
        )
    
    def _classify_domain(self, text: str, context: str = "") -> str:
        """Classify text into a domain based on keywords."""
        combined = (text + " " + context).lower()
        
        domain_keywords = {
            "medical": ["patient", "disease", "treatment", "hospital", "medicine", "symptom"],
            "technical": ["software", "code", "api", "database", "algorithm", "python"],
            "science": ["research", "experiment", "physics", "chemistry", "biology"],
            "legal": ["law", "court", "legal", "regulation", "contract"],
            "finance": ["stock", "market", "investment", "bank", "economy"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in combined for kw in keywords):
                return domain
        
        return "general"
    
    def _estimate_difficulty(self, text: str) -> str:
        """Estimate difficulty based on text complexity."""
        word_count = len(text.split())
        
        if word_count <= 15:
            return "easy"
        elif word_count <= 30:
            return "medium"
        else:
            return "hard"
    
    def load_combined(
        self,
        csv_path: Path | str | None = None,
        include_huggingface: bool = True,
        hf_max_samples: int = 500,
    ) -> HallucinationDataset:
        """
        Load combined dataset from CSV and HuggingFace.
        
        Args:
            csv_path: Path to local CSV dataset
            include_huggingface: Whether to include HuggingFace datasets
            hf_max_samples: Max samples from HuggingFace
            
        Returns:
            Combined HallucinationDataset
        """
        cases: list[HallucinationCase] = []
        
        # Load local CSV
        if csv_path:
            csv_dataset = self.load_csv(csv_path)
            cases.extend(csv_dataset.cases)
        elif self.dataset_path and self.dataset_path.exists():
            csv_dataset = self.load_csv()
            cases.extend(csv_dataset.cases)
        
        # Load HuggingFace
        if include_huggingface:
            try:
                hf_dataset = self.load_from_huggingface(max_samples=hf_max_samples)
                # Renumber IDs to avoid conflicts
                max_id = max((c.id for c in cases), default=0)
                for case in hf_dataset.cases:
                    cases.append(HallucinationCase(
                        id=max_id + case.id,
                        text=case.text,
                        label=case.label,
                        domain=case.domain,
                        difficulty=case.difficulty,
                        notes=case.notes,
                        hallucination_type=case.hallucination_type,
                        source=case.source,
                    ))
            except Exception:
                pass  # HuggingFace not available, continue with CSV only
        
        return HallucinationDataset(cases=cases, source_path="combined")
    
    def load_complete_benchmark_datasets(
        self,
        csv_path: Path | str | None = None,
        samples_per_dataset: int = 200,
    ) -> HallucinationDataset:
        """
        Load comprehensive benchmark datasets for COMPLETE mode.
        
        Loads from multiple sources for research-grade evaluation:
        - Local CSV dataset
        - aporia-ai/rag_hallucinations
        - SridharKumarKannam/neural-bridge-rag-hallucination
        - Jerry999/rag-hallucination
        - muntasir2179/rag-hallucination-combined-dataset-v1
        - neural-bridge/rag-hallucination-dataset-1000
        - cemuluoglakci/hallucination_evaluation
        
        Args:
            csv_path: Path to local CSV dataset
            samples_per_dataset: Max samples per HuggingFace dataset (balanced sampling)
            
        Returns:
            Combined HallucinationDataset from all sources
        """
        all_cases: list[HallucinationCase] = []
        next_id = 1
        
        # Load local CSV first
        if csv_path:
            try:
                csv_dataset = self.load_csv(csv_path)
                all_cases.extend(csv_dataset.cases)
                next_id = len(all_cases) + 1
            except Exception as e:
                print(f"Warning: Could not load CSV dataset: {e}")
        elif self.dataset_path and self.dataset_path.exists():
            try:
                csv_dataset = self.load_csv()
                all_cases.extend(csv_dataset.cases)
                next_id = len(all_cases) + 1
            except Exception as e:
                print(f"Warning: Could not load CSV dataset: {e}")
        
        # List of HuggingFace datasets to load
        hf_datasets = [
            "aporia-ai/rag_hallucinations",
            "SridharKumarKannam/neural-bridge-rag-hallucination",
            "Jerry999/rag-hallucination",
            "muntasir2179/rag-hallucination-combined-dataset-v1",
            "neural-bridge/rag-hallucination-dataset-1000",
            "cemuluoglakci/hallucination_evaluation",
        ]
        
        for dataset_name in hf_datasets:
            try:
                print(f"Loading {dataset_name}...")
                hf_dataset = self.load_from_huggingface(
                    dataset_name=dataset_name,
                    max_samples=samples_per_dataset,
                )
                
                # Renumber IDs to avoid conflicts
                for case in hf_dataset.cases:
                    all_cases.append(HallucinationCase(
                        id=next_id,
                        text=case.text,
                        label=case.label,
                        domain=case.domain,
                        difficulty=case.difficulty,
                        notes=case.notes,
                        hallucination_type=case.hallucination_type,
                        source=case.source,
                    ))
                    next_id += 1
                
                print(f"  ✓ Loaded {len(hf_dataset.cases)} cases from {dataset_name}")
            except Exception as e:
                print(f"  ⚠ Could not load {dataset_name}: {type(e).__name__}: {e}")
        
        dataset = HallucinationDataset(cases=all_cases, source_path="complete_benchmark")
        print(f"\n✓ Total: {dataset.total} cases ({dataset.factual_count} factual, {dataset.hallucination_count} hallucinations)")
        print(f"  Domains: {', '.join(sorted(dataset.domains))}")
        
        return dataset

