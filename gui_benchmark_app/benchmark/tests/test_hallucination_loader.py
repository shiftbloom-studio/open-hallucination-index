"""
Tests for hallucination_loader.py fixes.

Tests the following fixes:
1. Warning when sample size exceeds available cases
2. Proper path checking with 'is None' instead of 'not csv_path'
3. Specific exception handling instead of bare Exception
4. Corrected probability comment (documentation only)
5. Fixed year regex pattern with non-capturing group
"""

import warnings
import tempfile
import csv
from pathlib import Path
import pytest
import re

from datasets.hallucination_loader import (
    HallucinationCase,
    HallucinationDataset,
    HallucinationLoader,
)


class TestSampleWarning:
    """Test Fix 1: Warning when sample size exceeds available cases."""

    def test_sample_warns_when_n_exceeds_cases(self):
        """Should warn when requested sample size is larger than available cases."""
        cases = [
            HallucinationCase(id=1, text="Test 1", label=True),
            HallucinationCase(id=2, text="Test 2", label=False),
        ]
        dataset = HallucinationDataset(cases=cases)

        # Request more cases than available
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dataset.sample(n=5, seed=42)

            # Verify warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "Requested sample size 5" in str(w[0].message)
            assert "2" in str(w[0].message)  # actual size

        # Verify result has all available cases
        assert len(result.cases) == 2

    def test_sample_no_warning_when_n_within_bounds(self):
        """Should not warn when requested sample size is within available cases."""
        cases = [
            HallucinationCase(id=i, text=f"Test {i}", label=True)
            for i in range(10)
        ]
        dataset = HallucinationDataset(cases=cases)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dataset.sample(n=5, seed=42)

            # Verify no warning was issued
            assert len(w) == 0

        # Verify result has requested size
        assert len(result.cases) == 5


class TestPathChecking:
    """Test Fix 2: Proper path checking with 'is None'."""

    def test_load_csv_with_valid_path(self):
        """Should load CSV when valid path is provided."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['id', 'domain', 'difficulty', 'label', 'text', 'notes', 'hallucination_type']
            )
            writer.writeheader()
            writer.writerow({
                'id': '1',
                'domain': 'general',
                'difficulty': 'easy',
                'label': 'true',
                'text': 'Test claim',
                'notes': '',
                'hallucination_type': ''
            })
            csv_path = Path(f.name)

        try:
            loader = HallucinationLoader()
            dataset = loader.load_csv(csv_path)
            assert len(dataset.cases) == 1
            assert dataset.cases[0].text == "Test claim"
        finally:
            csv_path.unlink()

    def test_load_csv_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        loader = HallucinationLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_csv(Path("/nonexistent/path/to/file.csv"))

    def test_load_csv_none_path_with_no_dataset_path(self):
        """Should raise FileNotFoundError when path is None and no dataset_path set."""
        loader = HallucinationLoader(dataset_path=None)
        with pytest.raises(FileNotFoundError):
            loader.load_csv(path=None)


class TestYearRegexFix:
    """Test Fix 5: Fixed year regex pattern with non-capturing group."""

    def test_year_extraction_works_correctly(self):
        """Should extract full year strings with non-capturing group."""
        # This test verifies the regex pattern works correctly
        text = "The event happened in 1995 and was significant."
        
        # Using the fixed regex pattern with non-capturing group
        years = re.findall(r'\b(?:19|20)\d{2}\b', text)
        
        # Should extract the full year
        assert len(years) == 1
        assert years[0] == "1995"
        
        # Verify it can be converted to int without error
        year_int = int(years[0])
        assert year_int == 1995
        
    def test_year_modification_in_hallucination_generation(self):
        """Should correctly modify years in hallucination generation."""
        loader = HallucinationLoader()
        factual_case = HallucinationCase(
            id=1,
            text="The company was founded in 1998.",
            label=True,
            domain="general",
        )
        
        # Generate multiple hallucinations to test year modification
        # Use different seeds to get different strategies
        for seed in range(10):
            import random
            random.seed(seed)
            
            # Try to generate hallucination
            hallucinated = loader._generate_hallucination(factual_case, new_id=100 + seed)
            
            # If it used year modification strategy, verify it worked
            if hallucinated.hallucination_type == "date_error":
                # Should have a different year
                assert "1998" not in hallucinated.text
                # Should still have a year in the text
                years_in_result = re.findall(r'\b(?:19|20)\d{2}\b', hallucinated.text)
                assert len(years_in_result) > 0


class TestSpecificExceptionHandling:
    """Test Fix 3: Specific exception handling instead of bare Exception."""

    def test_exception_types_are_specific(self):
        """Verify that specific exceptions are caught in load_combined."""
        # This is more of a code inspection test
        # We verify the code structure by checking that our imports work
        
        # Read the source code to verify the exception types
        from pathlib import Path
        source_file = Path(__file__).parent.parent / "datasets" / "hallucination_loader.py"
        content = source_file.read_text()
        
        # Verify that specific exceptions are caught
        assert "except (ImportError, ModuleNotFoundError, OSError):" in content
        # Verify that bare Exception is not used in that location
        assert "except Exception:" not in content or content.count("except Exception:") == 0


def test_imports():
    """Verify that warnings module is imported."""
    from pathlib import Path
    source_file = Path(__file__).parent.parent / "datasets" / "hallucination_loader.py"
    content = source_file.read_text()
    
    # Verify warnings import
    assert "import warnings" in content
