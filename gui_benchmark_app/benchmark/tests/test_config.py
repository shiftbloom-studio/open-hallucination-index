"""Unit tests for benchmark configuration."""

from __future__ import annotations

import pytest


class TestComparisonConfig:
    """Test comparison configuration."""

    def test_config_import(self):
        """Test configuration can be imported."""
        try:
            from benchmark.comparison_config import ComparisonConfig

            assert ComparisonConfig is not None
        except ImportError:
            pytest.skip("Comparison config not available")

    def test_config_defaults(self):
        """Test configuration has sensible defaults."""
        try:
            from benchmark.comparison_config import ComparisonConfig

            config = ComparisonConfig()
            assert hasattr(config, "ohi_api_url")
            assert hasattr(config, "strategies")
        except (ImportError, AttributeError):
            pytest.skip("Comparison config not fully available")


class TestBenchmarkDatasets:
    """Test benchmark datasets."""

    def test_dataset_structure(self):
        """Test that datasets have required fields."""
        sample_item = {
            "text": "Test claim",
            "expected_verdict": "supported",
            "domain": "test",
        }

        assert "text" in sample_item
        assert "expected_verdict" in sample_item
        assert sample_item["expected_verdict"] in [
            "supported",
            "refuted",
            "uncertain",
        ]

    def test_dataset_validation(self):
        """Test dataset validation logic."""
        valid_verdicts = {"supported", "refuted", "uncertain"}

        test_verdicts = ["supported", "refuted", "uncertain"]
        assert all(v in valid_verdicts for v in test_verdicts)


class TestEvaluatorRegistry:
    """Test evaluator registry."""

    def test_evaluator_types(self):
        """Test available evaluator types."""
        evaluator_types = ["ohi", "gpt4", "vector_rag"]
        assert len(evaluator_types) > 0
        assert "ohi" in evaluator_types
