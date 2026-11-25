"""Unit tests for benchmark evaluators."""

from __future__ import annotations

import pytest


class TestGetEvaluatorFactory:
    """Test the get_evaluator factory function."""

    def test_factory_import(self):
        """Test that factory can be imported."""
        from benchmark.evaluators import get_evaluator

        assert get_evaluator is not None

    def test_factory_ohi_local(self):
        """Test OHI-Local evaluator creation."""
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import get_evaluator

        config = ComparisonBenchmarkConfig.from_env()
        evaluator = get_evaluator("ohi_local", config)
        assert evaluator is not None
        assert evaluator.name == "OHI-Local"

    def test_factory_ohi_max(self):
        """Test OHI-Max evaluator creation."""
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import get_evaluator

        config = ComparisonBenchmarkConfig.from_env()
        evaluator = get_evaluator("ohi_max", config)
        assert evaluator is not None
        assert evaluator.name == "OHI-Max"

    def test_factory_graph_rag(self):
        """Test GraphRAG evaluator creation."""
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import get_evaluator

        config = ComparisonBenchmarkConfig.from_env()
        evaluator = get_evaluator("graph_rag", config)
        assert evaluator is not None
        assert evaluator.name == "GraphRAG"

    def test_factory_vector_rag(self):
        """Test VectorRAG evaluator creation."""
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import get_evaluator

        config = ComparisonBenchmarkConfig.from_env()
        evaluator = get_evaluator("vector_rag", config)
        assert evaluator is not None
        assert evaluator.name == "VectorRAG"

    def test_factory_unknown_raises(self):
        """Test that unknown evaluator raises error."""
        from benchmark.comparison_config import ComparisonBenchmarkConfig
        from benchmark.evaluators import get_evaluator

        config = ComparisonBenchmarkConfig.from_env()
        with pytest.raises(ValueError, match="Unknown evaluator"):
            get_evaluator("unknown_evaluator", config)


class TestOHIEvaluator:
    """Test OHI evaluator functionality."""

    def test_evaluator_import(self):
        """Test that evaluator can be imported."""
        try:
            from benchmark.evaluators.ohi_evaluator import OHIEvaluator

            assert OHIEvaluator is not None
        except ImportError:
            pytest.skip("OHI evaluator not available")

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        try:
            from benchmark.evaluators.ohi_evaluator import OHIEvaluator

            evaluator = OHIEvaluator(api_url="http://localhost:8080")
            assert evaluator is not None
        except ImportError:
            pytest.skip("OHI evaluator not available")

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, sample_verification_response):
        """Test basic evaluation."""
        try:
            from unittest.mock import AsyncMock

            from benchmark.evaluators.ohi_evaluator import OHIEvaluator

            evaluator = OHIEvaluator(api_url="http://localhost:8080")
            evaluator.client = AsyncMock()
            evaluator.client.post.return_value.json.return_value = sample_verification_response

            result = await evaluator.evaluate("Python was created in 1991")
            assert result is not None
        except ImportError:
            pytest.skip("OHI evaluator not available")


class TestGraphRAGEvaluator:
    """Test GraphRAG evaluator functionality."""

    def test_evaluator_import(self):
        """Test that evaluator can be imported."""
        try:
            from benchmark.evaluators.graph_rag_evaluator import GraphRAGEvaluator

            assert GraphRAGEvaluator is not None
        except ImportError:
            pytest.skip("GraphRAG evaluator not available")

    def test_evaluator_initialization(self):
        """Test evaluator initialization with config."""
        try:
            from benchmark.comparison_config import ComparisonBenchmarkConfig
            from benchmark.evaluators.graph_rag_evaluator import GraphRAGEvaluator

            config = ComparisonBenchmarkConfig.from_env()
            evaluator = GraphRAGEvaluator(config)
            assert evaluator is not None
            assert evaluator.name == "GraphRAG"
        except ImportError:
            pytest.skip("GraphRAG evaluator not available")


class TestVectorRAGEvaluator:
    """Test VectorRAG evaluator functionality."""

    def test_evaluator_import(self):
        """Test that evaluator can be imported."""
        try:
            from benchmark.evaluators.vector_rag_evaluator import VectorRAGEvaluator

            assert VectorRAGEvaluator is not None
        except ImportError:
            pytest.skip("VectorRAG evaluator not available")

    def test_evaluator_initialization(self):
        """Test evaluator initialization with config."""
        try:
            from benchmark.comparison_config import ComparisonBenchmarkConfig
            from benchmark.evaluators.vector_rag_evaluator import VectorRAGEvaluator

            config = ComparisonBenchmarkConfig.from_env()
            evaluator = VectorRAGEvaluator(config)
            assert evaluator is not None
            assert evaluator.name == "VectorRAG"
        except ImportError:
            pytest.skip("VectorRAG evaluator not available")


class TestMetricsCalculation:
    """Test metrics calculation functionality."""

    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        predictions = ["supported", "refuted", "supported"]
        ground_truth = ["supported", "supported", "supported"]

        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions)

        assert 0.0 <= accuracy <= 1.0
        assert accuracy == pytest.approx(2 / 3)

    def test_precision_recall(self):
        """Test precision and recall calculation."""
        # True positives: 2, False positives: 1, False negatives: 1
        true_positives = 2
        false_positives = 1
        false_negatives = 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)

    def test_f1_score(self):
        """Test F1 score calculation."""
        precision = 0.8
        recall = 0.6

        f1 = 2 * (precision * recall) / (precision + recall)

        assert f1 == pytest.approx(0.685714, rel=1e-5)


class TestBenchmarkRunner:
    """Test benchmark runner functionality."""

    def test_runner_import(self):
        """Test that runner can be imported."""
        try:
            from benchmark.runner import ComparisonBenchmarkRunner

            assert ComparisonBenchmarkRunner is not None
        except ImportError:
            pytest.skip("Comparison runner not available")

    def test_dataset_loading(self, sample_benchmark_dataset):
        """Test dataset loading."""
        assert len(sample_benchmark_dataset) == 3
        assert all("text" in item for item in sample_benchmark_dataset)
        assert all("expected_verdict" in item for item in sample_benchmark_dataset)


class TestReporters:
    """Test reporter functionality."""

    def test_console_reporter_import(self):
        """Test console reporter can be imported."""
        try:
            from benchmark.reporters.console_reporter import ConsoleReporter

            assert ConsoleReporter is not None
        except ImportError:
            pytest.skip("Console reporter not available")

    def test_json_reporter_import(self):
        """Test JSON reporter can be imported."""
        try:
            from benchmark.reporters.json_reporter import JSONReporter

            assert JSONReporter is not None
        except ImportError:
            pytest.skip("JSON reporter not available")


class TestComparisonMetrics:
    """Test comparison metrics functionality."""

    def test_metrics_import(self):
        """Test metrics can be imported."""
        try:
            from benchmark.comparison_benchmark import ComparisonReport

            assert ComparisonReport is not None
        except ImportError:
            pytest.skip("Comparison metrics not available")

    def test_statistical_tests(self):
        """Test statistical significance tests."""
        import numpy as np

        # Sample data
        scores_a = np.array([0.8, 0.85, 0.9, 0.88, 0.82])
        scores_b = np.array([0.75, 0.78, 0.80, 0.77, 0.79])

        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)

        assert mean_a > mean_b
        assert 0.0 <= mean_a <= 1.0
        assert 0.0 <= mean_b <= 1.0
