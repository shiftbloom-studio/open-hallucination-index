"""Pytest fixtures for benchmark tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_httpx_client():
    """Mock HTTPX client for API calls."""
    return MagicMock()


@pytest.fixture
def sample_verification_response():
    """Sample verification response from API."""
    return {
        "text": "Python was created in 1991",
        "verdict": "supported",
        "confidence": 0.92,
        "claims": [
            {
                "text": "Python was created in 1991",
                "verdict": "supported",
                "confidence": 0.92,
                "evidence": [
                    {
                        "text": "Python was created by Guido van Rossum in 1991",
                        "source": "wikipedia",
                        "score": 0.95,
                    }
                ],
            }
        ],
        "strategy": "adaptive",
        "processing_time_ms": 250.5,
    }


@pytest.fixture
def sample_benchmark_dataset():
    """Sample benchmark dataset."""
    return [
        {
            "text": "Python was created in 1991",
            "expected_verdict": "supported",
            "domain": "technology",
        },
        {
            "text": "The Earth is flat",
            "expected_verdict": "refuted",
            "domain": "science",
        },
        {
            "text": "The population of Tokyo is unknown",
            "expected_verdict": "uncertain",
            "domain": "geography",
        },
    ]
