"""Wave 3 Stream P — Gibbs MCMC sanity tests."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from adapters._pcg_graph import FactorGraph  # noqa: E402
from adapters._pcg_gibbs import max_marginal_delta, run_gibbs  # noqa: E402


def test_gibbs_empty_graph_returns_empty_marginals() -> None:
    graph = FactorGraph(
        claim_ids=[],
        unary_log_factors=np.zeros((0, 2), dtype=np.float64),
        edges=[],
        binary_log_factors=None,
    )
    result = run_gibbs(graph, samples=100, burn_in=50)
    assert result.marginals.shape == (0, 2)


def test_gibbs_marginals_match_strong_unary() -> None:
    """Single node with strong evidence — Gibbs marginal should
    converge close to the true normalised unary."""
    uid = uuid4()
    unary = np.array([[np.log(0.2), np.log(0.8)]], dtype=np.float64)
    graph = FactorGraph(
        claim_ids=[uid],
        unary_log_factors=unary,
        edges=[],
        binary_log_factors=None,
    )
    result = run_gibbs(graph, samples=2000, burn_in=500, seed=42)
    # Expected p_true = 0.8; Gibbs should get within ~0.05.
    assert abs(result.marginals[0, 1] - 0.8) < 0.05


def test_max_marginal_delta_returns_zero_on_identical() -> None:
    m = np.array([[0.3, 0.7], [0.6, 0.4]])
    assert max_marginal_delta(m, m) == 0.0


def test_max_marginal_delta_returns_max_abs_delta() -> None:
    a = np.array([[0.3, 0.7]])
    b = np.array([[0.5, 0.5]])
    assert abs(max_marginal_delta(a, b) - 0.2) < 1e-9


def test_gibbs_deterministic_seed() -> None:
    """Same seed → identical marginals. Integration tests need this."""
    uid = uuid4()
    unary = np.array([[np.log(0.4), np.log(0.6)]], dtype=np.float64)
    graph = FactorGraph(
        claim_ids=[uid],
        unary_log_factors=unary,
        edges=[],
        binary_log_factors=None,
    )
    r1 = run_gibbs(graph, samples=500, burn_in=100, seed=123)
    r2 = run_gibbs(graph, samples=500, burn_in=100, seed=123)
    assert np.allclose(r1.marginals, r2.marginals)
