"""Wave 3 Stream P — damped LBP solver tests."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from adapters._pcg_graph import FactorGraph  # noqa: E402
from adapters._pcg_lbp import run_damped_lbp  # noqa: E402


def test_lbp_no_edges_returns_unary_marginals() -> None:
    """A graph with no binary factors collapses to normalised unaries."""
    claims = [uuid4()]
    unary = np.array([[np.log(0.3), np.log(0.7)]], dtype=np.float64)
    graph = FactorGraph(
        claim_ids=claims,
        unary_log_factors=unary,
        edges=[],
        binary_log_factors=None,
    )
    result = run_damped_lbp(graph)
    assert result.converged is True
    assert result.iterations == 0  # no-loops shortcut
    assert abs(result.marginals[0, 1] - 0.7) < 1e-6


def test_lbp_converges_on_simple_two_node_support() -> None:
    """Two claims, one SUPPORT edge, both with evidence → both should
    converge to same-state preferred posterior."""
    claims = [uuid4(), uuid4()]
    # Both claims lean T (evidence says so).
    unary = np.array([
        [np.log(0.2), np.log(0.8)],
        [np.log(0.2), np.log(0.8)],
    ], dtype=np.float64)
    # SUPPORT edge (both-same preferred).
    binary = np.array([[
        [np.log(0.9), np.log(0.1)],
        [np.log(0.1), np.log(0.9)],
    ]], dtype=np.float64)
    graph = FactorGraph(
        claim_ids=claims,
        unary_log_factors=unary,
        edges=[(0, 1)],
        binary_log_factors=binary,
    )
    result = run_damped_lbp(graph, max_iters=200)
    assert result.converged is True
    # Both nodes should have p_true > 0.7 (amplified by agreement edge).
    assert result.marginals[0, 1] > 0.7
    assert result.marginals[1, 1] > 0.7


def test_lbp_marginals_sum_to_one() -> None:
    """Every row of the marginal matrix is a proper probability distribution."""
    claims = [uuid4(), uuid4()]
    unary = np.array([
        [np.log(0.4), np.log(0.6)],
        [np.log(0.3), np.log(0.7)],
    ], dtype=np.float64)
    binary = np.array([[
        [np.log(0.5), np.log(0.5)],
        [np.log(0.5), np.log(0.5)],
    ]], dtype=np.float64)
    graph = FactorGraph(
        claim_ids=claims,
        unary_log_factors=unary,
        edges=[(0, 1)],
        binary_log_factors=binary,
    )
    result = run_damped_lbp(graph)
    for row in result.marginals:
        assert abs(row.sum() - 1.0) < 1e-6
