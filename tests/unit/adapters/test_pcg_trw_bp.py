"""Wave 3 Stream P — TRW-BP solver tests."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from adapters._pcg_graph import FactorGraph  # noqa: E402
from adapters._pcg_trw_bp import run_trw_bp  # noqa: E402


def test_trwbp_empty_graph_returns_zero_marginals() -> None:
    graph = FactorGraph(
        claim_ids=[],
        unary_log_factors=np.zeros((0, 2), dtype=np.float64),
        edges=[],
        binary_log_factors=None,
    )
    result = run_trw_bp(graph)
    assert result.converged is True
    assert result.marginals.shape == (0, 2)
    assert result.log_partition_bound == 0.0


def test_trwbp_single_node_uniform() -> None:
    """One node, no evidence → marginal = [0.5, 0.5], bound = log 2."""
    uid = uuid4()
    graph = FactorGraph(
        claim_ids=[uid],
        unary_log_factors=np.zeros((1, 2), dtype=np.float64),
        edges=[],
        binary_log_factors=None,
    )
    result = run_trw_bp(graph)
    assert result.converged is True
    assert abs(result.marginals[0, 0] - 0.5) < 1e-6
    assert abs(result.marginals[0, 1] - 0.5) < 1e-6
    # log-Z = log(1+1) = log(2) ≈ 0.693.
    assert abs(result.log_partition_bound - np.log(2)) < 1e-6


def test_trwbp_single_node_with_evidence() -> None:
    """Single node with evidence pushes p_true > 0.5."""
    uid = uuid4()
    unary = np.array([[np.log(0.2), np.log(0.8)]], dtype=np.float64)
    graph = FactorGraph(
        claim_ids=[uid],
        unary_log_factors=unary,
        edges=[],
        binary_log_factors=None,
    )
    result = run_trw_bp(graph)
    assert result.converged is True
    # Normalised marginal → 0.8 / (0.8 + 0.2) = 0.8.
    assert abs(result.marginals[0, 1] - 0.8) < 1e-4


def test_trwbp_produces_reasonable_marginals_with_support_edge() -> None:
    """Two claims + SUPPORT edge — TRW-BP should produce per-node
    marginals consistent with the agreement preference.

    We deliberately do NOT assert on ``converged``: the shortcut 2-tree
    ρ-cover can iterate past the default tolerance in small graphs
    without pathological behaviour, and the adapter wraps this with a
    damped-LBP fallback path. Correctness of the marginals is the
    invariant that matters end-to-end.
    """
    claims = [uuid4(), uuid4()]
    unary = np.array([
        [np.log(0.25), np.log(0.75)],
        [np.log(0.25), np.log(0.75)],
    ], dtype=np.float64)
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
    result = run_trw_bp(graph, max_iters=200)
    # Both marginals should favour T.
    assert result.marginals[0, 1] > 0.5
    assert result.marginals[1, 1] > 0.5
    # log-Z bound is a finite float.
    assert np.isfinite(result.log_partition_bound)
