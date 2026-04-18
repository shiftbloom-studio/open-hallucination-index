"""Wave 3 Stream P — factor-graph construction tests."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

# Put src/api on the import path.
_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from adapters._pcg_graph import (  # noqa: E402
    FactorGraph,
    build_factor_graph,
    log_normalize_2d,
)
from models.entities import Claim, ClaimType  # noqa: E402
from models.nli import NLIDistribution  # noqa: E402


def _claim(text: str = "c") -> Claim:
    return Claim(id=uuid4(), text=text, claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT)


def _nli(entail: float, contradict: float) -> NLIDistribution:
    neutral = 1.0 - entail - contradict
    return NLIDistribution(
        entail=entail,
        contradict=contradict,
        neutral=max(neutral, 0.0),
        variance=0.0,
        nli_model_id="stub",
    )


def test_log_normalize_uniform() -> None:
    p = log_normalize_2d(np.array([0.0, 0.0]))
    assert abs(p[0] - 0.5) < 1e-9
    assert abs(p[1] - 0.5) < 1e-9


def test_log_normalize_strong_tilt() -> None:
    # [log 0.01, log 0.99] → roughly [0.01, 0.99].
    p = log_normalize_2d(np.array([np.log(0.01), np.log(0.99)]))
    assert p[1] > 0.98
    assert p[0] < 0.02


def test_build_factor_graph_no_edges_no_evidence() -> None:
    """No evidence and no cc-NLI → graph with uniform unaries, no edges."""
    claims = [_claim("a"), _claim("b")]
    graph = build_factor_graph(claims, {}, {})
    assert graph.n_nodes == 2
    assert graph.n_edges == 0
    # Uniform unary log-factors (all zero).
    assert np.allclose(graph.unary_log_factors, 0.0)


def test_build_factor_graph_unary_from_evidence_nli() -> None:
    """Evidence NLI with strong entail should push unary log-factor
    toward state T (node = [F, T])."""
    c = _claim("Marie Curie won a Nobel prize.")
    claims = [c]
    nli = {c.id: [_nli(entail=0.9, contradict=0.05)]}
    graph = build_factor_graph(claims, nli, {})
    # log φ(T) > log φ(F).
    assert graph.unary_log_factors[0, 1] > graph.unary_log_factors[0, 0]


def test_build_factor_graph_binary_edge_from_cc_nli() -> None:
    """Two claims with a SUPPORT cc-NLI edge produce one binary factor
    preferring the same-label diagonal."""
    a = _claim("Einstein won the Nobel prize in 1921.")
    b = _claim("Einstein received the Nobel for photoelectric effect.")
    # Canonical (id_a < id_b).
    if a.id > b.id:
        a, b = b, a
    cc = {(a.id, b.id): _nli(entail=0.95, contradict=0.02)}
    graph = build_factor_graph([a, b], {}, cc)
    assert graph.n_edges == 1
    assert graph.binary_log_factors is not None
    # Diagonal (both-same) should be higher than off-diagonal.
    bf = graph.binary_log_factors[0]
    assert bf[0, 0] > bf[0, 1]  # (F, F) > (F, T)
    assert bf[1, 1] > bf[1, 0]  # (T, T) > (T, F)


def test_build_factor_graph_refute_edge_prefers_off_diagonal() -> None:
    """REFUTE cc-NLI → off-diagonal preferred (claims disagree)."""
    a = _claim("A")
    b = _claim("B")
    if a.id > b.id:
        a, b = b, a
    cc = {(a.id, b.id): _nli(entail=0.02, contradict=0.95)}
    graph = build_factor_graph([a, b], {}, cc)
    bf = graph.binary_log_factors[0]
    # (F, T) and (T, F) should be higher than diagonal.
    assert bf[0, 1] > bf[0, 0]
    assert bf[1, 0] > bf[1, 1]
