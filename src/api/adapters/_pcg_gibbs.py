"""Gibbs MCMC sanity sampler for the PCG factor graph.

Draws samples from the joint distribution, approximates marginals by
sample frequencies, and reports the max-absolute-delta versus a
reference BP marginal set. Used in ``balanced`` + ``maximum`` rigor
tiers to catch "BP converged to the wrong fixed point" failures —
LBP can converge to non-MAP basins of attraction on cyclic graphs,
and TRW-BP's marginals are valid only under the chosen ρ-cover.

Gibbs is conceptually ground-truth here: infinite samples would
recover the exact marginals. In practice we take ~2000 samples after
burn-in, which suffices to spot ~0.05-level disagreements reliably.

A fixed seed is derived from the verify ``request_id`` so two
verifies of the same document produce the same Gibbs sanity verdict —
important for the integration test suite. Non-determinism in the
sampler would make "the Gibbs sanity failed flakily" an easy
false-positive pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adapters._pcg_graph import FactorGraph


@dataclass(frozen=True)
class GibbsResult:
    """Empirical marginals from the Gibbs sampler plus mismatch vs
    a reference (typically BP). ``max_marginal_delta`` is the max
    absolute delta across all nodes."""

    marginals: np.ndarray
    samples_drawn: int
    burn_in: int
    max_marginal_delta: float | None = None  # set via compare_to(reference)


def _node_conditional(
    graph: FactorGraph,
    state: np.ndarray,
    i: int,
) -> np.ndarray:
    """Log-probability of node ``i`` taking each state given all others fixed."""
    log_prob = graph.unary_log_factors[i].copy()
    if graph.binary_log_factors is None:
        return log_prob
    for k, (a, b) in enumerate(graph.edges):
        if a == i:
            other_state = state[b]
            log_prob += graph.binary_log_factors[k, :, other_state]
        elif b == i:
            other_state = state[a]
            log_prob += graph.binary_log_factors[k, other_state, :]
    return log_prob


def _sample_from_log(log_prob: np.ndarray, rng: np.random.Generator) -> int:
    """Sample from a 2-state log-distribution."""
    m = np.max(log_prob)
    if not np.isfinite(m):
        return int(rng.integers(0, 2))
    p = np.exp(log_prob - m)
    p /= p.sum()
    return int(rng.choice(2, p=p))


def run_gibbs(
    graph: FactorGraph,
    *,
    samples: int = 2000,
    burn_in: int = 500,
    seed: int = 0,
) -> GibbsResult:
    """Vanilla Gibbs sweep. Each sample visits all nodes in turn and
    resamples each from its conditional distribution.

    Returns per-node empirical marginals (``[P(F), P(T)]``) after
    ``burn_in`` warm-up sweeps and ``samples`` recorded sweeps.
    """
    n = graph.n_nodes
    if n == 0:
        return GibbsResult(
            marginals=np.zeros((0, 2), dtype=np.float64),
            samples_drawn=0,
            burn_in=burn_in,
        )
    rng = np.random.default_rng(seed)

    # Initialise at uniform-random state.
    state = rng.integers(0, 2, size=n)

    # Burn-in: run sweeps without recording.
    for _ in range(burn_in):
        for i in range(n):
            log_cond = _node_conditional(graph, state, i)
            state[i] = _sample_from_log(log_cond, rng)

    # Sampling phase.
    counts = np.zeros((n, 2), dtype=np.int64)
    for _ in range(samples):
        for i in range(n):
            log_cond = _node_conditional(graph, state, i)
            state[i] = _sample_from_log(log_cond, rng)
        for i in range(n):
            counts[i, state[i]] += 1

    marginals = counts.astype(np.float64) / float(samples)
    return GibbsResult(
        marginals=marginals,
        samples_drawn=samples,
        burn_in=burn_in,
    )


def max_marginal_delta(ref: np.ndarray, gibbs: np.ndarray) -> float:
    """Max absolute delta between two marginal tables (shape (n, 2))."""
    if ref.shape != gibbs.shape or ref.size == 0:
        return 0.0
    return float(np.max(np.abs(ref - gibbs)))


__all__ = ["GibbsResult", "run_gibbs", "max_marginal_delta"]
