"""Damped Loopy Belief Propagation on the PCG factor graph.

Murphy-Weiss-Jordan 1999 damping scheme. Runs in log-space for
numerical stability on strongly-polarised NLI edges (entail=0.98 style
cases produce log-factors near zero on one state and highly-negative on
the other). Convergence tolerance checked in the marginal domain, not
message domain — matches the ``PosteriorBelief.p_true`` contract users
observe.

Fallback role in the Stream P spec: when TRW-BP does not converge
within ``PCG_MAX_ITERS`` on a particular graph, the adapter retries
with damped LBP. If damped LBP also fails, caller returns the last-
iteration marginal with ``converged=False`` and
``algorithm="LBP-nonconvergent"`` so the frontend can render a
"verdict approximate" badge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adapters._pcg_graph import FactorGraph, log_normalize_2d


@dataclass(frozen=True)
class LBPResult:
    """Per-node marginals + convergence provenance."""

    marginals: np.ndarray  # shape (n, 2), row-normalised probabilities
    converged: bool
    iterations: int


def _init_messages(graph: FactorGraph) -> np.ndarray:
    """Uniform log-messages in both directions per edge.

    Indexed ``msg[edge_idx, dir, state]`` where ``dir ∈ {0, 1}`` is
    ``(i → j)`` vs ``(j → i)`` for edge ``k = (i, j)`` with ``i < j``.
    Log-space, so uniform = zero.
    """
    return np.zeros((graph.n_edges, 2, 2), dtype=np.float64)


def _compute_marginals(
    graph: FactorGraph,
    messages: np.ndarray,
) -> np.ndarray:
    """Per-node marginal = unary_log + Σ incoming messages, normalised."""
    n = graph.n_nodes
    marginals = np.zeros((n, 2), dtype=np.float64)
    log_acc = graph.unary_log_factors.copy()
    for k, (i, j) in enumerate(graph.edges):
        # message from j to i is msg[k, 1, :] (dir=1 per the convention).
        log_acc[i] += messages[k, 1, :]
        log_acc[j] += messages[k, 0, :]
    for i in range(n):
        marginals[i] = log_normalize_2d(log_acc[i])
    return marginals


def run_damped_lbp(
    graph: FactorGraph,
    *,
    max_iters: int = 200,
    convergence_tol: float = 1e-4,
    damping_factor: float = 0.8,
) -> LBPResult:
    """Damped loopy BP. Returns marginals + convergence flag.

    ``damping_factor`` is the weight on the NEW message; 1.0 = no
    damping (classic LBP), 0.8 = standard damped (weighted average of
    new + old). Too aggressive (near 0) slows convergence; too passive
    (near 1) oscillates on tight cycles.

    Convergence check: max absolute delta on node marginals between
    iterations. Not on messages — messages can chase their tail while
    the marginals have already settled, and it's the marginals users
    see in ``PosteriorBelief``.
    """
    if graph.n_edges == 0:
        # Tree-free, no loops → marginal is just the normalised unary.
        marginals = np.stack(
            [log_normalize_2d(graph.unary_log_factors[i]) for i in range(graph.n_nodes)],
            axis=0,
        )
        return LBPResult(marginals=marginals, converged=True, iterations=0)

    assert graph.binary_log_factors is not None  # graph.n_edges > 0 invariant
    binary = graph.binary_log_factors
    messages = _init_messages(graph)
    prev_marginals = _compute_marginals(graph, messages)

    for it in range(1, max_iters + 1):
        new_messages = messages.copy()

        # Collect neighbours per node for the "all incoming except
        # sender" message computation. Small graphs (n ≤ ~30) → rebuild
        # each iteration is cheaper than maintaining adjacency lists.
        for k, (i, j) in enumerate(graph.edges):
            # -- i → j message --
            # Sum incoming messages to i excluding the one from j.
            incoming_i = graph.unary_log_factors[i].copy()
            for kk, (ii, jj) in enumerate(graph.edges):
                if kk == k:
                    continue
                if ii == i:
                    incoming_i += messages[kk, 1, :]  # from jj to ii
                elif jj == i:
                    incoming_i += messages[kk, 0, :]  # from ii to jj
            # Outgoing message to j: log-sum-exp over x_i of
            # (incoming_i[x_i] + binary[k, x_i, x_j]).
            for xj in range(2):
                log_vals = incoming_i + binary[k, :, xj]
                m = np.max(log_vals)
                if not np.isfinite(m):
                    new_messages[k, 0, xj] = 0.0
                else:
                    new_messages[k, 0, xj] = m + float(np.log(np.sum(np.exp(log_vals - m))))
            # Normalise (subtract log-sum-exp over states — keeps numbers bounded).
            lse = new_messages[k, 0, 0] + float(
                np.log(np.exp(new_messages[k, 0, 0] - new_messages[k, 0, 0])
                       + np.exp(new_messages[k, 0, 1] - new_messages[k, 0, 0]))
            )
            new_messages[k, 0, :] -= lse

            # -- j → i message (symmetric) --
            incoming_j = graph.unary_log_factors[j].copy()
            for kk, (ii, jj) in enumerate(graph.edges):
                if kk == k:
                    continue
                if ii == j:
                    incoming_j += messages[kk, 1, :]
                elif jj == j:
                    incoming_j += messages[kk, 0, :]
            for xi in range(2):
                log_vals = incoming_j + binary[k, xi, :]
                m = np.max(log_vals)
                if not np.isfinite(m):
                    new_messages[k, 1, xi] = 0.0
                else:
                    new_messages[k, 1, xi] = m + float(np.log(np.sum(np.exp(log_vals - m))))
            lse = new_messages[k, 1, 0] + float(
                np.log(np.exp(new_messages[k, 1, 0] - new_messages[k, 1, 0])
                       + np.exp(new_messages[k, 1, 1] - new_messages[k, 1, 0]))
            )
            new_messages[k, 1, :] -= lse

        # Damping: weighted combination of new + old in log-space.
        # (True damping is on raw messages; the log-version is a
        # geometric mean that behaves similarly in the strongly-
        # polarised regime we care about.)
        messages = damping_factor * new_messages + (1.0 - damping_factor) * messages

        marginals = _compute_marginals(graph, messages)
        delta = float(np.max(np.abs(marginals - prev_marginals)))
        prev_marginals = marginals
        if delta < convergence_tol:
            return LBPResult(marginals=marginals, converged=True, iterations=it)

    return LBPResult(marginals=prev_marginals, converged=False, iterations=max_iters)


__all__ = ["LBPResult", "run_damped_lbp"]
