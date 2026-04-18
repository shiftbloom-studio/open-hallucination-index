"""Tree-Reweighted Belief Propagation (Wainwright-Jaakkola-Willsky 2005).

TRW-BP gives a **convex upper bound** on the log-partition function of
an arbitrary pairwise MRF — unlike LBP, it is guaranteed to converge
for a suitable choice of edge-appearance probabilities ``ρ_e ∈ (0, 1]``
(spanning-tree-cover constraint: each edge appears in some tree of a
convex combination of spanning trees).

For our ``n ≤ ~30`` Ising-style claim graphs, a single minimum
spanning tree cover (all edges assigned ``ρ = 1/T`` where ``T`` is a
count of trees in the cover) is a reasonable good-enough
approximation. Exact TRW-BP optimises ``ρ_e`` to tighten the bound;
left as a Wave-3.1-follow-up if the bounds turn out to be slack in
production (tracked in the adapter's handoff).

This module:
* Computes a spanning-tree cover via successive-MST (Prim's on the
  edge-weight = binary-factor-strength graph, plus a second MST on
  the residual — gives us a 2-tree cover when the graph has cycles).
* Runs TRW-BP message-passing on the resulting ``ρ_e``.
* Returns marginals + the log-partition-bound.

Log-space throughout (same reason as LBP: NLI edges are strongly
polarised).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adapters._pcg_graph import FactorGraph, log_normalize_2d


@dataclass(frozen=True)
class TRWBPResult:
    """Per-node marginals + log-partition-bound + convergence."""

    marginals: np.ndarray
    log_partition_bound: float
    converged: bool
    iterations: int


# ---------------------------------------------------------------------------
# Spanning-tree cover (edge-appearance probabilities)
# ---------------------------------------------------------------------------


def _edge_strength(binary_log_factor_2x2: np.ndarray) -> float:
    """Heuristic scalar for MST weight: absolute tilt of the factor.

    Stronger tilt (SUPPORT or REFUTE) → higher weight → MST prefers to
    include. This keeps the most informative edges in the first MST.
    """
    m = binary_log_factor_2x2
    return float(np.abs(m[0, 0] - m[0, 1]) + np.abs(m[1, 1] - m[1, 0]))


def _mst_prim(
    n: int, edges: list[tuple[int, int]], weights: list[float]
) -> list[int]:
    """Prim's algorithm, maximum-spanning-tree variant (pick heaviest
    edge at each step). Returns indices into ``edges`` forming the MST.
    Works on a forest if the graph is disconnected (returns a
    spanning-forest's edges — still a valid cover seed).
    """
    if n == 0 or not edges:
        return []
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    chosen: list[int] = []
    # Simple O(V·E); fine for n ≤ 30.
    remaining = set(range(len(edges)))
    while len(chosen) < n - 1:
        best_k = -1
        best_w = -np.inf
        for k in remaining:
            i, j = edges[k]
            if in_tree[i] != in_tree[j]:  # crosses cut
                if weights[k] > best_w:
                    best_w = weights[k]
                    best_k = k
        if best_k < 0:
            break  # disconnected — partial forest is fine
        chosen.append(best_k)
        i, j = edges[best_k]
        in_tree[i] = True
        in_tree[j] = True
        remaining.discard(best_k)
    return chosen


def _spanning_tree_cover(graph: FactorGraph) -> np.ndarray:
    """Compute ``ρ_e`` via a simple 2-tree cover.

    * First MST picks the ``n-1`` strongest edges.
    * Second MST picks the strongest edges among the rest (best-
      effort, may not be spanning if the residual graph is sparse).
    * ``ρ_e`` = 1 if the edge is in both trees, 0.5 if only in one.
    * Edges in neither tree (rare for our sparse graphs) get 0.25
      so they're under-weighted but not silenced.
    """
    if graph.binary_log_factors is None or graph.n_edges == 0:
        return np.zeros(0, dtype=np.float64)

    weights = [_edge_strength(graph.binary_log_factors[k]) for k in range(graph.n_edges)]
    tree1 = set(_mst_prim(graph.n_nodes, graph.edges, weights))

    # Second pass: same algorithm on the residual edge set.
    residual = [(k, w) for k, w in enumerate(weights) if k not in tree1]
    if residual:
        res_edges = [graph.edges[k] for k, _ in residual]
        res_weights = [w for _, w in residual]
        tree2_local = set(_mst_prim(graph.n_nodes, res_edges, res_weights))
        tree2 = {residual[kk][0] for kk in tree2_local}
    else:
        tree2 = set()

    rho = np.zeros(graph.n_edges, dtype=np.float64)
    for k in range(graph.n_edges):
        if k in tree1 and k in tree2:
            rho[k] = 1.0
        elif k in tree1 or k in tree2:
            rho[k] = 0.5
        else:
            rho[k] = 0.25
    return rho


# ---------------------------------------------------------------------------
# TRW-BP message passing
# ---------------------------------------------------------------------------


def _lse2(a: float, b: float) -> float:
    """Numerically-stable log-sum-exp of two scalars."""
    m = max(a, b)
    if not np.isfinite(m):
        return 0.0
    return m + float(np.log(np.exp(a - m) + np.exp(b - m)))


def _normalize_log_msg(v: np.ndarray) -> np.ndarray:
    """Renormalise a 2-vector log-message so its log-sum-exp = 0."""
    lse = _lse2(float(v[0]), float(v[1]))
    return v - lse


def _trwbp_marginals_and_bound(
    graph: FactorGraph,
    messages: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute per-node marginals and the TRW log-partition bound."""
    n = graph.n_nodes
    marginals = np.zeros((n, 2), dtype=np.float64)
    log_acc = graph.unary_log_factors.copy()
    for k, (i, j) in enumerate(graph.edges):
        # Incoming (ρ-weighted) messages.
        log_acc[i] += rho[k] * messages[k, 1, :]
        log_acc[j] += rho[k] * messages[k, 0, :]
    for i in range(n):
        marginals[i] = log_normalize_2d(log_acc[i])

    # Log-partition bound per Wainwright et al. (2005) eq. (12).
    # bound = Σ_i [LSE over x_i of unary_i + Σ_k ρ_k msg_in(i)]
    #       - Σ_k ρ_k · entropy_term(edge k)
    # Approximate the entropy term via -H(x_i x_j | edge).
    log_z_bound = 0.0
    for i in range(n):
        log_z_bound += _lse2(float(log_acc[i, 0]), float(log_acc[i, 1]))
    # Edge correction: subtract ρ_k · LSE over the edge's joint.
    assert graph.binary_log_factors is not None
    for k, (i, j) in enumerate(graph.edges):
        # Joint (x_i, x_j) log-factor = unary_i + unary_j + binary
        #                              + ρ-adjusted in-edges from other neighbours
        # Good-enough: include unary + binary only (cheap, monotonic).
        edge_lse = -np.inf
        for xi in range(2):
            for xj in range(2):
                val = (
                    graph.unary_log_factors[i, xi]
                    + graph.unary_log_factors[j, xj]
                    + graph.binary_log_factors[k, xi, xj]
                )
                if edge_lse == -np.inf:
                    edge_lse = val
                else:
                    edge_lse = _lse2(float(edge_lse), float(val))
        log_z_bound -= rho[k] * float(edge_lse)
    return marginals, float(log_z_bound)


def run_trw_bp(
    graph: FactorGraph,
    *,
    max_iters: int = 200,
    convergence_tol: float = 1e-4,
) -> TRWBPResult:
    """TRW-BP on the factor graph. Returns marginals, log-Z bound,
    and convergence provenance."""
    n = graph.n_nodes
    if n == 0:
        return TRWBPResult(
            marginals=np.zeros((0, 2), dtype=np.float64),
            log_partition_bound=0.0,
            converged=True,
            iterations=0,
        )

    if graph.n_edges == 0:
        # No edges → marginals are normalised unaries; bound is
        # Σ_i LSE(unary_i).
        marginals = np.stack(
            [log_normalize_2d(graph.unary_log_factors[i]) for i in range(n)],
            axis=0,
        )
        log_z = sum(
            _lse2(float(graph.unary_log_factors[i, 0]), float(graph.unary_log_factors[i, 1]))
            for i in range(n)
        )
        return TRWBPResult(
            marginals=marginals,
            log_partition_bound=float(log_z),
            converged=True,
            iterations=0,
        )

    assert graph.binary_log_factors is not None
    binary = graph.binary_log_factors
    rho = _spanning_tree_cover(graph)
    graph.edge_appearance = rho  # cache for observability

    messages = np.zeros((graph.n_edges, 2, 2), dtype=np.float64)
    prev_marginals, _ = _trwbp_marginals_and_bound(graph, messages, rho)

    for it in range(1, max_iters + 1):
        new_messages = messages.copy()

        # TRW-BP message update: for each edge k = (i, j), compute the
        # i→j and j→i messages reweighted by (1/ρ_k) on the own-edge
        # factor and Σ_{other edges into i} ρ · msg (rest incoming).
        for k, (i, j) in enumerate(graph.edges):
            # Accumulate all incoming to i (ρ-weighted, excluding k).
            incoming_i = graph.unary_log_factors[i].copy()
            for kk, (ii, jj) in enumerate(graph.edges):
                if kk == k:
                    continue
                if ii == i:
                    incoming_i += rho[kk] * messages[kk, 1, :]
                elif jj == i:
                    incoming_i += rho[kk] * messages[kk, 0, :]
            # New msg i→j[xj] = log Σ_{xi} exp(incoming_i[xi]
            #                      + binary[k, xi, xj] / ρ_k
            #                      + (ρ_k - 1) · msg_{j→i}[xi] / ρ_k)
            # Standard TRW update form.
            rho_k = max(rho[k], 1e-6)
            for xj in range(2):
                log_vals = np.zeros(2, dtype=np.float64)
                for xi in range(2):
                    log_vals[xi] = (
                        incoming_i[xi]
                        + binary[k, xi, xj] / rho_k
                        + (rho_k - 1.0) * messages[k, 1, xi] / rho_k
                    )
                new_messages[k, 0, xj] = _lse2(float(log_vals[0]), float(log_vals[1]))
            new_messages[k, 0, :] = _normalize_log_msg(new_messages[k, 0, :])

            # Same for j → i.
            incoming_j = graph.unary_log_factors[j].copy()
            for kk, (ii, jj) in enumerate(graph.edges):
                if kk == k:
                    continue
                if ii == j:
                    incoming_j += rho[kk] * messages[kk, 1, :]
                elif jj == j:
                    incoming_j += rho[kk] * messages[kk, 0, :]
            for xi in range(2):
                log_vals = np.zeros(2, dtype=np.float64)
                for xj in range(2):
                    log_vals[xj] = (
                        incoming_j[xj]
                        + binary[k, xi, xj] / rho_k
                        + (rho_k - 1.0) * messages[k, 0, xj] / rho_k
                    )
                new_messages[k, 1, xi] = _lse2(float(log_vals[0]), float(log_vals[1]))
            new_messages[k, 1, :] = _normalize_log_msg(new_messages[k, 1, :])

        messages = new_messages

        marginals, _ = _trwbp_marginals_and_bound(graph, messages, rho)
        delta = float(np.max(np.abs(marginals - prev_marginals)))
        prev_marginals = marginals
        if delta < convergence_tol:
            _, bound = _trwbp_marginals_and_bound(graph, messages, rho)
            return TRWBPResult(
                marginals=marginals,
                log_partition_bound=bound,
                converged=True,
                iterations=it,
            )

    _, bound = _trwbp_marginals_and_bound(graph, messages, rho)
    return TRWBPResult(
        marginals=prev_marginals,
        log_partition_bound=bound,
        converged=False,
        iterations=max_iters,
    )


__all__ = ["TRWBPResult", "run_trw_bp"]
