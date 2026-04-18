"""Wave 3 Stream P — factor-graph data structure for PCG inference.

Ising-style binary nodes (one per claim, state ∈ {F=0, T=1}) with

* **unary factors** from claim-evidence NLI distributions
  (``NLIDistribution.entail`` → T-pref, ``contradict`` → F-pref), and
* **binary factors** from claim-claim NLI distributions between pairs
  that survive the entity-overlap short-circuit (SUPPORT →
  same-label-preferred, REFUTE → opposite-label-preferred).

The graph is the sole intermediate the three solver modules
(``_pcg_lbp``, ``_pcg_trw_bp``, ``_pcg_gibbs``) consume. Keeping it as
dense numpy arrays (rather than e.g. a pgmpy FactorGraph) keeps Lambda
cold-starts tight and avoids pulling a heavy pinned-version dependency
for one function.

Node / edge construction is deterministic in input order — callers can
rely on the node-index → claim-id mapping (``claim_ids[i]``) when
reading marginals back out.

Factor convention (critical for the solvers):

* **Unary** ``u[i] = [log φ(x_i=0), log φ(x_i=1)]`` in log space. For a
  claim with evidence NLI mass `(entail, contradict, neutral)`,
  ``φ(x=T) ∝ entail + 0.5·neutral`` and
  ``φ(x=F) ∝ contradict + 0.5·neutral`` — neutral mass is split
  evenly so it adds no signal in either direction (matches the Stream
  P spec semantic that neutrals are treated as no-signal; see Hebel A
  on the Beta-posterior path for the equivalent rule at the
  Phase-2 adapter).
* **Binary** ``b[k] = [[log φ(00), log φ(01)], [log φ(10), log φ(11)]]``
  for edge ``k = (i, j)``. SUPPORT tilts toward the ``00, 11`` diagonal
  (claims agree); REFUTE tilts toward the ``01, 10`` off-diagonal
  (claims disagree). Edge strength is ``max(entail, contradict) -
  neutral`` clipped to [0, 1] and scaled by a per-edge weight that
  shows up as the magnitude of the log-factor tilt.

Log space throughout: probabilistic factors multiply, log-factors add,
so message-passing loops are numerically stable for the strongly-
polarised edges that NLI produces (e.g. entail=0.98, contradict=0.01).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from models.entities import Claim
    from models.nli import NLIDistribution


# ---------------------------------------------------------------------------
# Factor-graph container
# ---------------------------------------------------------------------------


@dataclass
class FactorGraph:
    """Ising-style binary claim graph for joint-inference solvers.

    ``claim_ids[i]`` is the UUID of the claim at node ``i``. Solver
    outputs map back to claim UUIDs via this array. Edges are
    undirected and stored once (``i < j``) to keep the MST cover
    deterministic in TRW-BP.
    """

    claim_ids: list[UUID]
    unary_log_factors: np.ndarray  # shape (n, 2); [F, T] log-factor per node
    edges: list[tuple[int, int]] = field(default_factory=list)
    binary_log_factors: np.ndarray | None = None  # shape (E, 2, 2) or None
    # Edge-appearance-probability cache for TRW-BP; computed lazily.
    edge_appearance: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.claim_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


# ---------------------------------------------------------------------------
# Unary factor construction (claim-evidence NLI)
# ---------------------------------------------------------------------------


def _unary_log_factor_from_evidence_nli(
    nli_list: list["NLIDistribution"],
    *,
    min_unary_tilt: float = 1e-3,
) -> np.ndarray:
    """Aggregate per-claim evidence NLI distributions into a
    ``[log φ(F), log φ(T)]`` unary factor pair.

    Per-evidence contributions multiply (add in log space). Neutral
    mass is split 50/50 so off-topic evidence adds zero net signal —
    matches the Phase-2 Hebel-A rule for neutral-label skip at the
    posterior layer. The ``min_unary_tilt`` floor prevents a single
    extreme (entail=1.0, contradict=0.0) from collapsing the log
    factor to ``-inf`` and breaking the solvers; it lets a
    counter-example still perturb the posterior.
    """
    log_f_false = 0.0
    log_f_true = 0.0
    for nli in nli_list:
        # Entail → supports T, contradict → supports F, neutral split evenly.
        f_true = max(nli.entail + 0.5 * nli.neutral, min_unary_tilt)
        f_false = max(nli.contradict + 0.5 * nli.neutral, min_unary_tilt)
        log_f_true += float(np.log(f_true))
        log_f_false += float(np.log(f_false))
    return np.array([log_f_false, log_f_true], dtype=np.float64)


def _uniform_unary_log_factor() -> np.ndarray:
    """No evidence → uniform prior (log φ=0 both states)."""
    return np.zeros(2, dtype=np.float64)


# ---------------------------------------------------------------------------
# Binary factor construction (claim-claim NLI)
# ---------------------------------------------------------------------------


def _binary_log_factor_from_cc_nli(
    nli: "NLIDistribution",
    *,
    edge_weight_scale: float = 1.0,
    min_binary_tilt: float = 1e-3,
) -> np.ndarray:
    """Convert a claim-claim NLI distribution into a 2×2 log-factor.

    Factor matrix indexed ``[x_i, x_j]``:

    * SUPPORT-tilted (entail dominant) → diagonal preferred
      (both T or both F).
    * REFUTE-tilted (contradict dominant) → off-diagonal preferred
      (exactly one T).
    * Neutral-tilted → near-uniform (no edge effectively).

    The magnitude of the tilt scales with ``edge_weight = max(entail,
    contradict) - neutral`` clamped to [0, 1]. ``edge_weight_scale``
    lets callers dial down edge influence for ``balanced`` vs
    ``maximum`` rigor without rebuilding the whole graph.
    """
    entail = nli.entail
    contradict = nli.contradict
    neutral = nli.neutral
    edge_weight = max(entail, contradict) - neutral
    edge_weight = max(0.0, min(1.0, edge_weight)) * edge_weight_scale

    # base factor = [[support-tilt, refute-tilt], [refute-tilt, support-tilt]]
    #             ≈ [[entail+neutral/2, contradict+neutral/2], [contradict..., entail...]]
    # Clamp with min_binary_tilt so log is finite.
    diag = max(entail + 0.5 * neutral, min_binary_tilt)
    off_diag = max(contradict + 0.5 * neutral, min_binary_tilt)
    # Re-weight by edge_weight: when edge_weight → 0, collapse to uniform;
    # when edge_weight → 1, keep full tilt.
    diag_w = diag ** edge_weight
    off_w = off_diag ** edge_weight
    factor = np.array(
        [[diag_w, off_w], [off_w, diag_w]],
        dtype=np.float64,
    )
    return np.log(factor)


# ---------------------------------------------------------------------------
# Graph construction entry point
# ---------------------------------------------------------------------------


def build_factor_graph(
    claims: list["Claim"],
    nli_claim_evidence: dict[UUID, list["NLIDistribution"]],
    nli_claim_claim: dict[tuple[UUID, UUID], "NLIDistribution"],
    *,
    edge_weight_scale: float = 1.0,
) -> FactorGraph:
    """Build a ``FactorGraph`` from decomposed claims + NLI tables.

    ``nli_claim_evidence[claim_id]`` may be missing or empty — that
    claim gets a uniform unary factor (nothing to say).

    ``nli_claim_claim`` keys are **ordered** tuples ``(id_a, id_b)``
    with ``id_a < id_b`` lexical (matches ``NLIService.claim_claim``
    contract). The dispatcher upstream applies the entity-overlap
    short-circuit and hard cap before handing pairs to this function;
    we trust its filtering and just consume what's there.
    """
    claim_ids = [c.id for c in claims]
    n = len(claim_ids)
    unary = np.zeros((n, 2), dtype=np.float64)
    for i, c in enumerate(claims):
        nli_list = nli_claim_evidence.get(c.id, [])
        unary[i] = (
            _unary_log_factor_from_evidence_nli(nli_list)
            if nli_list
            else _uniform_unary_log_factor()
        )

    # Map UUID → node index for binary-factor lookup.
    uuid_to_idx = {cid: i for i, cid in enumerate(claim_ids)}
    edges: list[tuple[int, int]] = []
    binary_rows: list[np.ndarray] = []
    for (id_a, id_b), nli in nli_claim_claim.items():
        if id_a not in uuid_to_idx or id_b not in uuid_to_idx:
            continue  # stale pair referencing a claim not in this batch
        i = uuid_to_idx[id_a]
        j = uuid_to_idx[id_b]
        if i == j:
            continue
        # Canonical ordering (i < j) so MST cover in TRW-BP is deterministic.
        if i > j:
            i, j = j, i
        edges.append((i, j))
        binary_rows.append(
            _binary_log_factor_from_cc_nli(nli, edge_weight_scale=edge_weight_scale)
        )

    binary = (
        np.stack(binary_rows, axis=0)
        if binary_rows
        else None
    )

    return FactorGraph(
        claim_ids=claim_ids,
        unary_log_factors=unary,
        edges=edges,
        binary_log_factors=binary,
    )


# ---------------------------------------------------------------------------
# Marginal → PosteriorBelief helpers (shared by solvers)
# ---------------------------------------------------------------------------


def log_normalize_2d(log_marginal: np.ndarray) -> np.ndarray:
    """Log-sum-exp normalise a log-space binary marginal to a proper
    probability distribution over {F, T}.

    Guards against all-``-inf`` inputs (shouldn't happen with the
    ``min_unary_tilt`` floor, but be defensive — the conformal layer
    downstream will reject invalid probabilities anyway)."""
    m = np.max(log_marginal)
    if not np.isfinite(m):
        # Degenerate input — fall back to uniform.
        return np.array([0.5, 0.5], dtype=np.float64)
    lse = m + float(np.log(np.sum(np.exp(log_marginal - m))))
    return np.exp(log_marginal - lse)


__all__ = [
    "FactorGraph",
    "build_factor_graph",
    "log_normalize_2d",
]
