"""Gaussian copula joint-truth aggregator. Spec §9.

Computes ``Φ_R(Φ⁻¹(b_1), ..., Φ⁻¹(b_N))`` — the Gaussian-copula joint
probability that all N claims are true, accounting for their correlation
structure inferred from L3 claim↔claim NLI.

Three cases:
  * N == 0 → 1.0 (no claims = nothing to distrust).
  * N == 1 → b_c (degenerate; copula collapses to the marginal).
  * N ≥ 2, R = I (Phase 1 default) → fast path: ∏_i b_i.
  * N ≥ 2, R ≠ I → SciPy's multivariate-normal CDF for N ≤ 10, Monte
    Carlo for larger N. Both branches return equivalent values up to
    MC-error (< 0.005 at 10k samples).

PSD enforcement: raw NLI-derived correlation matrices are generally not
positive-semi-definite. ``nearest_psd`` projects to the closest PSD
matrix in Frobenius norm via eigenvalue clipping, then re-normalizes the
diagonal to 1. The Frobenius distance is returned alongside so callers
can log it to ``DocumentVerdict.model_versions`` for auditability.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Floor to avoid probit(0) = -inf and probit(1) = +inf
_EPS = 1e-6


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1.0 - _EPS)


def nearest_psd(matrix: np.ndarray, *, eps: float = 1e-4) -> tuple[np.ndarray, float]:
    """Project a square matrix to the nearest PSD matrix (Frobenius norm).

    Algorithm: eigendecomposition, clip negative eigenvalues to ``eps``,
    reconstruct, then rescale the diagonal to unit so the result is a
    valid correlation matrix.

    Returns ``(projected_matrix, frobenius_distance_from_input)``.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("nearest_psd requires a square matrix")
    # Symmetrize first (eigh needs symmetric input; raw NLI correlations may drift)
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, eps)
    reconstructed = (eigvecs * clipped) @ eigvecs.T

    # Rescale to unit diagonal (preserve correlation-matrix semantics)
    diag = np.sqrt(np.clip(np.diag(reconstructed), eps, None))
    inv_sqrt_diag = 1.0 / diag
    # Outer product scaling: R[i,j] / sqrt(R[i,i]*R[j,j])
    normalised = reconstructed * inv_sqrt_diag[:, None] * inv_sqrt_diag[None, :]
    # Ensure exact 1.0 on the diagonal
    np.fill_diagonal(normalised, 1.0)

    frobenius_distance = float(np.linalg.norm(normalised - matrix, ord="fro"))
    return normalised, frobenius_distance


def _is_identity(r: np.ndarray, *, tol: float = 1e-8) -> bool:
    n = r.shape[0]
    return bool(np.allclose(r, np.eye(n), atol=tol))


def gaussian_copula_joint(
    p_per_claim: np.ndarray,
    correlation_matrix_R: np.ndarray,
    *,
    n_mc_samples: int = 10_000,
    seed: int | None = None,
) -> float:
    """Compute ``Φ_R(Φ⁻¹(p_1), ..., Φ⁻¹(p_N))``.

    Fast paths:
      * N == 0 → 1.0
      * N == 1 → p_per_claim[0]
      * R = I (any N) → ∏_i p_per_claim[i]  (exact, no MC error)

    Exact path (N ≤ 10): SciPy's Genz algorithm via
    ``scipy.stats.multivariate_normal.cdf``.

    Monte Carlo path (N > 10): draw n_mc_samples from N(0, R), count
    the fraction where every coordinate exceeds -Φ⁻¹(p_i).
    """
    p = np.asarray(p_per_claim, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return 1.0
    if n == 1:
        return float(p[0])

    p = _clip_probs(p)

    r = np.asarray(correlation_matrix_R, dtype=np.float64)
    if r.shape != (n, n):
        raise ValueError(f"correlation_matrix_R must be {n}×{n}, got {r.shape}")

    # Fast path when claims are independent
    if _is_identity(r):
        return float(np.prod(p))

    # General path: compute Φ_R(Φ⁻¹(p))
    from scipy.stats import multivariate_normal, norm

    z = norm.ppf(p)  # probit transform

    if n <= 10:
        try:
            # scipy's MVN CDF with the Genz algorithm is the exact answer
            # (up to Genz's own quasi-MC tolerance, default 1e-8).
            return float(multivariate_normal.cdf(z, mean=np.zeros(n), cov=r))
        except Exception as exc:
            logger.warning("SciPy MVN CDF failed (%s); falling back to Monte Carlo", exc)

    # Monte Carlo fallback for large N or scipy failure
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        mean=np.zeros(n), cov=r, size=n_mc_samples, check_valid="raise"
    )
    # P(Z_c ≤ z_c ∀ c) = empirical joint CDF
    all_below = np.all(samples <= z[None, :], axis=1)
    return float(all_below.mean())


def build_correlation_matrix_identity(n: int) -> np.ndarray:
    """Phase 1 helper — returns an N×N identity matrix.

    When Phase 2 PCG lands, the real correlation matrix will be built
    from L3 claim↔claim NLI (entail − contradict) via
    ``nearest_psd``. Using identity in Phase 1 means the copula returns
    ``∏ p_i`` exactly, which is the right "claims are independent"
    answer given we don't yet have edge information.
    """
    return np.eye(n)
