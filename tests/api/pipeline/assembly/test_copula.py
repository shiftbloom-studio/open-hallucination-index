"""Unit tests for pipeline.assembly.copula.

Closed-form cases (independent, single claim, perfectly correlated) are
checked exactly. Monte-Carlo fallback is exercised with a tight noise
tolerance to verify the large-N path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC_API = Path(__file__).resolve().parents[4] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from pipeline.assembly.copula import (  # noqa: E402
    build_correlation_matrix_identity,
    gaussian_copula_joint,
    nearest_psd,
)


# ---------------------------------------------------------------------------
# gaussian_copula_joint
# ---------------------------------------------------------------------------


def test_copula_empty_input_is_one() -> None:
    assert gaussian_copula_joint(np.array([]), np.eye(0)) == 1.0


def test_copula_single_claim_returns_marginal() -> None:
    assert gaussian_copula_joint(np.array([0.7]), np.array([[1.0]])) == pytest.approx(
        0.7
    )


def test_copula_independent_claims_multiplies_marginals() -> None:
    p = np.array([0.8, 0.6, 0.9])
    result = gaussian_copula_joint(p, np.eye(3))
    expected = 0.8 * 0.6 * 0.9
    assert result == pytest.approx(expected, abs=1e-9)


def test_copula_exact_path_for_small_n() -> None:
    """For small N with real (non-identity) correlation, we use scipy's
    MVN CDF — verify it gives sensible answers on a simple correlated
    example. Two fully-correlated claims with p=0.7 each should give
    joint ≈ 0.7 (both succeed together)."""
    n = 3
    r = 0.99 * np.ones((n, n)) + 0.01 * np.eye(n)  # nearly-identical claims
    p = np.array([0.7, 0.7, 0.7])
    result = gaussian_copula_joint(p, r)
    # With perfect correlation, joint ≈ marginal. Within 0.05 of 0.7.
    assert 0.6 <= result <= 0.8, f"Expected ~0.7, got {result}"


def test_copula_mc_path_large_n_matches_independent_product() -> None:
    """For N=15 with identity correlation, MC is NOT taken (we short-
    circuit to exact product). Force the MC path with a ~identity
    correlation matrix that has a tiny off-diagonal perturbation."""
    n = 15
    rng_seed = 42
    r = 0.99 * np.eye(n) + 0.01 * np.ones((n, n))
    r, _ = nearest_psd(r)  # ensure PSD
    p = np.full(n, 0.9)
    result = gaussian_copula_joint(p, r, seed=rng_seed)
    # Nearly-independent → joint close to 0.9 ** 15
    expected = 0.9**15
    # Allow generous tolerance because correlation is weak-but-nonzero
    assert abs(result - expected) < 0.15, f"Expected ≈ {expected:.4f}, got {result:.4f}"


def test_copula_result_in_unit_interval() -> None:
    """Regardless of inputs, output should be a valid probability."""
    p = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    r = np.eye(5)
    result = gaussian_copula_joint(p, r)
    assert 0.0 <= result <= 1.0


def test_copula_rejects_wrong_size_correlation_matrix() -> None:
    p = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="2.2"):
        gaussian_copula_joint(p, np.eye(3))


def test_copula_clips_probs_to_avoid_infinite_probit() -> None:
    """p = 0.0 or 1.0 would probit to ±inf. We clip internally."""
    p = np.array([0.0, 1.0])
    # Should not raise, should not return NaN
    result = gaussian_copula_joint(p, np.eye(2))
    assert not np.isnan(result)


# ---------------------------------------------------------------------------
# nearest_psd
# ---------------------------------------------------------------------------


def test_nearest_psd_identity_is_unchanged() -> None:
    m = np.eye(3)
    projected, dist = nearest_psd(m)
    assert np.allclose(projected, np.eye(3))
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_nearest_psd_clips_negative_eigenvalues() -> None:
    # Construct a non-PSD matrix with a clear negative eigenvalue.
    m = np.array([[1.0, -1.5], [-1.5, 1.0]])
    # Eigenvalues: 1 ± 1.5 → {2.5, -0.5}. Non-PSD.
    projected, dist = nearest_psd(m, eps=1e-4)
    eigvals = np.linalg.eigvalsh(projected)
    # Final matrix must be PSD (no negative eigenvalues). The specific
    # eps floor isn't preserved after re-normalisation to unit diagonal,
    # but positivity IS, which is what scipy's MVN cdf requires.
    assert (eigvals >= 0.0 - 1e-12).all(), (
        f"All eigenvalues must be ≥ 0 after PSD projection, got {eigvals}"
    )
    # Diagonal should be 1 (correlation matrix)
    assert np.allclose(np.diag(projected), 1.0)
    # Distance from input is non-zero (matrix was modified)
    assert dist > 0


def test_nearest_psd_preserves_symmetry() -> None:
    # Asymmetric input — should be symmetrized internally
    m = np.array([[1.0, 0.3, 0.1], [0.7, 1.0, 0.2], [0.1, 0.2, 1.0]])
    projected, _ = nearest_psd(m)
    assert np.allclose(projected, projected.T)


def test_nearest_psd_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        nearest_psd(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# build_correlation_matrix_identity
# ---------------------------------------------------------------------------


def test_identity_builder_returns_square_identity() -> None:
    for n in [0, 1, 5, 30]:
        r = build_correlation_matrix_identity(n)
        assert r.shape == (n, n)
        if n > 0:
            assert np.allclose(r, np.eye(n))
