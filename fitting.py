"""
stage/fitting.py

Least-squares fitting utilities for the NEBULA Zodiacal Light (ZL) stage.

Role in pipeline
----------------
The ZL stage evaluates zodiacal-light samples on a grid of *sample directions on sky* per
frame (via WCS projection). Those samples are expensive to compute (WSL + m4opt), so the
Windows-side stage stores a compact representation per frame:

  1) plane(3):    c0 + c1*u + c2*v
  2) quad(6):     c0 + c1*u + c2*v + c3*u^2 + c4*u*v + c5*v^2

This module performs those per-frame fits and returns both coefficients and an RMS residual
diagnostic per frame.

Key concept: normalized (u, v)
------------------------------
The ZL stage uses a *normalized detector-plane coordinate system* for fitting:

- u_norm and v_norm are dimensionless coordinates for each sample location.
- They are typically constructed so that the sampled field maps into a stable range
  such as [-1, +1] in each axis, independent of the sensor resolution.
- Normalization improves numerical conditioning (especially for quadratic terms) and
  makes coefficients comparable across sensors/configs, as long as the normalization
  convention is consistent.

This module does not define how u_norm/v_norm are computed; it assumes upstream code
(stage/projection_wcs.py) provides them and enforces basic shape/validity constraints.

Design constraints
------------------
- Windows-safe: numpy-only, no WSL/m4opt imports.
- FAIL-FAST: validates shapes and non-finite values.
- Deterministic: uses a shared design matrix per window/grid and solves via pseudo-inverse.

Notes on method
---------------
We fit multiple frames at once by building one design matrix X for the sample grid
(shape: n_samples × n_params) and computing its pseudo-inverse. For each frame i,
coeffs[i] are solved in the least-squares sense as:

    coeffs[i] = y[i] @ pinv(X).T

where y[i] is the sample vector for frame i (length n_samples).

This matches the implementation currently embedded in NEBULA_ZODIACAL_LIGHT_STAGE.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_fit_inputs(phi: np.ndarray, u_norm: np.ndarray, v_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and normalize inputs for fitting.

    Parameters
    ----------
    phi : np.ndarray
        Sample values, expected shape (n_frames, n_samples).
        Each row corresponds to one time/frame, each column corresponds to one sample
        location in the (u_norm, v_norm) grid.
    u_norm, v_norm : np.ndarray
        Normalized sample coordinates, each expected shape (n_samples,).

    Returns
    -------
    y : np.ndarray
        phi cast to float64, shape (n_frames, n_samples).
    u_ : np.ndarray
        u_norm cast to float64 and flattened, shape (n_samples,).
    v_ : np.ndarray
        v_norm cast to float64 and flattened, shape (n_samples,).

    Raises
    ------
    ValueError
        If shapes are inconsistent or any array contains non-finite values.
    """
    y = np.asarray(phi, dtype=np.float64)
    u_ = np.asarray(u_norm, dtype=np.float64).reshape(-1)
    v_ = np.asarray(v_norm, dtype=np.float64).reshape(-1)

    if y.ndim != 2:
        raise ValueError("phi must have shape (n_frames, n_samples)")
    if u_.ndim != 1 or v_.ndim != 1:
        raise ValueError("u_norm and v_norm must be 1D arrays of shape (n_samples,)")
    if u_.size != y.shape[1] or v_.size != y.shape[1]:
        raise ValueError("u_norm/v_norm must match phi's n_samples")

    # FAIL-FAST: non-finite values should not silently propagate into coefficients.
    if not np.isfinite(y).all():
        raise ValueError("phi contains non-finite values (NaN/Inf); cannot fit.")
    if not np.isfinite(u_).all() or not np.isfinite(v_).all():
        raise ValueError("u_norm/v_norm contain non-finite values (NaN/Inf); cannot fit.")

    return y, u_, v_


def fit_plane3(phi: np.ndarray, u_norm: np.ndarray, v_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit plane coefficients per frame:

        y = c0 + c1*u + c2*v

    Parameters
    ----------
    phi : np.ndarray
        Samples, shape (n_frames, n_samples).
    u_norm, v_norm : np.ndarray
        Normalized sample coordinates, shape (n_samples,).

    Returns
    -------
    coeffs : np.ndarray
        Shape (n_frames, 3) with columns [c0, c1, c2].
        These are coefficients in the (u_norm, v_norm) coordinate system.
    rms_per_frame : np.ndarray
        Shape (n_frames,), RMS residuals against the samples:
            sqrt(mean((y - y_hat)^2)) for each frame.

    Notes
    -----
    Uses a shared design matrix across frames and solves with a pseudo-inverse for speed.
    This is a least-squares fit when the system is overdetermined (n_samples > 3).
    """
    y, u_, v_ = _validate_fit_inputs(phi, u_norm, v_norm)

    n_samples = int(y.shape[1])
    n_params = 3
    if n_samples < n_params:
        raise ValueError(f"fit_plane3 requires at least {n_params} samples; got n_samples={n_samples}.")

    # Design matrix: (n_samples, 3) for [1, u, v]
    X = np.stack([np.ones_like(u_), u_, v_], axis=1)

    # Fail-fast if the design matrix is rank-deficient (degenerate sample geometry).
    rank = int(np.linalg.matrix_rank(X))
    if rank < n_params:
        raise ValueError(f"fit_plane3 design matrix is rank-deficient: rank={rank} < {n_params}.")

    # Moore-Penrose pseudo-inverse: (3, n_samples)
    pinv = np.linalg.pinv(X)

    # Solve all frames at once:
    #   y:     (n_frames, n_samples)
    #   pinv.T (n_samples, 3)
    # => coeffs: (n_frames, 3)
    coeffs = y @ pinv.T

    # Predicted samples and residuals for RMS diagnostic
    yhat = coeffs @ X.T  # (n_frames, n_samples)
    resid = y - yhat
    rms = np.sqrt(np.mean(resid * resid, axis=1))

    return coeffs.astype(np.float64), rms.astype(np.float64)


def fit_quad6(phi: np.ndarray, u_norm: np.ndarray, v_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit quadratic coefficients per frame:

        y = c0 + c1*u + c2*v + c3*u^2 + c4*u*v + c5*v^2

    Parameters
    ----------
    phi : np.ndarray
        Samples, shape (n_frames, n_samples).
    u_norm, v_norm : np.ndarray
        Normalized sample coordinates, shape (n_samples,).

    Returns
    -------
    coeffs : np.ndarray
        Shape (n_frames, 6) with columns [c0, c1, c2, c3, c4, c5] corresponding to:
            [1, u, v, u^2, u*v, v^2]
    rms_per_frame : np.ndarray
        Shape (n_frames,), RMS residuals against the samples:
            sqrt(mean((y - y_hat)^2)) for each frame.

    Notes
    -----
    Uses a shared design matrix across frames and solves with a pseudo-inverse for speed.
    Normalized coordinates (u_norm, v_norm) are especially important here to keep u^2, v^2
    terms numerically well-conditioned.
    """
    y, u_, v_ = _validate_fit_inputs(phi, u_norm, v_norm)

    n_samples = int(y.shape[1])
    n_params = 6
    if n_samples < n_params:
        raise ValueError(f"fit_quad6 requires at least {n_params} samples; got n_samples={n_samples}.")

    # Design matrix: (n_samples, 6) for [1, u, v, u^2, u*v, v^2]
    X = np.stack(
        [
            np.ones_like(u_),
            u_,
            v_,
            u_ * u_,
            u_ * v_,
            v_ * v_,
        ],
        axis=1,
    )

    # Fail-fast if the design matrix is rank-deficient (degenerate sample geometry).
    rank = int(np.linalg.matrix_rank(X))
    if rank < n_params:
        raise ValueError(f"fit_quad6 design matrix is rank-deficient: rank={rank} < {n_params}.")

    # Moore-Penrose pseudo-inverse: (6, n_samples)
    pinv = np.linalg.pinv(X)

    # Solve all frames at once
    coeffs = y @ pinv.T  # (n_frames, 6)

    # Predicted samples and residuals for RMS diagnostic
    yhat = coeffs @ X.T
    resid = y - yhat
    rms = np.sqrt(np.mean(resid * resid, axis=1))

    return coeffs.astype(np.float64), rms.astype(np.float64)
