from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def make_sample_pixel_grid(
    *,
    rows: int,
    cols: int,
    n_u: int,
    n_v: int,
    margin_pix: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create a modest sampling grid for fitting plane/quadratic coefficients.

    Parameters
    ----------
    rows, cols : int
        Detector shape.
    n_u, n_v : int
        Number of sample points across the detector in x (u) and y (v).
        Example: n_u=3, n_v=3 -> 9 samples.
    margin_pix : int
        Exclude pixels within this margin of the detector edges.

    Returns
    -------
    dict with arrays:
      x_pix : (n_samples,) float64
      y_pix : (n_samples,) float64
      u_norm : (n_samples,) float64   normalized [-1,1] coordinate
      v_norm : (n_samples,) float64

    Notes
    -----
    Flatten order is row-major on the (v,u) meshgrid:
      v index (y) varies slow, u index (x) varies fast.
    """
    if n_u < 2 or n_v < 2:
        raise ValueError("n_u and n_v should be >= 2 for stable fitting.")
    if margin_pix < 0:
        raise ValueError("margin_pix must be >= 0.")

    x0 = float(margin_pix)
    x1 = float(cols - 1 - margin_pix)
    y0 = float(margin_pix)
    y1 = float(rows - 1 - margin_pix)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("margin_pix too large; no sampling area remains.")

    x_lin = np.linspace(x0, x1, int(n_u), dtype=np.float64)
    y_lin = np.linspace(y0, y1, int(n_v), dtype=np.float64)

    xx, yy = np.meshgrid(x_lin, y_lin)  # shapes (n_v, n_u)
    x_pix = xx.reshape(-1)
    y_pix = yy.reshape(-1)

    u_norm, v_norm = normalized_pixel_coords(x_pix, y_pix, rows=rows, cols=cols)

    return {
        "x_pix": x_pix,
        "y_pix": y_pix,
        "u_norm": u_norm,
        "v_norm": v_norm,
    }


def make_full_map_pixel_grid(
    *,
    rows: int,
    cols: int,
    downsample: int = 1,
    margin_pix: int = 0,
) -> Dict[str, Any]:
    """
    Create a full (or downsampled) pixel grid for exporting a 2D zodiacal map.

    Parameters
    ----------
    rows, cols : int
        Detector shape.
    downsample : int
        1 => every pixel; 2 => every other pixel, etc.
    margin_pix : int
        Optional margin exclusion.

    Returns
    -------
    dict with:
      x_pix : (n_samples,) float64
      y_pix : (n_samples,) float64
      n_u : int   number of x samples
      n_v : int   number of y samples
      downsample : int

    Notes
    -----
    The returned x_pix/y_pix are flattened in row-major order (v slow, u fast).
    The stage reshapes the returned phi array back into (n_frames, n_v, n_u).
    """
    downsample = max(1, int(downsample))

    xs = np.arange(margin_pix, cols - margin_pix, downsample, dtype=np.int32)
    ys = np.arange(margin_pix, rows - margin_pix, downsample, dtype=np.int32)

    if xs.size == 0 or ys.size == 0:
        raise ValueError("Full-map grid is empty (check margin/downsample).")

    xx, yy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    x_pix = xx.reshape(-1)
    y_pix = yy.reshape(-1)

    return {
        "x_pix": x_pix,
        "y_pix": y_pix,
        "n_u": int(xs.size),
        "n_v": int(ys.size),
        "downsample": downsample,
    }


def normalized_pixel_coords(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to normalized coordinates in [-1,1] centered on detector.

    Parameters
    ----------
    x_pix, y_pix : np.ndarray
        Flattened pixel coordinates.
    rows, cols : int
        Detector shape.

    Returns
    -------
    (u_norm, v_norm) : (np.ndarray, np.ndarray)
        Normalized coordinates where:
          u_norm = (x - (cols-1)/2) / ((cols-1)/2)
          v_norm = (y - (rows-1)/2) / ((rows-1)/2)
    """
    cx = 0.5 * (float(cols) - 1.0)
    cy = 0.5 * (float(rows) - 1.0)
    sx = cx if cx != 0.0 else 1.0
    sy = cy if cy != 0.0 else 1.0
    u_norm = (x_pix - cx) / sx
    v_norm = (y_pix - cy) / sy
    return u_norm.astype(np.float64), v_norm.astype(np.float64)
