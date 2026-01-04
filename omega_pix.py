from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig
from .logging_utils import _get_logger
from .projection_wcs import build_nebula_wcs_list, wcs_pixel_to_world_deg


OmegaPixValue = Union[float, np.ndarray]


def compute_omega_pix_product(
    *,
    mode: str,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    sensor: SensorConfig,
    boresight_ra_deg: Optional[np.ndarray] = None,
    boresight_dec_deg: Optional[np.ndarray] = None,
    boresight_roll_deg: Optional[np.ndarray] = None,
    times_utc: Optional[Sequence[Any]] = None,
    wcs_list: Optional[Sequence[Any]] = None,
    jacobian_eps_pix: float = 0.5,
    time_dependence: str = "auto",
    invariance_probe_frames: int = 3,
    invariance_rel_tol: float = 1e-6,
    time_invariance_tol_rel: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Compute pixel solid angle Ω_pix in steradians per pixel using one of three modes:
      - analytic_scalar: Ω ≈ (pixel_pitch / focal_length)^2
      - local_jacobian: finite-difference Jacobian of the WCS mapping to unit vectors
      - pixel_corners: spherical polygon area from WCS-mapped pixel corner unit vectors

    Output shape policy:
      - analytic_scalar -> scalar float
      - local_jacobian / pixel_corners -> (n_samples,) if time-invariant else (n_frames, n_samples)

    Time dependence policy (WCS-backed modes)
    ----------------------------------------
    For mode in {"local_jacobian","pixel_corners"}, this function honors `time_dependence`:

      - "per_window": compute Ω_pix once (frame 0) and return (n_samples,)
      - "per_frame" : compute Ω_pix for each frame and return (n_frames, n_samples)
      - "auto"      : probe invariance across `invariance_probe_frames` frames and compare
                      vs frame 0 using `invariance_rel_tol` (max relative difference)

    Backwards compatibility
    -----------------------
    Older callers may pass `time_invariance_tol_rel`. If provided (non-None), it overrides
    `invariance_rel_tol` for the auto decision.

    Returns
    -------
    dict
        {
          "omega_pix_sr": float or ndarray,
          "mode": str,
          "time_dependent": bool,
          "shape_tag": str,
          "units": "sr",
          "params": {...}
        }
    """
    lg = _get_logger(logger)

    mode = str(mode).strip()
    valid_modes = {"analytic_scalar", "local_jacobian", "pixel_corners"}
    if mode not in valid_modes:
        raise ValueError(
            f"compute_omega_pix_product(): unsupported mode={mode!r}; expected one of {sorted(valid_modes)}"
        )

    time_dep_req = str(time_dependence).strip().lower()
    if time_dep_req not in {"auto", "per_window", "per_frame"}:
        raise ValueError(
            "compute_omega_pix_product(): time_dependence must be one of {'auto','per_window','per_frame'} "
            f"(case-insensitive). Got: {time_dependence!r}"
        )

    try:
        n_probe = int(invariance_probe_frames)
    except Exception as e:
        raise TypeError("compute_omega_pix_product(): invariance_probe_frames must be int-like") from e
    if n_probe < 1:
        raise ValueError("compute_omega_pix_product(): invariance_probe_frames must be >= 1")

    # Legacy alias: if provided, treat as the authoritative tolerance for auto gating.
    if time_invariance_tol_rel is not None:
        invariance_rel_tol = float(time_invariance_tol_rel)

    tol_rel = float(invariance_rel_tol)
    if (not np.isfinite(tol_rel)) or tol_rel < 0.0:
        raise ValueError("compute_omega_pix_product(): invariance_rel_tol must be finite and >= 0")

    x = np.asarray(x_pix, dtype=np.float64).reshape(-1)
    y = np.asarray(y_pix, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("compute_omega_pix_product(): x_pix and y_pix must have the same number of samples.")
    if x.size == 0:
        raise ValueError("compute_omega_pix_product(): x_pix/y_pix must be non-empty.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("compute_omega_pix_product(): x_pix/y_pix must be finite.")

    # --- Analytic mode: scalar only ---
    if mode == "analytic_scalar":
        omega_scalar = _analytic_omega_pix_sr_from_sensor(sensor)
        product: Dict[str, Any] = {
            "omega_pix_sr": float(omega_scalar),
            "mode": mode,
            "time_dependent": False,
            "shape_tag": "scalar",
            "units": "sr",
            "params": {
                "analytic_source": "pitch_over_focal",
            },
        }
        lg.debug("Ω_pix analytic_scalar: %.6e sr/pix", omega_scalar)
        return product

    # --- WCS-backed modes require pointing and time info ---
    if boresight_ra_deg is None or boresight_dec_deg is None or boresight_roll_deg is None or times_utc is None:
        raise ValueError(f"compute_omega_pix_product(): mode={mode!r} requires boresight_* arrays and times_utc.")

    ra_b = np.asarray(boresight_ra_deg, dtype=np.float64).reshape(-1)
    dec_b = np.asarray(boresight_dec_deg, dtype=np.float64).reshape(-1)
    roll_b = np.asarray(boresight_roll_deg, dtype=np.float64).reshape(-1)
    n_frames = int(ra_b.size)
    if dec_b.size != n_frames or roll_b.size != n_frames:
        raise ValueError("compute_omega_pix_product(): boresight arrays must have the same length.")
    if len(times_utc) != n_frames:
        raise ValueError("compute_omega_pix_product(): times_utc must have length n_frames.")

    if not np.isfinite(ra_b).all() or not np.isfinite(dec_b).all() or not np.isfinite(roll_b).all():
        raise ValueError("compute_omega_pix_product(): boresight arrays must be finite.")

    if wcs_list is None:
        wcs_list = build_nebula_wcs_list(
            boresight_ra_deg=ra_b,
            boresight_dec_deg=dec_b,
            boresight_roll_deg=roll_b,
            times_utc=times_utc,
            sensor=sensor,
        )
    else:
        if len(wcs_list) != n_frames:
            raise ValueError(f"compute_omega_pix_product(): wcs_list length={len(wcs_list)} != n_frames={n_frames}")

    # Always compute frame-0 once (used for per_window and as the auto reference).
    omega0 = _compute_omega_pix_sr_for_frame(
        mode=mode,
        wcs_obj=wcs_list[0],
        x_pix=x,
        y_pix=y,
        jacobian_eps_pix=jacobian_eps_pix,
    )

    time_dep_eff = time_dep_req
    probe_indices = [0]
    max_rel = 0.0

    if n_frames <= 1:
        omega_out = omega0
        time_dep_eff = "per_window"

    elif time_dep_req == "per_window":
        omega_out = omega0

    elif time_dep_req == "per_frame":
        omega_out = np.full((n_frames, x.size), np.nan, dtype=np.float64)
        for i in range(n_frames):
            omega_out[i, :] = _compute_omega_pix_sr_for_frame(
                mode=mode,
                wcs_obj=wcs_list[i],
                x_pix=x,
                y_pix=y,
                jacobian_eps_pix=jacobian_eps_pix,
            )

    else:
        # auto: probe invariance across a subset of frames.
        n_probe_eff = min(n_frames, max(2, n_probe))
        probe_indices = np.unique(np.linspace(0, n_frames - 1, num=n_probe_eff, dtype=int)).tolist()

        max_rel = 0.0
        for idx in probe_indices[1:]:
            omega_i = _compute_omega_pix_sr_for_frame(
                mode=mode,
                wcs_obj=wcs_list[idx],
                x_pix=x,
                y_pix=y,
                jacobian_eps_pix=jacobian_eps_pix,
            )
            max_rel = max(max_rel, float(_max_relative_difference(omega0, omega_i)))

        if max_rel <= tol_rel:
            omega_out = omega0
            time_dep_eff = "per_window"
        else:
            omega_out = np.full((n_frames, x.size), np.nan, dtype=np.float64)
            for i in range(n_frames):
                omega_out[i, :] = _compute_omega_pix_sr_for_frame(
                    mode=mode,
                    wcs_obj=wcs_list[i],
                    x_pix=x,
                    y_pix=y,
                    jacobian_eps_pix=jacobian_eps_pix,
                )
            time_dep_eff = "per_frame"

        lg.debug(
            "Ω_pix %s auto gate: probe=%s max_rel=%.3e tol=%.3e -> time_dependence=%s",
            mode,
            probe_indices,
            max_rel,
            tol_rel,
            time_dep_eff,
        )

    if isinstance(omega_out, np.ndarray) and omega_out.ndim == 2:
        shape_tag = "n_frames_n_samples"
        time_dependent = True
    else:
        shape_tag = "n_samples"
        time_dependent = False

    # Final validation: finite and positive
    if isinstance(omega_out, np.ndarray):
        if not np.isfinite(omega_out).all():
            raise ValueError(f"compute_omega_pix_product(): computed omega_pix_sr has non-finite values (mode={mode}).")
        if not (omega_out > 0).all():
            raise ValueError(f"compute_omega_pix_product(): computed omega_pix_sr has non-positive values (mode={mode}).")
    else:
        if not np.isfinite(omega_out) or omega_out <= 0:
            raise ValueError(f"compute_omega_pix_product(): computed omega_pix_sr is invalid (mode={mode}).")

    product = {
        "omega_pix_sr": omega_out,
        "mode": mode,
        "time_dependent": bool(time_dep_eff == "per_frame"),
        "shape_tag": shape_tag,
        "units": "sr",
        "params": {
            "jacobian_eps_pix": float(jacobian_eps_pix),
            "time_dependence_requested": str(time_dep_req),
            "time_dependence_effective": str(time_dep_eff),
            "invariance_probe_frames": int(n_probe),
            "invariance_rel_tol": float(tol_rel),
            "invariance_probe_indices": probe_indices,
            "invariance_max_rel": float(max_rel),
        },
    }
    return product



# -----------------------------
# Mode implementations (per-frame)
# -----------------------------

def _compute_omega_pix_sr_for_frame(
    *,
    mode: str,
    wcs_obj: Any,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    jacobian_eps_pix: float,
) -> np.ndarray:
    if mode == "local_jacobian":
        return _omega_pix_sr_local_jacobian(
            wcs_obj=wcs_obj,
            x_pix=x_pix,
            y_pix=y_pix,
            eps_pix=float(jacobian_eps_pix),
        )
    if mode == "pixel_corners":
        return _omega_pix_sr_pixel_corners(
            wcs_obj=wcs_obj,
            x_pix=x_pix,
            y_pix=y_pix,
        )
    raise ValueError(f"_compute_omega_pix_sr_for_frame(): unsupported mode={mode!r}")


def _omega_pix_sr_local_jacobian(*, wcs_obj: Any, x_pix: np.ndarray, y_pix: np.ndarray, eps_pix: float) -> np.ndarray:
    """
    Ω ≈ || (d r̂ / dx) × (d r̂ / dy) || where r̂ is the unit sky direction vector.
    Finite differences use step eps_pix in pixel units.
    """
    if eps_pix <= 0 or not np.isfinite(eps_pix):
        raise ValueError("_omega_pix_sr_local_jacobian(): eps_pix must be finite and > 0.")

    # Sample at +/- eps in x and y
    ra_xp, dec_xp = wcs_pixel_to_world_deg(wcs_obj, x_pix + eps_pix, y_pix)
    ra_xm, dec_xm = wcs_pixel_to_world_deg(wcs_obj, x_pix - eps_pix, y_pix)
    ra_yp, dec_yp = wcs_pixel_to_world_deg(wcs_obj, x_pix, y_pix + eps_pix)
    ra_ym, dec_ym = wcs_pixel_to_world_deg(wcs_obj, x_pix, y_pix - eps_pix)

    rxp = _radec_deg_to_unitvec(ra_xp, dec_xp)
    rxm = _radec_deg_to_unitvec(ra_xm, dec_xm)
    ryp = _radec_deg_to_unitvec(ra_yp, dec_yp)
    rym = _radec_deg_to_unitvec(ra_ym, dec_ym)

    drdx = (rxp - rxm) / (2.0 * eps_pix)
    drdy = (ryp - rym) / (2.0 * eps_pix)

    cross = np.cross(drdx, drdy)
    omega = np.linalg.norm(cross, axis=1)
    return omega.astype(np.float64)


def _omega_pix_sr_pixel_corners(*, wcs_obj: Any, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
    """
    Compute pixel solid angle by mapping the four pixel corners to unit vectors
    and computing the spherical quadrilateral area via two spherical triangles.
    Pixel center convention: corners are at (x±0.5, y±0.5).
    """
    x0 = x_pix - 0.5
    x1 = x_pix + 0.5
    y0 = y_pix - 0.5
    y1 = y_pix + 0.5

    # Corner order (counterclockwise in pixel coordinates):
    # c0: (x0,y0), c1: (x1,y0), c2: (x1,y1), c3: (x0,y1)
    ra0, dec0 = wcs_pixel_to_world_deg(wcs_obj, x0, y0)
    ra1, dec1 = wcs_pixel_to_world_deg(wcs_obj, x1, y0)
    ra2, dec2 = wcs_pixel_to_world_deg(wcs_obj, x1, y1)
    ra3, dec3 = wcs_pixel_to_world_deg(wcs_obj, x0, y1)

    c0 = _radec_deg_to_unitvec(ra0, dec0)
    c1 = _radec_deg_to_unitvec(ra1, dec1)
    c2 = _radec_deg_to_unitvec(ra2, dec2)
    c3 = _radec_deg_to_unitvec(ra3, dec3)

    # Split quad into triangles (c0,c1,c2) and (c0,c2,c3)
    a1 = _spherical_triangle_area_unit_vectors(c0, c1, c2)
    a2 = _spherical_triangle_area_unit_vectors(c0, c2, c3)
    omega = a1 + a2
    return omega.astype(np.float64)


# -----------------------------
# Geometry helpers
# -----------------------------

def _analytic_omega_pix_sr_from_sensor(sensor: SensorConfig) -> float:
    """
    Geometry-based approximation: Ω ≈ (pixel_pitch / focal_length)^2.
    """
    if not hasattr(sensor, "pixel_pitch") or not hasattr(sensor, "focal_length"):
        raise ValueError("SensorConfig must provide pixel_pitch and focal_length for analytic Ω_pix.")

    pitch = float(sensor.pixel_pitch)
    fl = float(sensor.focal_length)

    if not np.isfinite(pitch) or not np.isfinite(fl) or pitch <= 0 or fl <= 0:
        raise ValueError(f"Invalid sensor geometry: pixel_pitch={pitch}, focal_length={fl}")

    omega = (pitch / fl) ** 2
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError(f"Computed analytic omega_pix_sr invalid: {omega}")
    return float(omega)


def _radec_deg_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """
    Convert RA/Dec degrees to unit vectors in ICRS Cartesian coordinates.
    Output shape: (n, 3)
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))

    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    v = np.stack([x, y, z], axis=1)
    # Normalize defensively (should already be unit-length).
    n = np.linalg.norm(v, axis=1)
    v = v / n[:, None]
    return v


def _spherical_triangle_area_unit_vectors(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Robust spherical triangle area on the unit sphere using a vector formula:
      area = 2 * atan2(|det(a,b,c)|, 1 + a·b + b·c + c·a)

    Returns area in steradians, shape (n,).
    """
    # Scalar triple product det = a · (b × c)
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    det_abs = np.abs(det)

    ab = np.einsum("ij,ij->i", a, b)
    bc = np.einsum("ij,ij->i", b, c)
    ca = np.einsum("ij,ij->i", c, a)

    denom = 1.0 + ab + bc + ca
    # Guard against negative/zero denom due to numerical issues
    denom = np.maximum(denom, 1e-15)

    area = 2.0 * np.arctan2(det_abs, denom)
    return area


def _max_relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute max relative difference between two positive arrays a and b:
      max(|a-b| / max(a, tiny))
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("_max_relative_difference(): shape mismatch.")
    tiny = 1e-30
    denom = np.maximum(np.abs(a), tiny)
    rel = np.abs(a - b) / denom
    return float(np.max(rel))
