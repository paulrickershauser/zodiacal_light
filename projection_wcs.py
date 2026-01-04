from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig
from .logging_utils import _get_logger


def build_nebula_wcs_list(
    *,
    boresight_ra_deg: np.ndarray,
    boresight_dec_deg: np.ndarray,
    boresight_roll_deg: np.ndarray,
    times_utc: Sequence[Any],
    sensor: SensorConfig,
) -> Sequence[Any]:
    """
    Build a per-frame list of WCS objects using NEBULA_WCS.build_wcs_for_observer.

    This is a single source of truth for WCS construction in the Zodiacal Light stage
    and is intended to be reused by both:
      - projection_wcs (pixel -> sky directions)
      - omega_pix (pixel solid angle methods that require WCS)

    Returns
    -------
    Sequence[Any]
        List-like sequence of length n_frames. Each element is a WCS-like object
        supporting at least pixel_to_world(x, y), or an equivalent all_pix2world API.

    Raises
    ------
    ValueError
        If input lengths are inconsistent or returned WCS count mismatches.
    ImportError / Exception
        If NEBULA_WCS cannot be imported or WCS construction fails.
    """
    n_frames = int(len(boresight_ra_deg))
    if len(boresight_dec_deg) != n_frames or len(boresight_roll_deg) != n_frames:
        raise ValueError("Pointing arrays must all have the same length.")
    if len(times_utc) != n_frames:
        raise ValueError("times_utc must have length n_frames (match pointing arrays).")

    # Canonical import path (your repo includes NEBULA_WCS.py).
    from NEBULA_WCS import build_wcs_for_observer  # type: ignore

    track_stub: Dict[str, Any] = {
        "times": list(times_utc),
        "pointing_boresight_ra_deg": np.asarray(boresight_ra_deg, dtype=np.float64),
        "pointing_boresight_dec_deg": np.asarray(boresight_dec_deg, dtype=np.float64),
        "pointing_boresight_roll_deg": np.asarray(boresight_roll_deg, dtype=np.float64),
    }

    wcs_obj = build_wcs_for_observer(track_stub, sensor_config=sensor)

    if isinstance(wcs_obj, list):
        wcs_list = wcs_obj
    else:
        wcs_list = [wcs_obj] * n_frames

    if len(wcs_list) != n_frames:
        raise ValueError(f"build_wcs_for_observer returned {len(wcs_list)} WCS objects, expected {n_frames}.")

    return wcs_list


def compute_sample_radec_deg(
    *,
    boresight_ra_deg: np.ndarray,
    boresight_dec_deg: np.ndarray,
    boresight_roll_deg: np.ndarray,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    times_utc: Sequence[Any],
    sensor: SensorConfig,
    prefer_nebula_wcs: bool = True,
    strict: bool = False,
    wcs_list: Optional[Sequence[Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute ICRS sample directions (RA/Dec) for each frame and each sample pixel.

    This is the *Windows-side* geometry step that turns a sampling grid in detector
    pixels into a set of sky directions that the WSL worker can evaluate.

    WCS-only policy (fail-fast)
    ---------------------------
    This function *only* uses NEBULA's WCS implementation (NEBULA_WCS.build_wcs_for_observer).
    Any historical "tangent-plane fallback" path has been removed by design.

    Parameters
    ----------
    boresight_ra_deg, boresight_dec_deg, boresight_roll_deg : np.ndarray
        Arrays of length n_frames describing the pointing (ICRS) for each frame.
        Roll is passed through to NEBULA_WCS because the WCS construction may depend on it.
    x_pix, y_pix : np.ndarray
        Sample pixel coordinates. Must have the same number of elements.
        Convention: x_pix is column-like, y_pix is row-like (WCS "pixel" ordering).
    times_utc : Sequence[Any]
        Per-frame times. Must have length n_frames. Passed into the WCS builder.
        Items must be whatever NEBULA_WCS expects for 'times' (stage does not coerce types).
    sensor : SensorConfig
        Active sensor configuration. Passed into NEBULA_WCS (sensor_config=...).
    prefer_nebula_wcs : bool
        Must be True. Present only for transition compatibility with older call sites.
    strict : bool
        Deprecated in WCS-only mode. Present only for transition compatibility.
    wcs_list : Sequence[Any] | None
        Optional pre-built per-frame WCS list (length n_frames). If provided, avoids
        rebuilding WCS and guarantees consistency with omega_pix computations.
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    np.ndarray
        RA/Dec degrees with shape (n_frames, n_samples, 2), float64.
        [:, :, 0] is RA(deg), [:, :, 1] is Dec(deg).

    Raises
    ------
    ValueError
        For shape/length mismatches.
    ImportError / AttributeError / Exception
        If NEBULA_WCS cannot be imported or WCS conversion fails.
    """
    lg = _get_logger(logger)

    if not prefer_nebula_wcs:
        raise ValueError(
            "compute_sample_radec_deg(): prefer_nebula_wcs=False is not supported. "
            "This stage is WCS-only (tangent-plane fallback removed)."
        )

    x = np.asarray(x_pix, dtype=np.float64).reshape(-1)
    y = np.asarray(y_pix, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x_pix and y_pix must have the same number of samples.")

    n_frames = int(len(boresight_ra_deg))
    if len(boresight_dec_deg) != n_frames or len(boresight_roll_deg) != n_frames:
        raise ValueError("Pointing arrays must all have the same length.")
    if len(times_utc) != n_frames:
        raise ValueError("times_utc must have length n_frames (match pointing arrays).")

    sample_xy_pix = np.stack([x, y], axis=1)  # (n_samples, 2)

    # WCS-only: any failure should raise with a concrete message.
    radec = try_compute_radec_with_nebula_wcs(
        sample_xy_pix=sample_xy_pix,
        boresight_ra_deg=np.asarray(boresight_ra_deg, dtype=np.float64),
        boresight_dec_deg=np.asarray(boresight_dec_deg, dtype=np.float64),
        boresight_roll_deg=np.asarray(boresight_roll_deg, dtype=np.float64),
        times_utc=times_utc,
        sensor=sensor,
        wcs_list=wcs_list,
        logger=lg,
    )

    lg.debug("Computed sample RA/Dec via NEBULA WCS for %d frames.", n_frames)
    return radec


def try_compute_radec_with_nebula_wcs(
    *,
    sample_xy_pix: np.ndarray,
    boresight_ra_deg: np.ndarray,
    boresight_dec_deg: np.ndarray,
    boresight_roll_deg: np.ndarray,
    times_utc: Sequence[Any],
    sensor: SensorConfig,
    wcs_list: Optional[Sequence[Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute per-frame RA/Dec for detector sample points using NEBULA's WCS.

    No-guesswork behavior
    ---------------------
    - Uses exactly one import path: `from NEBULA_WCS import build_wcs_for_observer`.
    - If import fails, if WCS construction fails, or if pixel->world conversion fails,
      this function raises immediately.
    - No approximations, no alternate geometry paths.

    Parameters
    ----------
    sample_xy_pix : np.ndarray
        Sample pixel coordinates with shape (n_samples, 2).
    boresight_ra_deg, boresight_dec_deg, boresight_roll_deg : np.ndarray
        Pointing arrays of length n_frames.
    times_utc : Sequence[Any]
        Length n_frames; passed into the WCS builder as track_stub["times"].
    sensor : SensorConfig
        Passed into build_wcs_for_observer(..., sensor_config=sensor).
    wcs_list : Sequence[Any] | None
        Optional pre-built per-frame WCS list (length n_frames). If not provided,
        WCS is built via build_nebula_wcs_list(...).

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_samples, 2) float64, with RA/Dec in degrees.

    Raises
    ------
    ValueError
        If shapes/lengths are inconsistent or returned WCS count mismatches.
    ImportError / Exception
        If NEBULA_WCS is unavailable or WCS conversion fails.
    """
    lg = _get_logger(logger)

    if sample_xy_pix.ndim != 2 or sample_xy_pix.shape[1] != 2:
        raise ValueError("sample_xy_pix must have shape (n_samples, 2).")

    n_frames = int(len(boresight_ra_deg))
    if len(boresight_dec_deg) != n_frames or len(boresight_roll_deg) != n_frames:
        raise ValueError("Pointing arrays must all have the same length.")
    if len(times_utc) != n_frames:
        raise ValueError("times_utc must have length n_frames (match pointing arrays).")

    if wcs_list is None:
        wcs_list = build_nebula_wcs_list(
            boresight_ra_deg=boresight_ra_deg,
            boresight_dec_deg=boresight_dec_deg,
            boresight_roll_deg=boresight_roll_deg,
            times_utc=times_utc,
            sensor=sensor,
        )
    else:
        # Fail-fast if a caller passes a mismatched list.
        if len(wcs_list) != n_frames:
            raise ValueError(f"wcs_list has length {len(wcs_list)}, expected n_frames={n_frames}.")

    x = sample_xy_pix[:, 0]
    y = sample_xy_pix[:, 1]

    out = np.full((n_frames, sample_xy_pix.shape[0], 2), np.nan, dtype=np.float64)
    for i, w in enumerate(wcs_list):
        ra, dec = _wcs_pixel_to_world_deg(w, x, y)
        out[i, :, 0] = ra
        out[i, :, 1] = dec

    lg.debug("try_compute_radec_with_nebula_wcs(): success for %d frames.", n_frames)
    return out


def _wcs_pixel_to_world_deg(wcs_obj: Any, x_pix: np.ndarray, y_pix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to (RA, Dec) in degrees using common WCS interfaces.

    Supported patterns (fail-fast; no silent fallbacks)
    --------------------------------------------------
    1) High-level astropy WCS API:
         wcs_obj.pixel_to_world(x, y) -> SkyCoord-like object with .ra/.dec

    2) Low-level astropy WCS API:
         wcs_obj.all_pix2world(x, y, origin) -> (ra, dec)
       OR:
         wcs_obj.wcs.all_pix2world(x, y, origin) -> (ra, dec)

    Notes
    -----
    - We always use origin=0 (Python convention: first pixel is index 0).
    - x_pix/y_pix are treated as (x, y) pixel coordinates, not (row, col).

    Raises
    ------
    AttributeError
        If no supported interface is present.
    """
    # High-level API: pixel_to_world -> SkyCoord
    if hasattr(wcs_obj, "pixel_to_world"):
        sc = wcs_obj.pixel_to_world(x_pix, y_pix)
        if hasattr(sc, "ra") and hasattr(sc, "dec"):
            return np.asarray(sc.ra.deg, dtype=np.float64), np.asarray(sc.dec.deg, dtype=np.float64)

    # Low-level API on WCS object itself
    if hasattr(wcs_obj, "all_pix2world"):
        ra, dec = wcs_obj.all_pix2world(x_pix, y_pix, 0)
        return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)

    # Low-level API via wcs.wcs (Wcsprm)
    if hasattr(wcs_obj, "wcs") and hasattr(wcs_obj.wcs, "all_pix2world"):
        ra, dec = wcs_obj.wcs.all_pix2world(x_pix, y_pix, 0)
        return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)

    raise AttributeError(
        "Unsupported WCS object: cannot find pixel_to_world, all_pix2world, or wcs.all_pix2world."
    )


# Public alias for reuse (e.g., omega_pix.py). Kept separate to avoid breaking
# any internal/private naming assumptions while still providing a stable import.
wcs_pixel_to_world_deg = _wcs_pixel_to_world_deg


# Optional transition helpers (kept for compatibility with earlier drafts).
# Not currently called by the WCS-only pipeline but useful if you later refactor
# to allow either a single static WCS or a per-frame list of WCS objects.

def _apply_single_wcs(
    wcs_obj: Any, x_pix: np.ndarray, y_pix: np.ndarray, n_frames: int
) -> Optional[np.ndarray]:
    """
    Apply one WCS object to sample pixels and replicate across frames.

    Returns None if the WCS object cannot be applied (conversion raises).
    """
    try:
        ra, dec = _wcs_pixel_to_world_deg(wcs_obj, x_pix, y_pix)
        radec = np.stack([ra, dec], axis=-1)                 # (n_samples, 2)
        radec = np.repeat(radec[None, :, :], n_frames, axis=0)  # (n_frames, n_samples, 2)
        return radec
    except Exception:
        return None


def _apply_wcs_list(wcs_list: Sequence[Any], x_pix: np.ndarray, y_pix: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply a per-frame list of WCS objects.

    Returns None if any frame fails.
    """
    n_frames = len(wcs_list)
    n_samples = x_pix.size
    out = np.empty((n_frames, n_samples, 2), dtype=np.float64)

    for i, w in enumerate(wcs_list):
        try:
            ra, dec = _wcs_pixel_to_world_deg(w, x_pix, y_pix)
            out[i, :, 0] = ra
            out[i, :, 1] = dec
        except Exception:
            return None
    return out
