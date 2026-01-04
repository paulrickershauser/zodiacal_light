from __future__ import annotations

from datetime import datetime
from typing import Any, List, Tuple

import numpy as np


def slice_window_frame_data(
    *,
    obs_track: Any,
    start_index: int,
    end_index: int,
    n_frames_expected: int,
) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice per-frame data arrays for a window using inclusive indexing.

    This is a strict, fail-fast slicer for the observer_geometry arrays in
    obs_window_sources.pkl.

    Parameters
    ----------
    obs_track : Any
        Observer TrackDict-like structure from obs_window_sources.pkl for one observer.
        Required structure (dict-like indexing):
            obs_track["observer_geometry"]["state_vectors"]["times"]
            obs_track["observer_geometry"]["state_vectors"]["r_eci_km"]
            obs_track["observer_geometry"]["pointing"]["pointing_boresight_ra_deg"]
            obs_track["observer_geometry"]["pointing"]["pointing_boresight_dec_deg"]
            obs_track["observer_geometry"]["pointing"]["pointing_boresight_roll_deg"]

    start_index, end_index : int
        Inclusive indices into the arrays above.

    n_frames_expected : int
        Window's n_frames value. If >0, we validate that the inclusive slice length
        equals n_frames_expected.

    Returns
    -------
    times_dt : list[datetime]
        Length n_frames.
    boresight_ra_deg : np.ndarray
        Shape (n_frames,), float64.
    boresight_dec_deg : np.ndarray
        Shape (n_frames,), float64.
    boresight_roll_deg : np.ndarray
        Shape (n_frames,), float64.
    r_eci_km : np.ndarray
        Shape (n_frames, 3), float64.

    Raises
    ------
    KeyError
        If required keys are missing.
    ValueError
        If slice length does not match n_frames_expected (when provided) or shapes
        are not as expected.
    """
    og = obs_track["observer_geometry"]
    sv = og["state_vectors"]
    pt = og["pointing"]

    # Inclusive slicing: [start_index, end_index]
    sl = slice(start_index, end_index + 1)

    times_dt = list(sv["times"][sl])
    r_eci_km = np.asarray(sv["r_eci_km"][sl], dtype=np.float64)

    boresight_ra_deg = np.asarray(pt["pointing_boresight_ra_deg"][sl], dtype=np.float64)
    boresight_dec_deg = np.asarray(pt["pointing_boresight_dec_deg"][sl], dtype=np.float64)
    boresight_roll_deg = np.asarray(pt["pointing_boresight_roll_deg"][sl], dtype=np.float64)

    n_frames = len(times_dt)
    if n_frames_expected > 0 and n_frames != n_frames_expected:
        raise ValueError(
            f"Window slice length mismatch: got {n_frames} frames from indices "
            f"[{start_index},{end_index}] but window says n_frames={n_frames_expected}."
        )

    expected_1d = (n_frames,)
    if boresight_ra_deg.shape != expected_1d:
        raise ValueError(
            f"boresight_ra_deg slice has shape {boresight_ra_deg.shape}, expected {expected_1d}."
        )
    if boresight_dec_deg.shape != expected_1d:
        raise ValueError(
            f"boresight_dec_deg slice has shape {boresight_dec_deg.shape}, expected {expected_1d}."
        )
    if boresight_roll_deg.shape != expected_1d:
        raise ValueError(
            f"boresight_roll_deg slice has shape {boresight_roll_deg.shape}, expected {expected_1d}."
        )

    if r_eci_km.shape != (n_frames, 3):
        raise ValueError(f"r_eci_km slice has shape {r_eci_km.shape}, expected {(n_frames,3)}.")

    if not np.all(np.isfinite(boresight_ra_deg)):
        bad = int(np.flatnonzero(~np.isfinite(boresight_ra_deg))[0])
        raise ValueError(f"boresight_ra_deg contains non-finite value at frame {bad}: {boresight_ra_deg[bad]!r}")
    if not np.all(np.isfinite(boresight_dec_deg)):
        bad = int(np.flatnonzero(~np.isfinite(boresight_dec_deg))[0])
        raise ValueError(f"boresight_dec_deg contains non-finite value at frame {bad}: {boresight_dec_deg[bad]!r}")
    if not np.all(np.isfinite(boresight_roll_deg)):
        bad = int(np.flatnonzero(~np.isfinite(boresight_roll_deg))[0])
        raise ValueError(f"boresight_roll_deg contains non-finite value at frame {bad}: {boresight_roll_deg[bad]!r}")

    if not np.all(np.isfinite(r_eci_km)):
        bad_rc = np.argwhere(~np.isfinite(r_eci_km))[0]
        bad_f = int(bad_rc[0])
        bad_c = int(bad_rc[1])
        raise ValueError(
            f"r_eci_km contains non-finite value at frame {bad_f}, component {bad_c}: {r_eci_km[bad_f, bad_c]!r}"
        )

    return times_dt, boresight_ra_deg, boresight_dec_deg, boresight_roll_deg, r_eci_km
