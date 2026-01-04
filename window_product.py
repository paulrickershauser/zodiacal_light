"""
stage/window_product.py

Build per-window Zodiacal Light (ZL) products for obs_zodiacal_light.pkl.

What this module owns
---------------------
This module creates the final *per-window* dictionary that the top-level stage
will write into the standalone ZL pickle.

It is responsible for:
1) Slicing the observer geometry/pointing arrays to the requested window
   using inclusive [start_index, end_index] indexing.
2) Defining the detector sampling grid (pixel locations) used for WSL/m4opt
   background evaluation and for Windows-side polynomial fitting.
3) Computing sample directions on the sky (ICRS RA/Dec) using NEBULA WCS only
   (strict; no tangent-plane fallback).
4) Executing the WSL worker (via stage/wslexec → bridge) and validating the
   returned arrays.
5) Storing three unit conventions for the same underlying quantity:
   - per_pixel   : ph m^-2 s^-1 pix^-1   (backend output)
   - per_sr      : ph m^-2 s^-1 sr^-1    (derived by dividing by Ω_pix)
   - per_arcsec2 : ph m^-2 s^-1 arcsec^-2 (derived from per_sr)

Fail-fast policy
----------------
This module does not attempt to "figure out" missing wiring. It raises
immediately if required inputs are missing or inconsistent:
- missing observer_geometry fields
- missing roll / boresight arrays
- mismatched array lengths
- missing WSL response arrays
- shape mismatches between requested samples and responses

Pickle naming is authoritative
------------------------------
This module treats the keys present in obs_window_sources.pkl as the single
source of truth. It does not accept alternate key spellings.

In particular, observer position uses:
    obs_track["observer_geometry"]["state_vectors"]["r_eci_km"]

If the pickle does not provide this exact key, execution raises immediately.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR, SensorConfig
from Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG import (
    ARCSEC2_TO_SR,
    ZODIACAL_LIGHT_CONFIG,
    ZodiacalLightConfig,
)

from Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_LIGHT_IO import read_payload, write_payload

from .bandpass import build_bandpass_dict
from .fitting import fit_plane3, fit_quad6
from .omega_pix import compute_omega_pix_product
from .projection_wcs import build_nebula_wcs_list, compute_sample_radec_deg
from .slicing import slice_window_frame_data
from .sampling import make_full_map_pixel_grid, make_sample_pixel_grid, normalized_pixel_coords
from .wslexec import make_tmp_bases, invoke_wsl_worker


OmegaPixValue = Union[float, np.ndarray]


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Type-check helper: enforce that a value is a dict."""
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict, got {type(obj).__name__}")
    return obj


def _req(d: Dict[str, Any], k: str) -> Any:
    """Fail-fast helper: require key k in dict d."""
    if k not in d:
        raise KeyError(f"Missing required key: {k!r}")
    return d[k]


def _times_to_utc_iso(times: Sequence[datetime]) -> List[str]:
    """
    Convert tz-aware datetimes to UTC ISO strings with a 'Z' suffix.

    This stage treats time as authoritative and requires tz-aware datetimes.
    """
    out: List[str] = []
    for t in times:
        if not isinstance(t, datetime):
            raise TypeError(f"Expected datetime entries in times, got {type(t).__name__}")
        if t.tzinfo is None:
            raise ValueError("Encountered tz-naive datetime; expected tz-aware UTC datetimes.")
        t_utc = t.astimezone(timezone.utc)
        s = t_utc.isoformat()
        if s.endswith("+00:00"):
            s = s[:-6] + "Z"
        out.append(s)
    return out


def _validate_sensor_name(sensor_name_from_pickle: Optional[str], *, sensor: SensorConfig) -> None:
    """
    Enforce that the pickle's sensor_name (if present) matches ACTIVE_SENSOR.name.
    """
    if sensor_name_from_pickle is None:
        return
    if str(sensor_name_from_pickle) != str(sensor.name):
        raise RuntimeError(
            "Sensor mismatch between scene metadata and ACTIVE_SENSOR.\n"
            f"  pickle sensor_name={sensor_name_from_pickle!r}\n"
            f"  ACTIVE_SENSOR.name={sensor.name!r}"
        )


def _eval_plane3(coeffs: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Evaluate plane model:
        y = c0 + c1*u + c2*v

    coeffs: (n_frames, 3)
    u, v  : (n_pts,)
    returns: (n_frames, n_pts)
    """
    c = np.asarray(coeffs, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"_eval_plane3(): coeffs must have shape (n_frames,3); got {c.shape}")
    u_ = np.asarray(u, dtype=np.float64).reshape(-1)
    v_ = np.asarray(v, dtype=np.float64).reshape(-1)
    return c[:, 0:1] + c[:, 1:2] * u_[None, :] + c[:, 2:3] * v_[None, :]


def _eval_quad6(coeffs: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Evaluate quadratic model:
        y = c0 + c1*u + c2*v + c3*u^2 + c4*u*v + c5*v^2

    coeffs: (n_frames, 6)
    u, v  : (n_pts,)
    returns: (n_frames, n_pts)
    """
    c = np.asarray(coeffs, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 6:
        raise ValueError(f"_eval_quad6(): coeffs must have shape (n_frames,6); got {c.shape}")
    u_ = np.asarray(u, dtype=np.float64).reshape(-1)
    v_ = np.asarray(v, dtype=np.float64).reshape(-1)

    u2 = u_ * u_
    v2 = v_ * v_
    uv = u_ * v_

    return (
        c[:, 0:1]
        + c[:, 1:2] * u_[None, :]
        + c[:, 2:3] * v_[None, :]
        + c[:, 3:4] * u2[None, :]
        + c[:, 4:5] * uv[None, :]
        + c[:, 5:6] * v2[None, :]
    )


def _run_wsl_for_radec_samples(
    *,
    obs_name: str,
    window_index: int,
    suffix: str,
    times_utc_iso: List[str],
    omega_pix: Dict[str, Any],
    bandpass: Dict[str, Any],
    sample_radec_deg: np.ndarray,     # (n_frames, n_samples, 2)
    observer_eci_xyz_km: np.ndarray,  # (n_frames, 3)
    cfg: ZodiacalLightConfig,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Path, Path]:
    """
    Write request payload, invoke WSL worker, and read response payload.

    Contract (current backend)
    --------------------------
    Response arrays must contain:
        phi_ph_m2_s_pix : (n_frames, n_samples)
    with units ph m^-2 s^-1 pix^-1.

    Ω_pix request contract
    ----------------------
    - If omega_pix["shape_tag"] == "scalar":
        write JSON meta key "omega_pix_sr_scalar": float
    - Else:
        write NPZ array "omega_pix_sr": ndarray with shape either:
          (n_samples,) or (n_frames, n_samples)

    Returns
    -------
    resp_meta, resp_arrays, req_base, resp_base
        Returned bases are Windows Paths without extensions (for provenance/debug).

    Raises
    ------
    KeyError, ValueError, FileNotFoundError
        If any required component is missing or inconsistent.
    """
    req_base, resp_base = make_tmp_bases(
        obs_name=str(obs_name),
        window_index=int(window_index),
        suffix=str(suffix),
        cfg=cfg,
    )

    # Fail-fast: validate request array shapes before crossing into WSL.
    sample_arr = np.asarray(sample_radec_deg, dtype=np.float64)
    if sample_arr.ndim != 3 or sample_arr.shape[2] != 2:
        raise ValueError(
            "sample_radec_deg must have shape (n_frames, n_samples, 2).\n"
            f"  got shape={sample_arr.shape}"
        )
    if sample_arr.shape[0] != len(times_utc_iso):
        raise ValueError(
            "sample_radec_deg n_frames must match times_utc_iso length.\n"
            f"  sample_radec_deg n_frames={sample_arr.shape[0]}\n"
            f"  len(times_utc_iso)={len(times_utc_iso)}"
        )

    obs_arr = np.asarray(observer_eci_xyz_km, dtype=np.float64)
    if obs_arr.ndim != 2 or obs_arr.shape[1] != 3:
        raise ValueError(
            "observer_eci_xyz_km must have shape (n_frames, 3).\n"
            f"  got shape={obs_arr.shape}"
        )
    if obs_arr.shape[0] != len(times_utc_iso):
        raise ValueError(
            "observer_eci_xyz_km n_frames must match times_utc_iso length.\n"
            f"  observer_eci_xyz_km n_frames={obs_arr.shape[0]}\n"
            f"  len(times_utc_iso)={len(times_utc_iso)}"
        )

    # Ω_pix validation and request placement.
    if not isinstance(omega_pix, dict):
        raise TypeError(f"omega_pix must be a dict product, got {type(omega_pix).__name__}")

    if "shape_tag" not in omega_pix or "mode" not in omega_pix or "omega_pix_sr" not in omega_pix:
        raise KeyError("omega_pix product missing required keys: expected at least {mode, shape_tag, omega_pix_sr}.")

    shape_tag = str(omega_pix["shape_tag"])
    omega_val = omega_pix["omega_pix_sr"]

    n_frames = int(sample_arr.shape[0])
    n_samples = int(sample_arr.shape[1])

    req_meta: Dict[str, Any] = {
        "schema_version": cfg.io.schema_version,
        "observer_name": str(obs_name),
        "window_index": int(window_index),
        "times_utc_iso": times_utc_iso,
        "bandpass": bandpass,
        # Ω_pix provenance
        "omega_pix_mode": str(omega_pix.get("mode", "")),
        "omega_pix_shape_tag": shape_tag,
        "omega_pix_time_dependent": bool(omega_pix.get("time_dependent", False)),
        "omega_pix_units": str(omega_pix.get("units", "sr")),
        "omega_pix_params": omega_pix.get("params", {}),
    }
    req_arrays: Dict[str, np.ndarray] = {
        "sample_radec_deg": sample_arr,
        "observer_eci_xyz_km": obs_arr,
    }

    if shape_tag == "scalar":
        req_meta["omega_pix_sr_scalar"] = float(omega_val)
    else:
        omega_arr = np.asarray(omega_val, dtype=np.float64)

        if shape_tag == "n_samples":
            if omega_arr.shape != (n_samples,):
                raise ValueError(
                    "omega_pix_sr (n_samples) shape mismatch.\n"
                    f"  expected={(n_samples,)}\n"
                    f"  got={omega_arr.shape}"
                )
        elif shape_tag == "n_frames_n_samples":
            if omega_arr.shape != (n_frames, n_samples):
                raise ValueError(
                    "omega_pix_sr (n_frames,n_samples) shape mismatch.\n"
                    f"  expected={(n_frames, n_samples)}\n"
                    f"  got={omega_arr.shape}"
                )
        else:
            raise ValueError(f"Unsupported omega_pix.shape_tag={shape_tag!r}")

        req_arrays["omega_pix_sr"] = omega_arr

    write_payload(req_base, meta=req_meta, arrays=req_arrays)
    invoke_wsl_worker(request_base_win=req_base, response_base_win=resp_base, cfg=cfg)

    resp_meta, resp_arrays = read_payload(resp_base)

    if "phi_ph_m2_s_pix" not in resp_arrays:
        raise KeyError(
            "WSL response missing required array 'phi_ph_m2_s_pix'. "
            f"Available arrays: {sorted(resp_arrays.keys())}"
        )

    phi = np.asarray(resp_arrays["phi_ph_m2_s_pix"], dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError(f"phi_ph_m2_s_pix must be 2D (n_frames,n_samples), got shape {phi.shape}")

    if phi.shape != (n_frames, n_samples):
        raise ValueError(
            "phi_ph_m2_s_pix shape mismatch.\n"
            f"  expected={(n_frames, n_samples)}\n"
            f"  got={phi.shape}"
        )

    return resp_meta, resp_arrays, req_base, resp_base


def build_window_zodi_product(
    *,
    obs_name: str,
    obs_track: Dict[str, Any],
    window: Dict[str, Any],
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> Dict[str, Any]:
    """
    Build the ZL product for a single observer window.
    """
    obs_track = _as_dict(obs_track)
    window = _as_dict(window)

    # Enforce ACTIVE_SENSOR-only operation.
    _validate_sensor_name(obs_track.get("sensor_name", None), sensor=ACTIVE_SENSOR)
    sensor = ACTIVE_SENSOR

    # Required window fields.
    window_index = int(_req(window, "window_index"))
    start_index = int(_req(window, "start_index"))
    end_index = int(_req(window, "end_index"))
    n_frames_expected = int(_req(window, "n_frames"))

    # Slice frame arrays (single source of truth).
    times_dt, boresight_ra_deg, boresight_dec_deg, boresight_roll_deg, r_eci_km = slice_window_frame_data(
        obs_track=obs_track,
        start_index=start_index,
        end_index=end_index,
        n_frames_expected=n_frames_expected,
    )
    times_utc_iso = _times_to_utc_iso(times_dt)
    n_frames = len(times_utc_iso)

    # Build WCS list once per window and reuse across:
    #   - sample RA/Dec
    #   - Ω_pix (jacobian/corners)
    wcs_list = build_nebula_wcs_list(
        boresight_ra_deg=boresight_ra_deg,
        boresight_dec_deg=boresight_dec_deg,
        boresight_roll_deg=boresight_roll_deg,
        times_utc=times_dt,
        sensor=sensor,
    )

    # Bandpass request dict (strict, validated within build_bandpass_dict).
    catalog_name = obs_track.get("catalog_name", None)
    catalog_band = obs_track.get("catalog_band", None)
    bandpass = build_bandpass_dict(
        catalog_name=catalog_name,
        catalog_band=catalog_band,
        cfg=cfg,
    )

    # Sampling grid on detector (canonical helper).
    if len(cfg.field.sample_grid) != 2:
        raise ValueError(
            "cfg.field.sample_grid must have length 2: [n_u, n_v].\n"
            f"  got={cfg.field.sample_grid!r}"
        )
    nu, nv = int(cfg.field.sample_grid[0]), int(cfg.field.sample_grid[1])
    if nu < 2 or nv < 2:
        raise ValueError(
            "cfg.field.sample_grid must be >=2 in each dimension for stable fitting.\n"
            f"  got={cfg.field.sample_grid!r}"
        )

    samp = make_sample_pixel_grid(
        rows=int(sensor.rows),
        cols=int(sensor.cols),
        n_u=nu,
        n_v=nv,
        margin_pix=int(cfg.field.sample_margin_pix),
    )
    x_samp = np.asarray(samp["x_pix"], dtype=np.float64)
    y_samp = np.asarray(samp["y_pix"], dtype=np.float64)

    if bool(cfg.field.normalized_uv):
        u_samp = np.asarray(samp["u_norm"], dtype=np.float64)
        v_samp = np.asarray(samp["v_norm"], dtype=np.float64)
        uv_basis = "normalized_uv"
    else:
        u_samp = np.asarray(x_samp, dtype=np.float64)
        v_samp = np.asarray(y_samp, dtype=np.float64)
        uv_basis = "pixel_xy"

    # Ω_pix product (supports analytic_scalar, local_jacobian, pixel_corners)
    omega_pix = compute_omega_pix_product(
       mode=str(cfg.omega_pix.mode),
       x_pix=x_samp,
       y_pix=y_samp,
       sensor=sensor,
       boresight_ra_deg=boresight_ra_deg,
       boresight_dec_deg=boresight_dec_deg,
       boresight_roll_deg=boresight_roll_deg,
       times_utc=times_dt,
       wcs_list=wcs_list,
       jacobian_eps_pix=float(cfg.omega_pix.jacobian_eps_pix),
       time_dependence=str(cfg.omega_pix.time_dependence),
       invariance_probe_frames=int(cfg.omega_pix.invariance_probe_frames),
       invariance_rel_tol=float(cfg.omega_pix.invariance_rel_tol),
   )
    omega_val: OmegaPixValue = omega_pix["omega_pix_sr"]

    # Compute per-frame sample directions on sky (strict WCS-only).
    sample_radec_deg = compute_sample_radec_deg(
        boresight_ra_deg=boresight_ra_deg,
        boresight_dec_deg=boresight_dec_deg,
        boresight_roll_deg=boresight_roll_deg,
        times_utc=times_dt,
        x_pix=x_samp,
        y_pix=y_samp,
        sensor=sensor,
        strict=True,
        wcs_list=wcs_list,
    )  # (n_frames, n_samples, 2)

    # Run WSL backend for sample grid.
    resp_meta_s, resp_arrays_s, req_base_s, resp_base_s = _run_wsl_for_radec_samples(
        obs_name=str(obs_name),
        window_index=window_index,
        suffix="samples",
        times_utc_iso=times_utc_iso,
        omega_pix=omega_pix,
        bandpass=bandpass,
        sample_radec_deg=sample_radec_deg,
        observer_eci_xyz_km=r_eci_km,
        cfg=cfg,
    )
    phi_per_pixel_samples = np.asarray(resp_arrays_s["phi_ph_m2_s_pix"], dtype=np.float64)  # (n_frames, n_samples)

    # Unit conversions (broadcast-safe).
    phi_per_sr_samples = phi_per_pixel_samples / omega_val
    phi_per_arcsec2_samples = phi_per_sr_samples * float(ARCSEC2_TO_SR)

    # Polynomial fits: Strategy A
    # - Fit per_sr and per_pixel
    # - Derive per_arcsec2 from per_sr via constant conversion
    models: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}

    if "plane3" in cfg.field.models_to_store:
        coeffs_sr, rms_sr = fit_plane3(phi_per_sr_samples, u_samp, v_samp)
        coeffs_px, rms_px = fit_plane3(phi_per_pixel_samples, u_samp, v_samp)
        coeffs_a2 = coeffs_sr * float(ARCSEC2_TO_SR)

        models["plane3"] = {
            "uv_basis": uv_basis,
            "coeffs_per_sr": coeffs_sr,
            "coeffs_per_pixel": coeffs_px,
            "coeffs_per_arcsec2": coeffs_a2,
        }
        diagnostics["plane3_rms_per_sr"] = rms_sr
        diagnostics["plane3_rms_per_pixel"] = rms_px

    if "quad6" in cfg.field.models_to_store:
        coeffs_sr, rms_sr = fit_quad6(phi_per_sr_samples, u_samp, v_samp)
        coeffs_px, rms_px = fit_quad6(phi_per_pixel_samples, u_samp, v_samp)
        coeffs_a2 = coeffs_sr * float(ARCSEC2_TO_SR)

        models["quad6"] = {
            "uv_basis": uv_basis,
            "coeffs_per_sr": coeffs_sr,
            "coeffs_per_pixel": coeffs_px,
            "coeffs_per_arcsec2": coeffs_a2,
        }
        diagnostics["quad6_rms_per_sr"] = rms_sr
        diagnostics["quad6_rms_per_pixel"] = rms_px

    # Optional 2D map export (downsampled).
    map2d: Optional[Dict[str, Any]] = None
    if bool(cfg.field.export_map2d):
        ds = int(cfg.field.map2d_downsample)
        if ds <= 0:
            raise ValueError("cfg.field.map2d_downsample must be >= 1.")

        grid = make_full_map_pixel_grid(
            rows=int(sensor.rows),
            cols=int(sensor.cols),
            downsample=ds,
            margin_pix=0,
        )
        x_map = np.asarray(grid["x_pix"], dtype=np.float64)
        y_map = np.asarray(grid["y_pix"], dtype=np.float64)

        # Axes for output/debugging (sorted unique values).
        x_axis = np.unique(x_map)
        y_axis = np.unique(y_map)

        ny = int(grid["n_v"])
        nx = int(grid["n_u"])

        # Choose best available model for map evaluation.
        model_name: Optional[str] = None
        if "quad6" in models:
            model_name = "quad6"
        elif "plane3" in models:
            model_name = "plane3"

        if model_name is not None:
            # Map grid UV coordinates consistent with chosen uv_basis.
            if uv_basis == "normalized_uv":
                u_map, v_map = normalized_pixel_coords(
                    x_map,
                    y_map,
                    rows=int(sensor.rows),
                    cols=int(sensor.cols),
                )
            else:
                u_map = np.asarray(x_map, dtype=np.float64)
                v_map = np.asarray(y_map, dtype=np.float64)

            if model_name == "quad6":
                phi_map_per_sr = _eval_quad6(models["quad6"]["coeffs_per_sr"], u_map, v_map)
                phi_map_per_pixel = _eval_quad6(models["quad6"]["coeffs_per_pixel"], u_map, v_map)
            else:
                phi_map_per_sr = _eval_plane3(models["plane3"]["coeffs_per_sr"], u_map, v_map)
                phi_map_per_pixel = _eval_plane3(models["plane3"]["coeffs_per_pixel"], u_map, v_map)

            phi_map_per_sr_2d = phi_map_per_sr.reshape(n_frames, ny, nx)
            phi_map_per_pixel_2d = phi_map_per_pixel.reshape(n_frames, ny, nx)
            phi_map_per_arcsec2_2d = phi_map_per_sr_2d * float(ARCSEC2_TO_SR)

            map2d = {
                "downsample": ds,
                "x_pix_axis": x_axis,
                "y_pix_axis": y_axis,
                "phi_per_pixel": phi_map_per_pixel_2d,
                "phi_per_sr": phi_map_per_sr_2d,
                "phi_per_arcsec2": phi_map_per_arcsec2_2d,
                "source": "model_eval",
                "model": str(model_name),
                "wsl_response_meta": None,
                "request_base_win": None,
                "response_base_win": None,
            }
        else:
            # Fallback: no models stored, so compute map directions and call WSL for the map grid.
            map_radec_deg = compute_sample_radec_deg(
                boresight_ra_deg=boresight_ra_deg,
                boresight_dec_deg=boresight_dec_deg,
                boresight_roll_deg=boresight_roll_deg,
                times_utc=times_dt,
                x_pix=x_map,
                y_pix=y_map,
                sensor=sensor,
                strict=True,
                wcs_list=wcs_list,
            )  # (n_frames, n_map_samples, 2)

            omega_pix_map = compute_omega_pix_product(
                mode=str(cfg.omega_pix.mode),
                x_pix=x_map,
                y_pix=y_map,
                sensor=sensor,
                boresight_ra_deg=boresight_ra_deg,
                boresight_dec_deg=boresight_dec_deg,
                boresight_roll_deg=boresight_roll_deg,
                times_utc=times_dt,
                wcs_list=wcs_list,
                jacobian_eps_pix=float(cfg.omega_pix.jacobian_eps_pix),
                time_dependence=str(cfg.omega_pix.time_dependence),
                invariance_probe_frames=int(cfg.omega_pix.invariance_probe_frames),
                invariance_rel_tol=float(cfg.omega_pix.invariance_rel_tol),
            )

            omega_map_val: OmegaPixValue = omega_pix_map["omega_pix_sr"]

            resp_meta_m, resp_arrays_m, req_base_m, resp_base_m = _run_wsl_for_radec_samples(
                obs_name=str(obs_name),
                window_index=window_index,
                suffix="map2d",
                times_utc_iso=times_utc_iso,
                omega_pix=omega_pix_map,
                bandpass=bandpass,
                sample_radec_deg=map_radec_deg,
                observer_eci_xyz_km=r_eci_km,
                cfg=cfg,
            )
            phi_map_per_pixel = np.asarray(resp_arrays_m["phi_ph_m2_s_pix"], dtype=np.float64)  # (n_frames, n_map_samples)

            phi_map_per_pixel_2d = phi_map_per_pixel.reshape(n_frames, ny, nx)
            phi_map_per_sr_2d = phi_map_per_pixel_2d / omega_map_val
            phi_map_per_arcsec2_2d = phi_map_per_sr_2d * float(ARCSEC2_TO_SR)

            map2d = {
                "downsample": ds,
                "x_pix_axis": x_axis,
                "y_pix_axis": y_axis,
                "phi_per_pixel": phi_map_per_pixel_2d,
                "phi_per_sr": phi_map_per_sr_2d,
                "phi_per_arcsec2": phi_map_per_arcsec2_2d,
                "source": "wsl",
                "model": None,
                "wsl_response_meta": resp_meta_m,
                "request_base_win": str(req_base_m),
                "response_base_win": str(resp_base_m),
                "omega_pix_map": {
                    "mode": str(omega_pix_map.get("mode", "")),
                    "shape_tag": str(omega_pix_map.get("shape_tag", "")),
                    "time_dependent": bool(omega_pix_map.get("time_dependent", False)),
                    "units": str(omega_pix_map.get("units", "sr")),
                    "params": omega_pix_map.get("params", {}),
                },
            }

    # Store Ω_pix product in a schema that supports scalar or arrays without ambiguity.
    omega_store: Dict[str, Any] = {
        "mode": str(omega_pix.get("mode", "")),
        "shape_tag": str(omega_pix.get("shape_tag", "")),
        "time_dependent": bool(omega_pix.get("time_dependent", False)),
        "units": str(omega_pix.get("units", "sr")),
        "params": omega_pix.get("params", {}),
    }
    if str(omega_pix.get("shape_tag", "")) == "scalar":
        omega_store["omega_pix_sr_scalar"] = float(omega_pix["omega_pix_sr"])
    else:
        omega_store["omega_pix_sr"] = np.asarray(omega_pix["omega_pix_sr"], dtype=np.float64)

    return {
        "observer_name": str(obs_name),
        "window_index": int(window_index),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "n_frames": int(n_frames),
        "times_utc_iso": times_utc_iso,
        "pointing": {
            "boresight_ra_deg": boresight_ra_deg,
            "boresight_dec_deg": boresight_dec_deg,
            "boresight_roll_deg": boresight_roll_deg,
        },
        "sensor": {
            "name": str(sensor.name),
            "rows": int(sensor.rows),
            "cols": int(sensor.cols),
        },
        "zodi": {
            "omega_pix": omega_store,
            "units": {
                "per_pixel": "ph m-2 s-1 pix-1",
                "per_sr": "ph m-2 s-1 sr-1",
                "per_arcsec2": "ph m-2 s-1 arcsec-2",
                "arcsec2_to_sr": float(ARCSEC2_TO_SR),
            },
            "bandpass_request": bandpass,
            "sampling": {
                "sample_grid": tuple(cfg.field.sample_grid),
                "sample_margin_pix": int(cfg.field.sample_margin_pix),
                "normalized_uv": bool(cfg.field.normalized_uv),
                "uv_basis": uv_basis,
                "x_pix_samples": x_samp,
                "y_pix_samples": y_samp,
                "u_samples": u_samp,
                "v_samples": v_samp,
            },
            "models": models,
            "map2d": map2d,
            "diagnostics": diagnostics,
            "provenance": {
                "wsl_response_meta_samples": resp_meta_s,
                "request_base_win_samples": str(req_base_s),
                "response_base_win_samples": str(resp_base_s),
            },
        },
    }
