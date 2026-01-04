"""
stage/orchestrator.py

Zodiacal Light (ZL) stage orchestration (Windows-side).

This module is intentionally **not** a CLI script and contains **no** `main()`.
It is imported and called by NEBULA’s pipeline/stage manager (or by a developer
from an interactive session) to build the standalone product:

    NEBULA_OUTPUT/ZODIACAL_LIGHT/obs_zodiacal_light.pkl

Fail-fast design
----------------
This orchestrator follows the project’s "fail fast if not wired" requirement:

- If required keys are missing from the input pickle, it raises immediately.
- If the pickle metadata contradicts ACTIVE_SENSOR / config expectations, it raises.
- If output exists and overwrite is disabled in config, it raises.
- It does not silently “fallback” to alternate geometry or alternate config styles.

Role in the refactor
--------------------
You are splitting the former monolithic NEBULA_ZODIACAL_LIGHT_STAGE.py into:

- stage/projection_wcs.py   (sampling sky directions via WCS)
- stage/bandpass.py         (constructing worker bandpass request dict)
- stage/wslexec.py          (temporary-base naming / bridge integration)
- stage/fitting.py          (plane3 / quad6 fits)
- stage/window_product.py   (per-window ZL product dict)
- stage/orchestrator.py     (THIS FILE: loops observers/windows, IO, output pickle)

This file should remain “boring”: IO + loops + provenance only. All computation
is delegated to stage/window_product.py.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# Central ZL configuration (dataclass-based config world).
# This module must exist and is treated as authoritative.
from Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG import (
    ZODIACAL_LIGHT_CONFIG,
    get_sensor_config,
    validate_config,
)

# NEBULA output root (Windows-side)
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# Per-window builder (does the actual work)
from .window_product import build_window_zodi_product


class ZodiacalLightStageError(RuntimeError):
    """Raised for stage-level wiring/contract violations (fail-fast)."""


# -----------------------------------------------------------------------------
# Small helpers (fail-fast)
# -----------------------------------------------------------------------------

_MISSING = object()


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """
    Return a usable logger for stage execution.

    - If a logger is provided, it is used directly.
    - Otherwise, returns a module-level logger with a basic configuration.

    This helper exists so callers can pass pipeline-managed loggers while still
    allowing simple interactive usage.
    """
    if logger is not None:
        return logger
    lg = logging.getLogger(__name__)
    if not lg.handlers:
        logging.basicConfig(level=logging.INFO)
    return lg


def _track_get(track: Any, key: str, default: Any = _MISSING) -> Any:
    """
    Retrieve `key` from a TrackDict-like object, in a way that is robust to the
    two common representations used in NEBULA:

    1) dict-like: track[key] / track.get(key)
    2) attribute-like: track.key

    Parameters
    ----------
    track : Any
        TrackDict-like object (typically loaded from obs_window_sources.pkl).
    key : str
        Key/attribute name to retrieve.
    default : Any
        If provided, returned when the key is missing. If not provided (the
        sentinel `_MISSING`), a missing key causes an exception.

    Returns
    -------
    Any
        The retrieved value.

    Raises
    ------
    ZodiacalLightStageError
        If the key is missing and no default was provided.
    """
    # dict
    if isinstance(track, dict):
        if key in track:
            return track[key]
        if default is not _MISSING:
            return default
        raise ZodiacalLightStageError(f"Missing required key in obs_track dict: {key!r}")

    # dict-like (TrackDict implementations often provide .get)
    if hasattr(track, "get"):
        try:
            val = track.get(key, _MISSING)  # type: ignore[attr-defined]
            if val is not _MISSING:
                return val
        except TypeError:
            # Some custom get() signatures may differ; fall through to attribute access.
            pass

    # attribute
    if hasattr(track, key):
        return getattr(track, key)

    if default is not _MISSING:
        return default
    raise ZodiacalLightStageError(f"Missing required key/attribute in obs_track: {key!r}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_obs_zodiacal_light_for_all_observers(
    *,
    window_sources_pickle_path: Optional[Path] = None,
    output_pickle_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build `obs_zodiacal_light.pkl` for all observers in `obs_window_sources.pkl`.
    """
    lg = _get_logger(logger)

    validate_config()
    cfg = ZODIACAL_LIGHT_CONFIG

    try:
        default_src = (NEBULA_OUTPUT_DIR / cfg.io.window_sources_relpath).resolve()
    except Exception as e:
        raise ZodiacalLightStageError(
            "ZODIACAL_LIGHT_CONFIG must provide io.window_sources_relpath "
            "for resolving obs_window_sources.pkl."
        ) from e

    try:
        default_out = (NEBULA_OUTPUT_DIR / cfg.io.output_relpath).resolve()
    except Exception as e:
        raise ZodiacalLightStageError(
            "ZODIACAL_LIGHT_CONFIG must provide io.output_relpath "
            "for resolving obs_zodiacal_light.pkl."
        ) from e

    src_path = window_sources_pickle_path.resolve() if window_sources_pickle_path else default_src
    out_path = output_pickle_path.resolve() if output_pickle_path else default_out

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"WINDOW_SOURCES pickle not found: {src_path}")

    overwrite = bool(getattr(cfg.io, "overwrite", False))
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output pickle already exists and overwrite is disabled (cfg.io.overwrite=False): {out_path}"
        )

    lg.info("Loading WINDOW_SOURCES pickle: %s", src_path)
    with open(src_path, "rb") as f:
        obs_window_sources: Dict[str, Any] = pickle.load(f)

    if not isinstance(obs_window_sources, dict):
        raise ZodiacalLightStageError(
            "obs_window_sources.pkl must deserialize to a dict[str, TrackDict-like]. "
            f"Got: {type(obs_window_sources)!r}"
        )

    created_utc = datetime.now(timezone.utc).isoformat()

    obs_zodi: Dict[str, Any] = {}
    for obs_name, obs_track in obs_window_sources.items():
        if not isinstance(obs_name, str) or not obs_name:
            raise ZodiacalLightStageError(f"Observer key must be a non-empty str; got {obs_name!r}")

        obs_entry = _build_obs_entry(obs_name, obs_track, created_utc, lg)
        obs_zodi[obs_name] = obs_entry

    lg.info("Writing zodiacal-light output pickle: %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(obs_zodi, f, protocol=pickle.HIGHEST_PROTOCOL)

    lg.info("Zodiacal Light stage complete: %d observers.", len(obs_zodi))
    return obs_zodi


def _build_obs_entry(
    obs_name: str,
    obs_track: Any,
    created_utc: str,
    lg: logging.Logger,
) -> Dict[str, Any]:
    """
    Build the output dict entry for one observer.
    """
    cfg = ZODIACAL_LIGHT_CONFIG

    sensor = get_sensor_config()
    sensor_name = str(getattr(sensor, "name", "UNKNOWN_SENSOR"))

    pickle_sensor_name = _track_get(obs_track, "sensor_name", default=None)
    if pickle_sensor_name is not None and str(pickle_sensor_name) != sensor_name:
        raise ZodiacalLightStageError(
            f"Observer {obs_name!r}: obs_track.sensor_name={pickle_sensor_name!r} "
            f"does not match ACTIVE_SENSOR.name={sensor_name!r}."
        )

    rows = int(getattr(sensor, "rows"))
    cols = int(getattr(sensor, "cols"))

    pickle_rows = _track_get(obs_track, "rows", default=None)
    pickle_cols = _track_get(obs_track, "cols", default=None)
    if pickle_rows is not None and int(pickle_rows) != rows:
        raise ZodiacalLightStageError(
            f"Observer {obs_name!r}: obs_track.rows={pickle_rows!r} does not match ACTIVE_SENSOR.rows={rows}."
        )
    if pickle_cols is not None and int(pickle_cols) != cols:
        raise ZodiacalLightStageError(
            f"Observer {obs_name!r}: obs_track.cols={pickle_cols!r} does not match ACTIVE_SENSOR.cols={cols}."
        )

    dt_frame_s_val = _track_get(obs_track, "dt_frame_s")
    try:
        dt_frame_s = float(dt_frame_s_val)
    except Exception as e:
        raise ZodiacalLightStageError(
            f"Observer {obs_name!r}: dt_frame_s must be a float-like value; got {dt_frame_s_val!r}"
        ) from e

    try:
        catalog_name = str(cfg.catalog.catalog_name_expected)
        catalog_band = str(cfg.catalog.catalog_band_expected)
    except Exception as e:
        raise ZodiacalLightStageError(
            "ZODIACAL_LIGHT_CONFIG must provide catalog.catalog_name_expected and "
            "catalog.catalog_band_expected for provenance and bandpass routing."
        ) from e

    pickle_catalog_name = _track_get(obs_track, "catalog_name", default=None)
    pickle_catalog_band = _track_get(obs_track, "catalog_band", default=None)

    if pickle_catalog_name is not None and str(pickle_catalog_name) != catalog_name:
        raise ZodiacalLightStageError(
            f"Observer {obs_name!r}: obs_track.catalog_name={pickle_catalog_name!r} "
            f"does not match config catalog_name_expected={catalog_name!r}."
        )
    if pickle_catalog_band is not None and str(pickle_catalog_band) != catalog_band:
        raise ZodiacalLightStageError(
            f"Observer {obs_name!r}: obs_track.catalog_band={pickle_catalog_band!r} "
            f"does not match config catalog_band_expected={catalog_band!r}."
        )

    windows = _track_get(obs_track, "windows")
    if not isinstance(windows, list):
        raise ZodiacalLightStageError(
            f"Observer {obs_name!r}: obs_track.windows must be a list; got {type(windows)!r}."
        )

    lg.info(
        "Observer %r: %d windows (sensor=%s, %dx%d, dt=%.6fs, catalog=%s/%s).",
        obs_name,
        len(windows),
        sensor_name,
        rows,
        cols,
        dt_frame_s,
        catalog_name,
        catalog_band,
    )

    schema_version = str(cfg.io.schema_version)

    obs_entry: Dict[str, Any] = {
        "observer_name": obs_name,
        "sensor_name": sensor_name,
        "rows": rows,
        "cols": cols,
        "dt_frame_s": dt_frame_s,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "schema_version": schema_version,
        "created_utc": created_utc,
        "windows": [],
    }

    for w in windows:
        if not isinstance(w, dict):
            raise ZodiacalLightStageError(
                f"Observer {obs_name!r}: each window must be a dict; got {type(w)!r}."
            )
        window_product = build_window_zodi_product(
            obs_name=str(obs_name),
            obs_track=obs_track,
            window=w,
            cfg=cfg,
        )
        obs_entry["windows"].append(window_product)

    return obs_entry
