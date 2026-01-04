"""
NEBULA_ZODIACAL_LIGHT_WSL_WORKER.py  (WSL-only entrypoint)

This is the stable WSL-side entrypoint that Windows calls via wsl.exe.
It MUST be executed inside WSL because it ultimately imports and calls m4opt
(indirectly, via NEBULA_ZODIACAL_M4OPT_BACKEND).

CLI contract
------------
Usage:
    python NEBULA_ZODIACAL_LIGHT_WSL_WORKER.py <request_base> <response_base>

Where <request_base> and <response_base> are *base paths without extensions*.

Input files (created on Windows side):
    <request_base>.json     # metadata (JSON)
    <request_base>.npz      # arrays (NPZ)

Output files (written by this worker):
    <response_base>.json
    <response_base>.npz

Separation of responsibilities (critical)
-----------------------------------------
- Windows/NEBULA:
    - builds sampling geometry (sample directions RA/Dec per frame) using NEBULA WCS,
    - serializes request payload,
    - orchestrates execution of this worker,
    - fits plane/quadratic coefficients and produces final pickle.

- This WSL worker:
    - loads request payload (JSON+NPZ),
    - merges meta+arrays into a single `req` dict,
    - calls compute backend (imports m4opt),
    - splits backend response into meta+arrays,
    - writes response payload (JSON+NPZ).

Import robustness
-----------------
This file lives in the Windows repo but runs in WSL. To make imports reliable,
we add the NEBULA repo root to sys.path at runtime based on this file location.

Backend API used
----------------
We call:
    Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_M4OPT_BACKEND.compute_zodi_for_window_request(req)

That backend returns a dict "resp" containing both metadata and numpy arrays.
We split them automatically:
- any np.ndarray values -> NPZ arrays
- everything else -> JSON metadata

Expected request keys (minimum)
-------------------------------
Backend requires keys such as:
- times_utc_iso (n_frames)
- sample_ra_deg + sample_dec_deg OR sample_radec_deg
- omega_pix_sr
- bandpass dict with mode and parameters

This worker does not validate the full schema; it performs basic checks and lets
the backend raise clear errors for missing keys or incompatible shapes.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def get_logger() -> logging.Logger:
    """
    Create a console logger for this worker.

    Returns
    -------
    logging.Logger
        Logger that writes to stderr (captured by the Windows bridge), INFO level by default.
    """
    lg = logging.getLogger("NEBULA_ZODIACAL_LIGHT_WSL_WORKER")
    if not lg.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
    return lg


# -----------------------------------------------------------------------------
# Repo root / sys.path plumbing
# -----------------------------------------------------------------------------

def add_repo_root_to_syspath() -> Path:
    """
    Add the NEBULA repo root to sys.path so imports work when executed in WSL.

    Returns
    -------
    Path
        Inferred repo root.

    Assumed location
    ----------------
    This file is expected at:
        <NEBULA_REPO>/Utility/ZODIACAL_LIGHT/NEBULA_ZODIACAL_LIGHT_WSL_WORKER.py

    Therefore the repo root is:
        Path(__file__).resolve().parents[2]
    """
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


# -----------------------------------------------------------------------------
# Imports (deferred until sys.path is prepared)
# -----------------------------------------------------------------------------

def _import_io():
    """
    Import the cross-environment payload IO functions.

    Returns
    -------
    (read_payload, write_payload) : (callable, callable)
        Functions from Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_LIGHT_IO
    """
    from Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_LIGHT_IO import (  # type: ignore
        read_payload,
        write_payload,
    )
    return read_payload, write_payload


def _import_backend():
    """
    Import the m4opt backend compute function.

    Returns
    -------
    compute_zodi_for_window_request : callable
        Backend function that imports/calls m4opt.

    Notes
    -----
    This import must occur inside WSL. Windows must never import this module.
    """
    from Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_M4OPT_BACKEND import (  # type: ignore
        compute_zodi_for_window_request,
    )
    return compute_zodi_for_window_request


# -----------------------------------------------------------------------------
# Payload utilities
# -----------------------------------------------------------------------------

def _merge_meta_and_arrays(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Merge JSON metadata and NPZ arrays into a single request dict for the backend.

    Parameters
    ----------
    meta : dict
        JSON metadata loaded from <request_base>.json
    arrays : dict[str, np.ndarray]
        Numeric arrays loaded from <request_base>.npz

    Returns
    -------
    req : dict
        Combined dict passed into compute_zodi_for_window_request(req).

    Collision policy
    ----------------
    If a key exists in both meta and arrays, the array value wins (arrays overwrite meta).
    This is intentional: numeric payload should take precedence.
    """
    req: Dict[str, Any] = dict(meta)
    for k, v in arrays.items():
        req[k] = v
    return req


def _split_resp_meta_and_arrays(resp: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Split backend response dict into JSON meta and NPZ arrays.

    Parameters
    ----------
    resp : dict
        Backend response. May contain strings, dicts, floats, and numpy arrays.

    Returns
    -------
    (out_meta, out_arrays) : (dict, dict[str, np.ndarray])
        out_meta:
            JSON-serializable metadata (everything not a numpy array).
        out_arrays:
            All np.ndarray values.

    Raises
    ------
    TypeError
        If resp contains unsupported array-like values (e.g., lists of arrays).
    ValueError
        If any output array is dtype=object (not allowed for our NPZ contract).
    """
    out_meta: Dict[str, Any] = {}
    out_arrays: Dict[str, np.ndarray] = {}

    for k, v in resp.items():
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                raise ValueError(
                    f"Backend output array '{k}' has dtype=object. "
                    f"Convert to numeric or fixed-width unicode before returning."
                )
            out_arrays[str(k)] = v
        else:
            out_meta[str(k)] = v

    return out_meta, out_arrays


def _preflight_iers(req: Dict[str, Any], lg: logging.Logger) -> None:
    """
    Preflight Astropy IERS auto-download for time/coordinate transforms.

    Motivation
    ----------
    TEME->ITRS (used in the backend when observer_eci_xyz_km is provided) is
    time-dependent and can require up-to-date Earth-orientation parameters.
    Astropy will auto-download IERS-A data by default, but failures then occur
    deep in the backend. This preflight makes failures immediate and actionable.

    Behavior
    --------
    - Runs only if 'observer_eci_xyz_km' is present in the request.
    - Forces IERS_Auto to open and (if needed) download updated tables.
    - Also touches Time(...).ut1 for the requested times to ensure coverage.
    """
    if "observer_eci_xyz_km" not in req:
        return  # not using TEME->ITRS path

    from astropy.time import Time
    from astropy.utils import iers
    from astropy.utils.iers import IERS_Auto

    # Make the "online is fine" assumption explicit and deterministic.
    iers.conf.auto_download = True
    iers.conf.remote_timeout = 30.0  # seconds (tune as needed)

    try:
        # Load IERS table (downloads if cache is missing/stale per auto_max_age).
        IERS_Auto.open()

        # Ensure UT1-UTC/polar motion are available for the specific requested times.
        # This will trigger download if needed and will raise if data cannot be obtained.
        if "times_utc_iso" in req:
            t = Time(list(req["times_utc_iso"]), scale="utc")
            _ = t.ut1

        lg.info(
            "IERS preflight OK (auto_download=%s, remote_timeout=%.1fs)",
            bool(iers.conf.auto_download),
            float(iers.conf.remote_timeout),
        )

    except Exception as e:
        raise RuntimeError(
            "Astropy IERS preflight failed. This typically indicates that the WSL "
            "environment cannot download/update Earth-orientation (IERS) data.\n"
            "If you are running online, check DNS/proxy/firewall and try increasing "
            "iers.conf.remote_timeout.\n"
            "If you later want offline support, install/update the astropy-iers-data "
            "package and disable auto-download explicitly.\n"
        ) from e


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    """
    Execute the WSL worker.

    Parameters
    ----------
    argv : list[str]
        sys.argv

    Returns
    -------
    int
        Exit code:
          0 = success
          2 = CLI usage error
          1 = runtime error
    """
    lg = get_logger()

    if len(argv) != 3:
        print(
            "Usage: python NEBULA_ZODIACAL_LIGHT_WSL_WORKER.py <request_base> <response_base>",
            file=sys.stderr,
        )
        return 2

    request_base = Path(argv[1])
    response_base = Path(argv[2])

    repo_root = add_repo_root_to_syspath()
    lg.info("Repo root (WSL): %s", repo_root)

    try:
        read_payload, write_payload = _import_io()
        compute = _import_backend()

        # 1) Read request payload (no object arrays, no pickling)
        meta, arrays = read_payload(
            request_base,
            allow_pickle=False,
            forbid_object_arrays=True,
        )

        # 2) Merge meta+arrays into the request dict expected by the backend
        req = _merge_meta_and_arrays(meta, arrays)

        # 2.5) Preflight IERS auto-download if we will use TEME->ITRS in the backend
        _preflight_iers(req, lg)

        # 3) Compute zodiacal background via m4opt backend (WSL-only)
        resp = compute(req)

        if not isinstance(resp, dict):
            raise TypeError(f"Backend must return dict; got {type(resp)!r}.")

        # 4) Split response into JSON metadata and NPZ arrays
        out_meta, out_arrays = _split_resp_meta_and_arrays(resp)

        # 5) Write response payload
        write_payload(
            response_base,
            out_meta,
            out_arrays,
            compress=True,
            overwrite=True,
            forbid_object_arrays=True,
        )

        lg.info("ZL WSL worker: success (wrote %s.[json|npz])", response_base)
        return 0

    except Exception as e:
        lg.error("ZL WSL worker failed: %s", e)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
