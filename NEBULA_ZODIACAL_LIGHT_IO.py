"""
NEBULA_ZODIACAL_LIGHT_IO.py

Cross-environment request/response IO utilities for Zodiacal Light (ZL) integration.

Why this file exists
--------------------
Windows NEBULA orchestrates the ZL computation but m4opt runs in WSL (Linux).
We therefore need a robust, environment-agnostic interchange format to pass:

  (A) small metadata (JSON)
  (B) large numeric arrays (NPZ)

This module standardizes that interchange format and provides a small API:

  - write_payload(base_path, meta, arrays, ...)
  - read_payload(base_path, ...)

File convention (base path, no extension)
-----------------------------------------
Given a base path like:

    C:\\...\\NEBULA_OUTPUT\\TMP\\ZODIACAL_LIGHT\\zodi_request

We write two files:

    zodi_request.json   # metadata, JSON-serializable only
    zodi_request.npz    # numpy arrays (multiple arrays per file)

Likewise for responses:

    zodi_response.json
    zodi_response.npz

Notes and constraints
---------------------
1) Arrays must NOT be "object dtype".
   We intentionally load NPZ with allow_pickle=False; object arrays require pickling
   and tend to be brittle across environments.

2) Datetimes and Paths in `meta` are serialized to strings:
   - datetime -> ISO 8601 string (datetime.isoformat())
   - Path -> str(path)

3) NPZ keys must be valid strings; keep them simple (letters, numbers, underscores).

Intended usage patterns
-----------------------
- Windows stage builds request:
    meta = {...}, arrays = {...}
    write_payload(req_base, meta, arrays)

- WSL worker reads:
    meta, arrays = read_payload(req_base)

- WSL worker writes response:
    write_payload(resp_base, out_meta, out_arrays)

- Windows reads response:
    out_meta, out_arrays = read_payload(resp_base)
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np


class ZLIOError(RuntimeError):
    """Raised when a payload read/write operation fails validation."""


# -----------------------------
# Path helpers
# -----------------------------

def payload_paths(base_path: str | Path) -> Tuple[Path, Path]:
    """
    Compute the JSON and NPZ filenames corresponding to a payload base path.

    Parameters
    ----------
    base_path : str | Path
        Base path WITHOUT extension. This can be absolute or relative.
        Examples:
            "zodi_request"
            "C:/.../zodi_request"
            Path(".../zodi_request")

    Returns
    -------
    (json_path, npz_path) : (Path, Path)
        Paths:
            <base_path>.json
            <base_path>.npz
    """
    base = Path(base_path)
    return base.with_suffix(".json"), base.with_suffix(".npz")


def ensure_parent_dir(path: str | Path) -> None:
    """
    Ensure the parent directory for a path exists.

    Parameters
    ----------
    path : str | Path
        Any file path.

    Returns
    -------
    None
        Creates parent directories if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# JSON serialization helpers
# -----------------------------

def _json_default(obj: Any) -> Any:
    """
    Default JSON serializer for objects not natively JSON-serializable.

    Supported conversions
    ---------------------
    - datetime/date -> ISO string (obj.isoformat())
    - pathlib.Path  -> str(obj)
    - numpy scalar  -> python scalar via .item()
    - dataclass     -> asdict(dataclass)

    Parameters
    ----------
    obj : Any
        Arbitrary object encountered by json.dump/json.dumps.

    Returns
    -------
    Any
        JSON-compatible representation.

    Raises
    ------
    TypeError
        If obj cannot be converted.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if is_dataclass(obj):
        return asdict(obj)

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _validate_meta_is_jsonable(meta: Mapping[str, Any]) -> None:
    """
    Validate that `meta` can be serialized as JSON using our default serializer.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Metadata dictionary.

    Returns
    -------
    None

    Raises
    ------
    ZLIOError
        If meta cannot be JSON-serialized.
    """
    try:
        json.dumps(meta, default=_json_default)
    except Exception as e:
        raise ZLIOError(f"Metadata is not JSON-serializable: {e}") from e


# -----------------------------
# Array validation helpers
# -----------------------------

def _is_object_array(a: np.ndarray) -> bool:
    """
    True if array has dtype=object.

    We forbid object arrays by default because loading them requires pickling,
    and we standardize on allow_pickle=False for portability.
    """
    return isinstance(a, np.ndarray) and a.dtype == object


def validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    forbid_object_arrays: bool = True,
) -> None:
    """
    Validate that the arrays dict is safe and consistent with our NPZ contract.

    Parameters
    ----------
    arrays : Mapping[str, np.ndarray]
        Dictionary mapping array names -> numpy arrays.
    forbid_object_arrays : bool
        If True (recommended), raise if any array has dtype=object.

    Returns
    -------
    None

    Raises
    ------
    ZLIOError
        If keys are invalid or arrays violate constraints.
    """
    for k, v in arrays.items():
        if not isinstance(k, str) or not k:
            raise ZLIOError(f"NPZ array key must be a non-empty str; got {k!r}.")

        if not isinstance(v, np.ndarray):
            raise ZLIOError(f"NPZ array value for key '{k}' must be np.ndarray; got {type(v)!r}.")

        if forbid_object_arrays and _is_object_array(v):
            raise ZLIOError(
                f"Array '{k}' has dtype=object. Convert to numeric or fixed-width unicode "
                f"(e.g., np.array(list_of_iso_strings, dtype='U'))."
            )


# -----------------------------
# Public API
# -----------------------------

def write_payload(
    base_path: str | Path,
    meta: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    compress: bool = True,
    overwrite: bool = True,
    forbid_object_arrays: bool = True,
) -> Tuple[Path, Path]:
    """
    Write a payload to <base>.json + <base>.npz.

    Parameters
    ----------
    base_path : str | Path
        Base path WITHOUT extension.
    meta : Mapping[str, Any]
        JSON-serializable metadata. Datetimes/Paths are allowed (they will be
        converted to strings by our serializer).
    arrays : Mapping[str, np.ndarray]
        Numpy arrays to store. Keys become variable names inside the NPZ.
    compress : bool
        If True, use np.savez_compressed (ZIP_DEFLATED compression).
        If False, use np.savez (no compression).
    overwrite : bool
        If False, raise if output files already exist.
    forbid_object_arrays : bool
        If True, raise if any array is dtype=object.

    Returns
    -------
    (json_path, npz_path) : (Path, Path)
        Paths written.

    Raises
    ------
    ZLIOError
        If meta/arrays fail validation or files exist and overwrite=False.
    """
    json_path, npz_path = payload_paths(base_path)

    _validate_meta_is_jsonable(meta)
    validate_arrays(arrays, forbid_object_arrays=forbid_object_arrays)

    ensure_parent_dir(json_path)

    if not overwrite:
        if json_path.exists():
            raise ZLIOError(f"Refusing to overwrite existing file: {json_path}")
        if npz_path.exists():
            raise ZLIOError(f"Refusing to overwrite existing file: {npz_path}")

    # Write JSON metadata
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dict(meta), f, indent=2, sort_keys=True, default=_json_default)

    # Write NPZ arrays
    # Note: np.savez* requires keyword arguments (names) -> arrays
    if compress:
        np.savez_compressed(npz_path, **{k: v for k, v in arrays.items()})
    else:
        np.savez(npz_path, **{k: v for k, v in arrays.items()})

    return json_path, npz_path


def read_payload(
    base_path: str | Path,
    *,
    allow_pickle: bool = False,
    forbid_object_arrays: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Read a payload from <base>.json + <base>.npz.

    Parameters
    ----------
    base_path : str | Path
        Base path WITHOUT extension.
    allow_pickle : bool
        Passed to numpy.load. Recommended False.
        If False, NPZ payloads containing object arrays will fail to load.
    forbid_object_arrays : bool
        If True, validate_arrays() will reject dtype=object even if allow_pickle=True.

    Returns
    -------
    (meta, arrays) : (dict, dict[str, np.ndarray])
        meta:
            The JSON metadata dict (datetimes remain strings; caller may parse).
        arrays:
            Dict mapping array names -> numpy arrays.

    Raises
    ------
    FileNotFoundError
        If either the .json or .npz file is missing.
    ZLIOError
        If the NPZ cannot be loaded or violates constraints.
    """
    json_path, npz_path = payload_paths(base_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Missing payload metadata JSON: {json_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing payload arrays NPZ: {npz_path}")

    # Read JSON metadata
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ZLIOError(f"Payload JSON root must be a dict; got {type(meta)!r}.")

    # Read NPZ arrays
    arrays: Dict[str, np.ndarray] = {}
    try:
        with np.load(npz_path, allow_pickle=allow_pickle) as z:
            for name in z.files:
                arrays[name] = z[name]
    except Exception as e:
        raise ZLIOError(f"Failed to load NPZ payload '{npz_path}': {e}") from e

    validate_arrays(arrays, forbid_object_arrays=forbid_object_arrays)

    return meta, arrays
