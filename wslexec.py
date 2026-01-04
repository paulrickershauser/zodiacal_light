"""
stage/wslexec.py

Stage-facing WSL execution utilities for the Zodiacal Light (ZL) pipeline.

Why this file exists
--------------------
The ZL stage needs to do two things related to WSL execution:

  (1) Choose deterministic *base paths* for request/response payloads.
      A "base path" is a path WITHOUT an extension. The payload layer appends
      the extensions:

          <request_base>.json  +  <request_base>.npz
          <response_base>.json +  <response_base>.npz

  (2) Invoke the WSL worker (Linux-side) to compute zodiacal light values,
      then return the captured stdout/stderr and enforce "response files exist".

This module is intentionally thin and delegates all WSL mechanics to:

    Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_LIGHT_WSL_BRIDGE

Key design rules (fail-fast, no legacy)
---------------------------------------
- No legacy call signatures are supported here. All functions are keyword-only.
  If a call site uses an older signature, it should raise immediately and be fixed.

- No subprocess usage here.
  All subprocess command construction, quoting, timeout handling, stdout/stderr capture,
  returncode checking, and response existence checking is centralized in the bridge.

- Dataclass-config only.
  All configuration comes from the single exported instance:
      Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG.ZODIACAL_LIGHT_CONFIG

Typical usage inside the stage
------------------------------
1) Determine deterministic base names for a given (observer, window):

    req_base, resp_base = make_tmp_bases(
        obs_name=obs_name,
        window_index=window_index,
        suffix="samples",  # optional
    )

2) Write request payload (handled by NEBULA_ZODIACAL_LIGHT_IO):
    write_payload(req_base, meta=request_meta, arrays=request_arrays)

3) Invoke WSL worker:
    result = invoke_wsl_worker(request_base_win=req_base, response_base_win=resp_base)

4) Read response payload:
    meta, arrays = read_payload(resp_base)

This module does not read/write payload contents; it just provides base paths and
delegates execution to the bridge.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG import (
    ZODIACAL_LIGHT_CONFIG,
    ZodiacalLightConfig,
)

from Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_LIGHT_WSL_BRIDGE import (
    WSLRunResult,
    WSLWorkerError,
    ensure_tmp_dir,
    make_request_response_bases,
    run_wsl_worker,
)

__all__ = [
    "make_tmp_bases",
    "invoke_wsl_worker",
    "WSLRunResult",
    "WSLWorkerError",
]

_LOGGER_NAME = "nebula.zodiacal_light.stage.wslexec"


def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger for this module.

    The stage frequently passes around a logger instance; when absent we fall back to a
    module-scoped logger name. We do not attach handlers here (that is a higher-level
    application concern); this keeps logging behavior consistent across the codebase.
    """
    return logger if logger is not None else logging.getLogger(_LOGGER_NAME)


def _require_base_path_no_suffix(value: Union[str, Path], *, name: str) -> Path:
    """
    Enforce the stage contract: base paths MUST NOT include a file extension.

    This prevents confusing behavior like:
      - caller passes ".../req.json"
      - payload layer expects ".../req.json.json" and ".../req.json.npz"
    """
    p = Path(value)
    if p.suffix:
        raise ValueError(
            f"{name} must be a base path WITHOUT an extension; got {str(p)!r} (suffix={p.suffix!r})."
        )
    return p


def make_tmp_bases(
    *,
    obs_name: str,
    window_index: int,
    suffix: Optional[str] = None,
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> Tuple[Path, Path]:
    """
    Construct deterministic Windows-side base paths for request and response payloads.

    This is a stage-level convenience wrapper over the bridge’s naming policy.

    Parameters
    ----------
    obs_name:
        Observer name (human-readable). The bridge sanitizes this into a filesystem-safe
        token so the resulting filenames are stable and portable.
    window_index:
        Window index. Encoded into the base name (e.g., w0000) to keep multiple windows
        distinct within the same observer.
    suffix:
        Optional discriminator for development or multiple payload passes per window
        (e.g., "samples", "map", "fit"). If provided, it becomes part of the base name.
    cfg:
        Active ZodiacalLightConfig instance. Defaults to the singleton
        ZODIACAL_LIGHT_CONFIG.

    Returns
    -------
    (request_base_win, response_base_win):
        Both are Windows Paths WITHOUT extensions. The payload I/O layer will append
        ".json" and ".npz" when reading/writing the actual files.

    Fail-fast behavior
    ------------------
    - Ensures the configured temp directory exists before returning base paths.
    - Performs no "guessing" about directories or naming.
    """
    # Ensure the configured temp directory exists; this makes the stage robust to call order.
    ensure_tmp_dir(cfg)

    # Delegate naming convention to the bridge (single source of truth).
    return make_request_response_bases(
        obs_name=obs_name,
        window_index=int(window_index),
        suffix=suffix,
        cfg=cfg,
    )


def invoke_wsl_worker(
    *,
    request_base_win: Union[str, Path],
    response_base_win: Union[str, Path],
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
    logger: Optional[logging.Logger] = None,
) -> WSLRunResult:
    """
    Invoke the ZL WSL worker via the bridge (subprocess lives in the bridge, not here).

    This function assumes the stage has already written:
        <request_base_win>.json
        <request_base_win>.npz

    The bridge will:
      - validate request payload files exist
      - resolve the worker script path
      - translate paths to WSL mount paths
      - run wsl.exe ... bash -lc "python worker req_base resp_base"
      - enforce timeout
      - raise on non-zero returncode
      - verify response payload files exist:
            <response_base_win>.json
            <response_base_win>.npz

    Parameters
    ----------
    request_base_win, response_base_win:
        Windows base paths (no extension) for request and response payloads.
    cfg:
        Active ZodiacalLightConfig instance (dataclass config world).
    logger:
        Optional logger for the bridge run.

    Returns
    -------
    WSLRunResult
        Captured stdout/stderr and argv used for the WSL invocation.

    Raises
    ------
    ValueError
        If a caller passes a path with an extension (violates base-path contract).
    FileNotFoundError
        - If request payload files are missing
        - If the worker script is missing
        - If wsl.exe cannot be found/executed
    WSLWorkerError
        - If the worker fails (non-zero returncode)
        - If the worker times out
        - If the worker exits successfully but response files are missing
    """
    lg = _get_logger(logger)
    lg.debug(
        "invoke_wsl_worker(): request_base_win=%s response_base_win=%s",
        request_base_win,
        response_base_win,
    )

    # Make invoke robust even if a caller bypasses make_tmp_bases().
    ensure_tmp_dir(cfg)

    # Enforce the base-path (no extension) contract at the stage boundary.
    req_base = _require_base_path_no_suffix(request_base_win, name="request_base_win")
    resp_base = _require_base_path_no_suffix(response_base_win, name="response_base_win")

    # Delegate all WSL execution mechanics to the bridge.
    return run_wsl_worker(
        request_base_win=req_base,
        response_base_win=resp_base,
        cfg=cfg,
        logger=lg,
    )
