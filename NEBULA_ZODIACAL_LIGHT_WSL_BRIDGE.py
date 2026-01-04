"""
NEBULA_ZODIACAL_LIGHT_WSL_BRIDGE.py

Windows-side bridge for invoking the Zodiacal Light (ZL) WSL worker and managing
Windows ↔ WSL path translation.

This module is part of the *Windows orchestration* side of the ZL pipeline.

High-level responsibilities
---------------------------
1) Translate Windows filesystem paths to corresponding WSL mount paths.

   Example mapping (standard WSL DrvFs mount):
     Windows:  C:\\Users\\you\\Project\\NEBULA_OUTPUT\\TMP\\file
     WSL:      /mnt/c/Users/you/Project/NEBULA_OUTPUT/TMP/file

2) Resolve the WSL worker script path deterministically from the NEBULA repo root.

3) Build and execute the WSL invocation command (via wsl.exe), capturing stdout/stderr,
   enforcing a timeout, and raising actionable errors when anything goes wrong.

4) Provide deterministic temp-base naming helpers so the stage can create request/response
   bases without re-implementing path conventions.

What this module intentionally does NOT do
------------------------------------------
- It does not define or validate the request/response *schema*. That contract is owned by:
    - the Windows stage (writes request payloads)
    - Utility.ZODIACAL_LIGHT.NEBULA_ZODIACAL_LIGHT_WSL_WORKER (reads request / writes response)
- It does not import m4opt or any WSL-only dependencies.

Configuration contract (no legacy)
----------------------------------
This bridge uses only the current ZL configuration dataclass instance:

    Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG.ZODIACAL_LIGHT_CONFIG

Legacy aliases (e.g., ACTIVE_ZODIACAL_LIGHT_CONFIG) are intentionally not used.
Any caller relying on legacy names is a wiring error and should be fixed upstream.

WSL worker argv contract
------------------------
The bridge invokes the worker inside WSL as:

    python <worker_script> <request_base> <response_base>

Where <request_base> and <response_base> are base paths (no extension). The worker expects:
    <request_base>.json
    <request_base>.npz
and must produce:
    <response_base>.json
    <response_base>.npz

Fail-fast philosophy
--------------------
This module fails loudly and early:
- Missing request payload files -> FileNotFoundError
- Worker script not found -> FileNotFoundError
- wsl.exe returns non-zero -> WSLWorkerError (with stdout/stderr + argv)
- Timeout -> WSLWorkerError (with partial stdout/stderr when available)
- Success but response files missing -> WSLWorkerError
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG import (
    ZODIACAL_LIGHT_CONFIG,
    ZodiacalLightConfig,
    resolve_tmp_dir,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

_LOGGER_NAME = "nebula.zodiacal_light.wsl_bridge"


def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger for this module.

    If a logger is provided by the caller, it is used directly. Otherwise a module-scoped
    logger name is used.
    """
    return logger if logger is not None else logging.getLogger(_LOGGER_NAME)


# -----------------------------------------------------------------------------
# Exceptions / results
# -----------------------------------------------------------------------------

class WSLWorkerError(RuntimeError):
    """
    Raised when the WSL worker invocation fails.

    The intent is “actionable failure”:
    - returncode, stdout, stderr, and argv are attached for immediate diagnosis.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        command: list[str],
    ) -> None:
        super().__init__(message)
        self.returncode = int(returncode)
        self.stdout = stdout
        self.stderr = stderr
        self.command = command


@dataclass(frozen=True)
class WSLRunResult:
    """
    Result of a successful WSL worker run (returncode==0).

    Attributes
    ----------
    returncode:
        Process return code (0 indicates success).
    stdout:
        Captured stdout from the worker invocation.
    stderr:
        Captured stderr from the worker invocation.
    command:
        The Windows-side argv list passed to subprocess.run().
    """
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


# -----------------------------------------------------------------------------
# Path translation (Windows -> WSL)
# -----------------------------------------------------------------------------

# Matches:
#   C:\path\to\file
#   C:/path/to/file
_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def win_path_to_wsl(path: Union[str, Path]) -> str:
    """
    Convert an absolute Windows drive-letter path to its corresponding WSL mount path.

    Supported input forms
    ---------------------
    - Drive-letter absolute paths:
        C:\\Users\\...   or   C:/Users/...
      These map to:
        /mnt/c/Users/...

    Fail-fast behavior
    ------------------
    Raises ValueError if the input is not a drive-letter absolute path.

    Notes
    -----
    - This targets the standard WSL DrvFs convention (/mnt/<drive>/...).
    - UNC paths (e.g., \\\\wsl$\\...) are intentionally not supported.
    """
    p = Path(path)

    # Resolve to an absolute Windows path when possible. We do NOT require existence.
    # If resolve() fails for any reason, we fall back to the raw string and enforce
    # strict parsing below.
    try:
        s = str(p.resolve())
    except Exception:
        s = str(p)

    m = _DRIVE_RE.match(s)
    if not m:
        raise ValueError(
            "win_path_to_wsl expects an absolute Windows drive-letter path like "
            "'C:/path/to/file'. Got: "
            f"{s!r}"
        )

    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _wsl_exe_prefix(distro: str) -> list[str]:
    """
    Build a wsl.exe argv prefix targeting a specific distro (if provided).

    Returns:
        ["wsl.exe", "-d", <distro>, "--"]  or  ["wsl.exe", "--"]
    """
    distro = str(distro).strip()
    if distro:
        return ["wsl.exe", "-d", distro, "--"]
    return ["wsl.exe", "--"]


def win_path_to_wslpath(path: Union[str, Path], *, distro: str) -> str:
    """
    Convert a Windows path to a WSL/Linux path by invoking `wslpath` inside WSL.

    This is more robust than assuming /mnt/<drive>/... because WSL automount roots
    can be customized via /etc/wsl.conf.

    Parameters
    ----------
    path:
        Windows path (absolute is recommended).
    distro:
        WSL distro name passed to wsl.exe -d. If empty, the default distro is used.

    Returns
    -------
    str:
        WSL/Linux path as returned by `wslpath`.

    Raises
    ------
    RuntimeError
        If wsl.exe or wslpath fails. The raised error message includes the argv and any
        captured output for immediate diagnosis.
    """
    p = Path(path)
    try:
        s = str(p.resolve())
    except Exception:
        s = str(p)

    # Use explicit '-u' to request a Unix-style output path. '-a' requests an absolute path.
    argv = _wsl_exe_prefix(distro) + ["wslpath", "-a", "-u", s]
    try:
        out = subprocess.check_output(argv, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to run wslpath because wsl.exe was not found on PATH.\n"
            f"  argv: {argv}\n"
        ) from e
    except subprocess.CalledProcessError as e:
        output = e.output or ""
        raise RuntimeError(
            "Failed to convert Windows path to WSL path via wslpath.\n"
            f"  argv: {argv}\n"
            f"  output:\n{output}\n"
        ) from e

    return out.strip()




# -----------------------------------------------------------------------------
# Repo + worker resolution
# -----------------------------------------------------------------------------

def resolve_repo_root_win() -> Path:
    """
    Resolve the NEBULA repository root directory (Windows side).

    This is the canonical helper used by build_wsl_command() when cfg.wsl.cwd_wsl is
    None/""/"AUTO". We intentionally delegate to nebula_repo_dir_win() so the bridge
    has exactly one repo-root resolution policy.
    """
    return nebula_repo_dir_win()


def nebula_repo_dir_win() -> Path:
    """
    Resolve the NEBULA repository root directory (Windows side).

    Assumed location
    ----------------
    This file is expected to live at:
        <NEBULA_REPO>/Utility/ZODIACAL_LIGHT/NEBULA_ZODIACAL_LIGHT_WSL_BRIDGE.py

    Therefore, the repo root is:
        Path(__file__).resolve().parents[2]
    """
    return Path(__file__).resolve().parents[2]


def resolve_worker_script_paths(
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> Tuple[Path, str]:
    """
    Resolve the worker script path on Windows and compute its WSL mount path.

    Parameters
    ----------
    cfg:
        ZodiacalLightConfig instance. Only cfg.wsl.worker_relpath is used here.

    Returns
    -------
    (worker_win, worker_wsl):
        worker_win:
            Absolute Windows Path to the worker script.
        worker_wsl:
            WSL path (e.g., /mnt/c/...) to the same file.

    Raises
    ------
    FileNotFoundError
        If the worker script does not exist at the resolved location.
    """
    repo_root = resolve_repo_root_win()
    worker_win = (repo_root / Path(cfg.wsl.worker_relpath)).resolve()

    if not worker_win.exists():
        raise FileNotFoundError(
            "ZL WSL worker script not found.\n"
            f"  resolved: {worker_win}\n"
            f"  cfg.wsl.worker_relpath: {cfg.wsl.worker_relpath!r}\n"
            f"  repo_root: {repo_root}\n"
        )

    return worker_win, win_path_to_wsl(worker_win)


# -----------------------------------------------------------------------------
# Temp directory helpers
# -----------------------------------------------------------------------------

def tmp_dir_win(cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG) -> Path:
    """
    Resolve the Windows-side temp directory used for request/response payload files.

    Delegates to Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG.resolve_tmp_dir() so the
    bridge and stage share exactly one path resolution policy.
    """
    return resolve_tmp_dir(cfg)


def ensure_tmp_dir(cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG) -> Path:
    """
    Ensure that the ZL temp directory exists on the Windows side.

    Returns
    -------
    Path
        The ensured directory path.
    """
    d = tmp_dir_win(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_request_response_bases(
    obs_name: str,
    window_index: int,
    *,
    suffix: Optional[str] = None,
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> Tuple[Path, Path]:
    """
    Create deterministic base paths for request and response payloads.

    Returned paths are *base paths* (no extension). The payload I/O layer appends
    .json and .npz.

    Naming
    ------
    The base names are structured to be:
    - deterministic
    - filesystem-safe
    - easy to associate with (observer, window)

    Example
    -------
    obs_name="SBSS (USA 216)", window_index=0, suffix=None =>
        <tmp>/zodi_req__SBSS_USA_216__w0000
        <tmp>/zodi_resp__SBSS_USA_216__w0000

    If suffix="samples" =>
        <tmp>/zodi_req__SBSS_USA_216__w0000__samples
        <tmp>/zodi_resp__SBSS_USA_216__w0000__samples

    Parameters
    ----------
    obs_name:
        Human-readable observer name. Sanitized for filesystem safety.
    window_index:
        Window index (integer).
    suffix:
        Optional additional discriminator. Use when you generate multiple payload
        sets per (observer, window) during development.
    cfg:
        ZL configuration (used only for temp directory resolution).

    Returns
    -------
    (request_base_win, response_base_win)
        Both are Windows Paths without extensions.
    """
    d = ensure_tmp_dir(cfg)

    safe_obs = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(obs_name)).strip("_")
    w = int(window_index)

    base = f"{safe_obs}__w{w:04d}"
    if suffix is not None:
        safe_suf = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(suffix)).strip("_")
        if safe_suf:
            base = f"{base}__{safe_suf}"

    req_base = d / f"zodi_req__{base}"
    resp_base = d / f"zodi_resp__{base}"
    return req_base, resp_base


# -----------------------------------------------------------------------------
# WSL invocation
# -----------------------------------------------------------------------------

def _bash_quote(s: str) -> str:
    """
    Quote a string for safe use as a single token in a POSIX shell command.
    """
    return shlex.quote(str(s))


def _bash_quote_python(python_exe: str) -> str:
    """
    Quote the WSL python executable command safely while preserving '~' expansion.

    Motivation
    ----------
    In bash, quoting a token that begins with '~' prevents tilde expansion.
    Your cfg.wsl.wsl_python default uses '~/venvs/.../python', so we preserve that.

    Policy (fail-fast)
    ------------------
    - If python_exe contains whitespace, we require it to be an absolute path and quote it.
      (Space-containing commands are error-prone in shell contexts; this is strict.)
    - If python_exe begins with '~' and contains no spaces, we return it unquoted.
    - Otherwise, we quote normally.
    """
    s = str(python_exe)
    if any(ch.isspace() for ch in s):
        # Fail-fast: space-containing python commands are not supported unless the
        # caller provides an absolute path (which we then quote).
        if s.startswith("~"):
            raise RuntimeError(
                "cfg.wsl.wsl_python contains whitespace and begins with '~', which is not supported.\n"
                f"  cfg.wsl.wsl_python={python_exe!r}\n"
                "Provide an absolute path without '~' if it contains spaces."
            )
        return _bash_quote(s)

    # Preserve tilde expansion for the common case.
    if s.startswith("~"):
        return s

    return _bash_quote(s)


def build_wsl_command(
    request_base_win: Union[str, Path],
    response_base_win: Union[str, Path],
    *,
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> list[str]:
    """
    Construct the Windows-side subprocess argv list to execute the WSL worker.

    Invocation style
    ----------------
        wsl.exe -d <distro> -- bash -lc "<script>"

    Where <script> performs:
        set -euo pipefail;
        cd <cfg.wsl.cwd_wsl>;
        <cfg.wsl.wsl_python> <worker> <req_base> <resp_base>

    Quoting
    -------
    - All paths passed into bash are shell-quoted with shlex.quote().
    - cfg.wsl.wsl_python is treated specially to preserve '~' expansion when applicable.

    Fail-fast behavior
    ------------------
    Raises FileNotFoundError if required request payload files are missing.

    Parameters
    ----------
    request_base_win, response_base_win:
        Windows base paths (no extension) for request and response payloads.
    cfg:
        ZL configuration containing WSL invocation settings.

    Returns
    -------
    list[str]
        argv list suitable for subprocess.run().
    """
    request_base_win = Path(request_base_win)
    response_base_win = Path(response_base_win)

    # The stage must have written request payloads before invoking WSL.
    req_json = Path(str(request_base_win) + ".json")
    req_npz = Path(str(request_base_win) + ".npz")
    if not req_json.exists() or not req_npz.exists():
        raise FileNotFoundError(
            "Request payload missing (both .json and .npz must exist before invoking WSL worker).\n"
            f"  expected: {req_json}\n"
            f"  expected: {req_npz}\n"
        )

    # Resolve worker script path deterministically.
    _worker_win, worker_wsl = resolve_worker_script_paths(cfg)

    # Determine which WSL distro to target. If empty, wsl.exe uses the default distro.
    distro = str(cfg.wsl.distro).strip()

    # Translate base paths into WSL paths.
    # Prefer calling `wslpath` inside the target distro for robustness, but allow a
    # strict /mnt/<drive>/... fallback if explicitly configured.
    if bool(getattr(cfg.wsl, "use_wslpath", False)):
        req_base_wsl = win_path_to_wslpath(request_base_win.resolve(), distro=distro)
        resp_base_wsl = win_path_to_wslpath(response_base_win.resolve(), distro=distro)
        worker_wsl = win_path_to_wslpath(_worker_win, distro=distro)
    else:
        req_base_wsl = win_path_to_wsl(request_base_win.resolve())
        resp_base_wsl = win_path_to_wsl(response_base_win.resolve())
        # worker_wsl is already /mnt/<drive>/... from resolve_worker_script_paths()
        # under the fallback mapping assumption.

    # Determine the working directory inside WSL.
    # - If cfg.wsl.cwd_wsl is set, use it (must be an absolute WSL path)
    # - Otherwise derive it from the repo root and convert it consistently.
    cwd_cfg = getattr(cfg.wsl, "cwd_wsl", None)
    if cwd_cfg is None or str(cwd_cfg).strip().upper() in ("", "AUTO"):
        repo_root_win = resolve_repo_root_win()
        if bool(getattr(cfg.wsl, "use_wslpath", False)):
            cwd_wsl = win_path_to_wslpath(repo_root_win, distro=distro)
        else:
            cwd_wsl = win_path_to_wsl(repo_root_win)
    else:
        cwd_wsl = str(cwd_cfg)
        if not cwd_wsl.startswith("/"):
            raise RuntimeError(
                "cfg.wsl.cwd_wsl must be an absolute WSL path (beginning with '/').\n"
                f"  cfg.wsl.cwd_wsl={cwd_cfg!r}"
            )

    
    # Build a bash script that will run inside the distro.
    # - set -euo pipefail: exit on error, undefined var, and pipeline failures
    # - cd to the repo root (or any configured working dir) so relative imports behave
    # - run python worker with request/response base paths
    bash_script = (
        "set -euo pipefail; "
        f"cd {_bash_quote(cwd_wsl)}; "
        f"{_bash_quote_python(str(cfg.wsl.wsl_python))} "
        f"{_bash_quote(worker_wsl)} "
        f"{_bash_quote(req_base_wsl)} "
        f"{_bash_quote(resp_base_wsl)}"
    )

    # Build wsl.exe argv.
    # If cfg.wsl.distro is empty/None, we omit -d and use the default distro.
    if distro:
        return ["wsl.exe", "-d", distro, "--", "bash", "-lc", bash_script]

    return ["wsl.exe", "--", "bash", "-lc", bash_script]


def run_wsl_worker(
    request_base_win: Union[str, Path],
    response_base_win: Union[str, Path],
    *,
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
    logger: Optional[logging.Logger] = None,
) -> WSLRunResult:
    """
    Run the Zodiacal Light WSL worker synchronously.

    Parameters
    ----------
    request_base_win:
        Windows base path (no extension) for the request payload.
    response_base_win:
        Windows base path (no extension) for the response payload.
    cfg:
        ZL configuration containing WSL execution parameters.
    logger:
        Optional logger. If omitted, uses the module logger name.

    Returns
    -------
    WSLRunResult
        Captured stdout/stderr and argv on success.

    Raises
    ------
    FileNotFoundError
        - Missing request payload files (.json/.npz)
        - Worker script not found
        - wsl.exe not found (subprocess raises FileNotFoundError)
    WSLWorkerError
        - Non-zero return code from wsl.exe / bash / python / worker
        - Timeout expired
        - Success but response payload files missing
    """
    lg = _get_logger(logger)

    # Ensure temp directory exists. The stage typically does this earlier, but doing it
    # here keeps the bridge deterministic and robust to call ordering.
    ensure_tmp_dir(cfg)

    cmd = build_wsl_command(request_base_win, response_base_win, cfg=cfg)
    lg.info("Invoking ZL WSL worker via: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(cfg.wsl.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired may carry partial stdout/stderr; convert safely.
        stdout = (
            e.stdout.decode(errors="replace")
            if isinstance(e.stdout, (bytes, bytearray))
            else (e.stdout or "")
        )
        stderr = (
            e.stderr.decode(errors="replace")
            if isinstance(e.stderr, (bytes, bytearray))
            else (e.stderr or "")
        )
        raise WSLWorkerError(
            "ZL WSL worker timed out.\n"
            f"  timeout_s: {cfg.wsl.timeout_s}\n"
            f"  command: {' '.join(cmd)}\n",
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            command=cmd,
        ) from e

    result = WSLRunResult(
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        command=cmd,
    )

    if result.returncode != 0:
        raise WSLWorkerError(
            "ZL WSL worker failed (non-zero return code).\n"
            f"  returncode: {result.returncode}\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr (first 4k chars):\n{result.stderr[:4096]}\n",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            command=result.command,
        )

    # Success return code is not sufficient: verify response payload exists.
    response_base_win = Path(response_base_win)
    resp_json = Path(str(response_base_win) + ".json")
    resp_npz = Path(str(response_base_win) + ".npz")
    if not resp_json.exists() or not resp_npz.exists():
        raise WSLWorkerError(
            "WSL worker returned success but response payload files are missing.\n"
            f"  expected: {resp_json}\n"
            f"  expected: {resp_npz}\n"
            f"  command: {' '.join(cmd)}\n",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            command=result.command,
        )

    return result
