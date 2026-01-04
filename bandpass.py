from __future__ import annotations

from typing import Any, Dict, Optional

from Configuration.NEBULA_ZODIACAL_LIGHT_CONFIG import (
    ZODIACAL_LIGHT_CONFIG,
    BandpassSpec,
    ZodiacalLightConfig,
)


def build_bandpass_dict(
    *,
    catalog_name: Optional[str],
    catalog_band: Optional[str],
    cfg: ZodiacalLightConfig = ZODIACAL_LIGHT_CONFIG,
) -> Dict[str, Any]:
    """
    Build the request['bandpass'] dict sent to the WSL backend.

    Single source of truth
    ----------------------
    In the current pipeline, the authoritative bandpass is defined by:

        cfg.catalog.bandpass   (BandpassSpec)

    The `catalog_name` / `catalog_band` inputs are not used for routing. They are
    only used for optional fail-fast validation against:

        cfg.catalog.catalog_name_expected
        cfg.catalog.catalog_band_expected

    Backend contract (WSL)
    ----------------------
    The backend validates a JSON-serializable dict with:
      bandpass.mode in {"tophat","svo"}

    If mode == "svo":
      bandpass.filter_id required

    If mode == "tophat":
      bandpass.lambda_min_nm and bandpass.lambda_max_nm required (nm)
      bandpass.lambda_eff_nm optional
    """
    # Normalize incoming metadata
    name = None if catalog_name is None else str(catalog_name)
    band = None if catalog_band is None else str(catalog_band)

    # Optional fail-fast checks against configuration expectations
    expected_name = getattr(cfg.catalog, "catalog_name_expected", None)
    expected_band = getattr(cfg.catalog, "catalog_band_expected", None)

    if expected_name is not None and name is not None and name != str(expected_name):
        raise ValueError(
            "Catalog name mismatch against configuration.\n"
            f"  catalog_name (scene)={name!r}\n"
            f"  catalog_name_expected (cfg)={str(expected_name)!r}"
        )

    if expected_band is not None and band is not None and band != str(expected_band):
        raise ValueError(
            "Catalog band mismatch against configuration.\n"
            f"  catalog_band (scene)={band!r}\n"
            f"  catalog_band_expected (cfg)={str(expected_band)!r}"
        )

    # Authoritative bandpass spec
    spec: BandpassSpec = cfg.catalog.bandpass

    # Convert to backend request schema and validate
    bp = _bandpass_spec_to_request(spec)
    return _validate_bandpass_request_dict(bp)



def _bandpass_spec_to_request(spec: BandpassSpec) -> Dict[str, Any]:
    """
    Convert a Windows-side BandpassSpec (config object) into the WSL request dict.

    Supported spec modes (config-side)
    ----------------------------------
    - "svo_id"    -> request mode "svo" with key "filter_id"
    - "tophat_nm" -> request mode "tophat" with lambda_min_nm/lambda_max_nm (+lambda_eff_nm)

    Any other mode raises (fail-fast), including deprecated "curve_file".
    """
    mode = str(spec.mode).strip().lower()

    if mode == "svo_id":
        if not spec.svo_filter_id:
            raise ValueError("BandpassSpec(mode='svo_id') requires svo_filter_id to be set.")
        return {
            "mode": "svo",
            "filter_id": str(spec.svo_filter_id),
            "description": str(getattr(spec, "description", "")),
        }

    if mode == "tophat_nm":
        if spec.center_nm is None or spec.width_nm is None:
            raise ValueError("BandpassSpec(mode='tophat_nm') requires center_nm and width_nm.")
        center = float(spec.center_nm)
        width = float(spec.width_nm)
        if width <= 0.0:
            raise ValueError("BandpassSpec.width_nm must be > 0.")
        lam_min = center - 0.5 * width
        lam_max = center + 0.5 * width
        return {
            "mode": "tophat",
            "lambda_min_nm": float(lam_min),
            "lambda_max_nm": float(lam_max),
            "lambda_eff_nm": float(center),
            "description": str(getattr(spec, "description", "")),
        }

    if mode == "curve_file":
        raise ValueError(
            "BandpassSpec(mode='curve_file') is not supported in the current pipeline. "
            "Remove this option from the config."
        )

    raise ValueError(f"Unsupported BandpassSpec.mode: {spec.mode!r}")


def _validate_bandpass_request_dict(bp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the bandpass dict against the WSL backend's parsing contract.

    This ensures the JSON we write on Windows will be accepted by the WSL backend
    without implicit coercion or silent defaulting.

    Returns the dict (potentially with normalized values) on success.

    Raises
    ------
    KeyError / ValueError / TypeError
        If required keys are missing or values are invalid.
    """
    if not isinstance(bp, dict):
        raise TypeError("bandpass must be a dict")

    mode = str(bp.get("mode", "")).strip().lower()
    if mode not in ("tophat", "svo"):
        raise ValueError("bandpass.mode must be 'tophat' or 'svo'")

    if mode == "svo":
        fid = bp.get("filter_id", None)
        if fid is None or str(fid).strip() == "":
            raise KeyError("bandpass.mode='svo' requires non-empty 'filter_id'")
        bp["filter_id"] = str(fid).strip()
        bp["mode"] = "svo"
        return bp

    # mode == "tophat"
    try:
        lam_min = float(bp["lambda_min_nm"])
        lam_max = float(bp["lambda_max_nm"])
    except KeyError as e:
        raise KeyError("bandpass.mode='tophat' requires 'lambda_min_nm' and 'lambda_max_nm'") from e
    except Exception as e:
        raise TypeError("lambda_min_nm/lambda_max_nm must be float-like") from e

    if lam_max <= lam_min:
        raise ValueError("tophat requires lambda_max_nm > lambda_min_nm")

    bp["lambda_min_nm"] = lam_min
    bp["lambda_max_nm"] = lam_max
    bp["mode"] = "tophat"

    # lambda_eff_nm optional; if present, validate as float-like.
    if "lambda_eff_nm" in bp and bp["lambda_eff_nm"] is not None:
        try:
            bp["lambda_eff_nm"] = float(bp["lambda_eff_nm"])
        except Exception as e:
            raise TypeError("lambda_eff_nm must be float-like if provided") from e

    return bp
