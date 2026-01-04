from __future__ import annotations

import logging
from typing import Optional


def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger for the Zodiacal Light stage code.

    Behavior is intentionally identical to the original monolithic stage:
    - If a logger is provided, use it.
    - Otherwise create/use a module logger named "NEBULA_ZODIACAL_LIGHT_STAGE".
    - If the logger has no handlers, attach a StreamHandler with a compact formatter.
    - Default level is INFO.
    """
    if logger is not None:
        return logger
    lg = logging.getLogger("NEBULA_ZODIACAL_LIGHT_STAGE")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg
