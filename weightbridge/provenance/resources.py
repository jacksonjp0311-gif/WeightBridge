"""Portable process resource probes with explicit unsupported-value behavior."""
from __future__ import annotations

import os
import sys


def peak_rss_bytes() -> int | None:
    """Return process peak resident memory where the OS exposes it."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        if os.name == "nt":
            peak = getattr(info, "peak_wset", None)
            return int(peak) if peak is not None else None
    except (ImportError, OSError, AttributeError):
        pass
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError, OSError):
        return None


def current_rss_bytes() -> int | None:
    """Return current process RSS for generation-time host budget checks."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError, AttributeError):
        return None
