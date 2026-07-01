"""Shared utilities: logging, directory creation, version info."""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"

    logger = logging.getLogger("feature_selection")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Use UTF-8 writer to avoid cp1252 encoding errors on Windows
    import io
    stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ch = logging.StreamHandler(stdout_utf8)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def make_dirs(base: Path, group: str) -> dict:
    """Create all required output subdirectories for a group."""
    dirs = {
        "base": base,
        "data_summary": base / "data_summary",
        "group": base / group,
        "preprocessing": base / group / "preprocessing",
        "stability": base / group / "stability_selection",
        "subject_capped": base / group / "subject_capped",
        "boruta": base / group / "boruta",
        "final": base / group / "final",
        "plots": base / group / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_version_info() -> dict:
    """Return dict of key package versions."""
    import numpy as np
    import pandas as pd
    import sklearn
    import scipy
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import seaborn as sns
        versions["seaborn"] = sns.__version__
    except ImportError:
        versions["seaborn"] = "not installed"
    try:
        import boruta
        versions["boruta"] = boruta.__version__
    except Exception:
        versions["boruta"] = "not installed"
    return versions
