"""
utils/logger.py
===============
Reusable logger factory used throughout the package.

All loggers write to both the console (StreamHandler) and a per-context log
file (FileHandler).  Duplicate handlers are suppressed so that re-importing
this module does not multiply log lines.

Usage
-----
    from poetryeeg_anlys.utils.logger import get_logger

    # Per-subject logger (writes to LOG_DIR/preprs/sub-021_log.log)
    logger = get_logger("sub-021", log_dir=Paths.LOG_PREPROCESS)

    # Named logger without a file handler (console only)
    logger = get_logger("features")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """
    Return a logger that writes to console and (optionally) a log file.

    Parameters
    ----------
    name : str
        Logger name — also used as the stem of the log filename when
        ``log_dir`` is provided.
    log_dir : Path, optional
        Directory in which to create ``{name}_log.log``.
        If *None*, only a StreamHandler is attached.
    level : int or str
        Logging level (e.g. ``logging.DEBUG``, ``"INFO"``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Honour the most verbose level that has been requested
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}_log.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
