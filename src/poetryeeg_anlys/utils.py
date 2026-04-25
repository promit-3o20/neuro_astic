import logging
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
    clear_handlers: bool = True,
) -> logging.Logger:
    """
    Create and configure a reusable logger.

    Parameters
    ----------
    name : str
        Name of the logger (e.g., script name, subject ID).
    log_file : str | Path | None, optional
        Path to log file. If None, file logging is disabled.
    level : int, optional
        Logging level (default: logging.INFO).
    console : bool, optional
        If True, log messages are also printed to console.
    clear_handlers : bool, optional
        If True, remove existing handlers to prevent duplication.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    - Supports both console and file logging.
    - Safe for reuse across scripts.
    - Prevents duplicate handlers unless explicitly disabled.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if clear_handlers and logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
