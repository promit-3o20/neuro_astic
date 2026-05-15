"""
config/settings.py
==================
Runtime-tunable settings.

Values here are **defaults** that can be overridden at runtime via environment
variables or by mutating the ``Settings`` object directly before calling any
pipeline function.

Example
-------
    import os
    os.environ["EEG_DEBUG"] = "1"           # single-subject mode
    os.environ["EEG_DEBUG_SUBJECT"] = "sub-021"

    from poetryeeg_anlys.config.settings import Settings
    print(Settings.DEBUG)  # True
"""

import os
from poetryeeg_anlys.config.paths import Paths

class Settings:
    """Flat namespace of runtime-tunable options."""

    # ── Execution mode ────────────────────────────────────────────────────
    DEBUG: bool            = os.environ.get("EEG_DEBUG", "0") == "1"
    DEBUG_SUBJECT: str     = os.environ.get("EEG_DEBUG_SUBJECT", "sub-021")

    # ── Preprocessing toggles ─────────────────────────────────────────────
    ENABLE_EPOCH_REJECTION: bool = (
        os.environ.get("EEG_REJECT_EPOCHS", "0") == "1"
    )
    ENABLE_REST_EPOCHING: bool = (
        os.environ.get("EEG_REST_EPOCHS", "0") == "1"
    )

    # ── Feature extraction ────────────────────────────────────────────────
    PARQUET_COMPRESSION: str = os.environ.get("EEG_PARQUET_COMPRESSION", "snappy")

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.environ.get("EEG_LOG_LEVEL", "INFO")

    # ── ML ────────────────────────────────────────────────────────────────
    RANDOM_STATE: int = int(os.environ.get("EEG_RANDOM_STATE", "42"))
    CV_FOLDS:     int = int(os.environ.get("EEG_CV_FOLDS", "5"))
