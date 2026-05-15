"""
utils/validation.py
===================
Reusable validation / guard functions shared across sub-packages.
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd


def require_file(path: Path, label: str = "") -> None:
    """Raise FileNotFoundError if *path* does not exist."""
    if not path.exists():
        tag = f" ({label})" if label else ""
        raise FileNotFoundError(f"Required file not found{tag}: {path}")


def require_columns(df: pd.DataFrame, columns: list[str], source: str = "") -> None:
    """Raise ValueError for any column missing from *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        tag = f" in {source}" if source else ""
        raise ValueError(f"Missing columns{tag}: {missing}")


def require_trial_count(df: pd.DataFrame, expected: int, subject: str = "") -> None:
    """Raise ValueError if *df* does not have *expected* rows."""
    if len(df) != expected:
        tag = f" for {subject}" if subject else ""
        raise ValueError(
            f"Expected {expected} trials{tag}, got {len(df)}"
        )


def require_marker(event_id: dict, marker: str, subject: str = "") -> None:
    """Raise ValueError if *marker* string is absent from *event_id*."""
    if marker not in event_id:
        tag = f" [{subject}]" if subject else ""
        raise ValueError(
            f"Marker '{marker}' not found in annotations{tag}. "
            f"Available: {list(event_id.keys())}"
        )
