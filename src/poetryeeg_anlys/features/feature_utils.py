"""
features/feature_utils.py
==========================
Shared utility functions for feature extraction modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_features(df: pd.DataFrame, path: Path, compression: str = "snappy") -> None:
    """Save a feature DataFrame to a Snappy-compressed Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), compression=compression)


def already_processed(subject: str, output_dir: Path) -> bool:
    """Return True if the subject's ROI feature Parquet already exists."""
    return (output_dir / f"{subject}_roi_bpfeatures.parquet").exists()
