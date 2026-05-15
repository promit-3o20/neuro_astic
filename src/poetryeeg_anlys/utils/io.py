"""
utils/io.py
===========
Shared I/O helpers: loading / saving MNE objects and DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import pandas as pd


def load_epochs(path: Path, preload: bool = True) -> mne.Epochs:
    """
    Load an MNE Epochs file (.fif).

    Parameters
    ----------
    path : Path
    preload : bool

    Returns
    -------
    mne.Epochs
    """
    return mne.read_epochs(str(path), preload=preload)


def save_epochs(epochs: mne.Epochs | None, path: Path) -> None:
    """Save epochs to .fif.  No-op when *epochs* is None."""
    if epochs is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(str(path), overwrite=True)


def save_raw(raw: mne.io.Raw, path: Path) -> None:
    """Save a Raw object to .fif."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(path), overwrite=True)


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str = "snappy",
    logger: logging.Logger | None = None,
) -> None:
    """
    Save a DataFrame as a Snappy-compressed Parquet file.

    Parameters
    ----------
    df          : pd.DataFrame
    path        : Path
    compression : str
    logger      : logging.Logger, optional
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), compression=compression)
    if logger:
        logger.info(f"Saved parquet → {path}  shape={df.shape}")
