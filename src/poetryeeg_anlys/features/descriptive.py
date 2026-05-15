"""
features/descriptive.py
========================
Descriptive statistics features (mean, variance, skewness, kurtosis)
computed per epoch across channels or temporal windows.

Stub — implement as needed.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def compute_descriptive_stats(epochs: mne.Epochs) -> pd.DataFrame:
    """
    Compute per-channel descriptive statistics for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs

    Returns
    -------
    pd.DataFrame
        Columns: ``{stat}_{channel}``  e.g. ``'mean_AF7'``.
    """
    data = epochs.get_data()                   # (n_ep, n_ch, n_t)
    ch_names = epochs.ch_names
    stats = {
        "mean":     data.mean(axis=2),
        "std":      data.std(axis=2),
        "variance": data.var(axis=2),
    }
    rows: dict[str, np.ndarray] = {}
    for stat_name, arr in stats.items():
        for ch_idx, ch in enumerate(ch_names):
            rows[f"{stat_name}_{ch}"] = arr[:, ch_idx]
    return pd.DataFrame(rows)
