"""
features/bandpower.py
=====================
Welch PSD computation and log-ratio baseline normalisation.

Implements Steps 2–3 of the ROI band-power pipeline (from roi_features.py).

Step 2 — Compute one-sided Welch PSD for each temporal segment.
Step 3 — Apply log-ratio baseline normalisation at the spectral level
          BEFORE band-averaging (critical ordering — see Query 4).
"""

from __future__ import annotations

from typing import Tuple

import mne
import numpy as np

from poetryeeg_anlys.config.constants import (
    SFREQ, WELCH_N_FFT, WELCH_N_OVERLAP, EPS,
)


def compute_segment_psd(
    epochs: mne.Epochs,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a fixed temporal segment and compute its one-sided Welch PSD.

    Welch parameters match ``SpectralPower.m``:
        window  = 512 samples (1 s @ 512 Hz)
        overlap = 256 samples (50 %)
        nfft    = 512 → frequency resolution = 1 Hz, 257 one-sided bins

    Parameters
    ----------
    epochs : mne.Epochs
        Full-length epochs (must span at least *tmin*–*tmax*).
    tmin, tmax : float
        Segment boundaries in seconds.

    Returns
    -------
    psds : np.ndarray, shape (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray, length 257
    """
    seg = epochs.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

    psd_obj = seg.compute_psd(
        method="welch",
        fmin=0.0,
        fmax=SFREQ / 2,
        n_fft=WELCH_N_FFT,
        n_overlap=WELCH_N_OVERLAP,
        verbose=False,
    )

    psds  = psd_obj.get_data()   # (n_epochs, n_channels, n_freqs)
    freqs = psd_obj.freqs

    return psds, freqs


def log_baseline_normalise(
    pow_cond: np.ndarray,
    pow_bl: np.ndarray,
) -> np.ndarray:
    """
    Apply log-ratio baseline correction at the spectral level.

    Formula (matches ``SpectralPower.m``)::

        pow_norm = 10·log10(pow_cond + EPS) − 10·log10(pow_bl + EPS)

    Normalisation is performed **before** band-averaging (Query 4).
    Reversing the order produces different results.

    Parameters
    ----------
    pow_cond : np.ndarray, shape (n_epochs, n_channels, n_freqs)
        Condition PSD.
    pow_bl : np.ndarray, shape (n_epochs, n_channels, n_freqs)
        Baseline PSD.

    Returns
    -------
    np.ndarray
        Normalised power in dB, same shape as inputs.
    """
    return 10.0 * np.log10(pow_cond + EPS) - 10.0 * np.log10(pow_bl + EPS)
