"""
features/roi_bandpower.py
=========================
ROI band-power feature extraction pipeline.

Refactored from ``roi_features.py`` — all logic preserved exactly.

Algorithm (critical ordering — Query 4)
---------------------------------------
1.  Pick EEG channels only.
2.  Compute Welch PSD per segment (baseline / early / late).
3.  Log-ratio baseline normalisation at the spectral level.
4.  Band-average the normalised spectra (after normalisation).
5.  Flatten per-channel band power → saved to ``bp/``.
6.  Average band power within each ROI cluster.
7.  Attach metadata; return both DataFrames.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd

from poetryeeg_anlys.config.constants import (
    FREQ_BANDS, BANDS_FOR_FEATURES, ROI_MAP, SEGMENTS,
)
from poetryeeg_anlys.utils.logger import get_logger
from .bandpower import compute_segment_psd, log_baseline_normalise


# Module-level logger (overridable by callers)
_log = get_logger("roi_bandpower")


# ---------------------------------------------------------------------------
# Step 1 – channel selection
# ---------------------------------------------------------------------------

def pick_eeg_channels(epochs: mne.Epochs) -> mne.Epochs:
    """Keep only scalp EEG channels (drop EXG / EOG / ECG)."""
    return epochs.copy().pick_types(eeg=True)


# ---------------------------------------------------------------------------
# Step 4 – band averaging  (after normalisation)
# ---------------------------------------------------------------------------

def band_average(
    psds_norm: np.ndarray,
    freqs: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
    logger: logging.Logger = _log,
) -> Dict[str, np.ndarray]:
    """
    Average normalised PSD within each frequency band.

    Parameters
    ----------
    psds_norm : np.ndarray, shape (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray
    bands : dict  {name: (fmin, fmax)}
    logger : logging.Logger

    Returns
    -------
    dict  {band_name: np.ndarray (n_epochs, n_channels)}
    """
    band_powers: Dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        if not idx.any():
            logger.warning(
                f"No frequency bins for band '{band_name}' "
                f"({fmin}–{fmax} Hz). Skipping."
            )
            continue
        band_powers[band_name] = psds_norm[:, :, idx].mean(axis=2)

    return band_powers


# ---------------------------------------------------------------------------
# Step 5 – ROI spatial averaging
# ---------------------------------------------------------------------------

def roi_average(
    band_powers: Dict[str, np.ndarray],
    ch_names: List[str],
    roi_map: Dict[str, List[str]],
    phase: str,
    bands_for_features: List[str],
    logger: logging.Logger = _log,
) -> Dict[str, np.ndarray]:
    """
    Average channel-wise band power across ROI electrode clusters.

    Parameters
    ----------
    band_powers : dict  {band_name: (n_epochs, n_channels)}
    ch_names : list[str]
    roi_map : dict  {roi_name: [electrode, ...]}
    phase : str  ``'early'`` or ``'late'``
    bands_for_features : list[str]  (gamma2 excluded)
    logger : logging.Logger

    Returns
    -------
    dict  {column_name: np.ndarray (n_epochs,)}
    """
    ch_index = {ch: i for i, ch in enumerate(ch_names)}
    roi_features: Dict[str, np.ndarray] = {}

    for band_name in bands_for_features:
        if band_name not in band_powers:
            logger.warning(f"Band '{band_name}' missing from band_powers — skipped.")
            continue

        bp = band_powers[band_name]  # (n_epochs, n_channels)

        for roi_name, electrodes in roi_map.items():
            available = [e for e in electrodes if e in ch_index]
            missing   = [e for e in electrodes if e not in ch_index]

            if missing:
                logger.warning(
                    f"[{phase}|{band_name}|{roi_name}] "
                    f"Missing electrodes: {missing}. "
                    f"Averaging over {len(available)}/{len(electrodes)}."
                )
            if not available:
                logger.error(
                    f"[{phase}|{band_name}|{roi_name}] "
                    "No electrodes available — ROI omitted."
                )
                continue

            indices = [ch_index[e] for e in available]
            roi_features[f"{phase}_{band_name}_{roi_name}"] = (
                bp[:, indices].mean(axis=1)
            )

    return roi_features


# ---------------------------------------------------------------------------
# Per-channel feature flattening (saved before ROI step)
# ---------------------------------------------------------------------------

def flatten_channel_features(
    early_bands: Dict[str, np.ndarray],
    late_bands: Dict[str, np.ndarray],
    ch_names: List[str],
    bands_for_features: List[str],
    logger: logging.Logger = _log,
) -> pd.DataFrame:
    """
    Flatten per-channel band power into a 2-D feature DataFrame.

    Column naming: ``{phase}_{band}_{channel}``  (e.g. ``'early_alpha_AF7'``).

    Parameters
    ----------
    early_bands, late_bands : dict  {band: (n_epochs, n_channels)}
    ch_names : list[str]
    bands_for_features : list[str]
    logger : logging.Logger

    Returns
    -------
    pd.DataFrame  shape (n_epochs, n_phases × n_bands × n_channels)
    """
    cols: Dict[str, np.ndarray] = {}

    for phase, band_powers in [("early", early_bands), ("late", late_bands)]:
        for band_name in bands_for_features:
            if band_name not in band_powers:
                logger.warning(
                    f"[flatten_channel] Band '{band_name}' missing — skipped."
                )
                continue
            bp = band_powers[band_name]
            for ch_idx, ch in enumerate(ch_names):
                cols[f"{phase}_{band_name}_{ch}"] = bp[:, ch_idx]

    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_roi_bandpower_features(
    epochs: mne.Epochs,
    bands: Dict[str, Tuple[float, float]] = FREQ_BANDS,
    bands_for_features: List[str] = BANDS_FOR_FEATURES,
    roi_map: Dict[str, List[str]] = ROI_MAP,
    logger: logging.Logger = _log,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full ROI band-power feature extraction pipeline.

    Parameters
    ----------
    epochs : mne.Epochs  (must span at least -4 to +10 s)
    bands : dict  (all six bands including gamma2)
    bands_for_features : list  (gamma2 excluded from output)
    roi_map : dict
    logger : logging.Logger

    Returns
    -------
    bp_df : pd.DataFrame
        Per-channel features. Columns: ``{phase}_{band}_{ch}`` + metadata.
    roi_df : pd.DataFrame
        ROI-averaged features. Columns: ``{phase}_{band}_{roi}`` + metadata.
    """
    logger.info("Starting ROI band-power feature extraction")

    # Step 1 – channel selection
    epochs   = pick_eeg_channels(epochs)
    ch_names = epochs.ch_names
    logger.info(f"EEG channels: {len(ch_names)}")

    # Step 2 – Welch PSD per segment
    logger.info("Computing Welch PSD for each temporal segment …")
    pow_bl,    freqs = compute_segment_psd(epochs, *SEGMENTS["baseline"])
    pow_early, _     = compute_segment_psd(epochs, *SEGMENTS["early"])
    pow_late,  _     = compute_segment_psd(epochs, *SEGMENTS["late"])
    logger.info(f"PSD shape per segment: {pow_bl.shape}")

    # Step 3 – log-ratio baseline normalisation (before band-averaging)
    logger.info("Applying log-ratio baseline normalisation …")
    pow_early_norm = log_baseline_normalise(pow_early, pow_bl)
    pow_late_norm  = log_baseline_normalise(pow_late,  pow_bl)

    # Step 4 – band averaging (after normalisation — critical ordering)
    logger.info("Band-averaging normalised spectra …")
    early_bands = band_average(pow_early_norm, freqs, bands, logger)
    late_bands  = band_average(pow_late_norm,  freqs, bands, logger)

    # Step 4b – per-channel DataFrame (saved to bp/ before ROI step)
    bp_df = flatten_channel_features(
        early_bands, late_bands, ch_names, bands_for_features, logger
    )
    logger.info(f"Per-channel feature matrix: {bp_df.shape}")

    # Step 5 – ROI spatial averaging
    logger.info("Averaging band power within ROI clusters …")
    early_roi = roi_average(early_bands, ch_names, roi_map, "early", bands_for_features, logger)
    late_roi  = roi_average(late_bands,  ch_names, roi_map, "late",  bands_for_features, logger)
    roi_df    = pd.DataFrame({**early_roi, **late_roi})

    n_expected = len(bands_for_features) * len(roi_map) * 2
    logger.info(
        f"ROI feature matrix: {roi_df.shape[0]} epochs × {roi_df.shape[1]} features "
        f"(expected {n_expected} feature columns)"
    )

    # Attach epoch metadata
    if epochs.metadata is not None:
        for col in epochs.metadata.columns:
            bp_df[col]  = epochs.metadata[col].values
            roi_df[col] = epochs.metadata[col].values
        logger.info(f"Metadata columns attached: {list(epochs.metadata.columns)}")

    # Quality checks
    for label, df in [("bp", bp_df), ("roi", roi_df)]:
        nan_count = df.isna().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if nan_count:
            logger.warning(f"[{label}] NaN values detected: {nan_count}")
        if inf_count:
            logger.warning(f"[{label}] Inf values detected: {inf_count}")

    logger.info("ROI band-power feature extraction complete.")
    return bp_df, roi_df
