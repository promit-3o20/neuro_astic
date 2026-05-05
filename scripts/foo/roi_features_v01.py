"""
ROI-First EEG Spectral Feature Extraction
========================================

Author : Pramit Biswas
Version : v0.1

Pipeline
--------
1. Select EEG channels
2. Segment epochs into sliding windows
3. Compute PSD using Welch method
4. Extract channel-level absolute band power
5. Aggregate channels into ROI
6. Compute ROI absolute band power
7. Compute ROI relative band power
8. Compute ROI band ratios
9. Compute ROI hemispheric asymmetry
10. Average across time windows (baseline, early, late)
11. Apply log baseline normalization
12. Flatten features
13. Attach metadata
14. Save as parquet
"""

from __future__ import annotations

import os
import glob
import logging
import numpy as np
import pandas as pd
import mne

from pathlib import Path
from typing import Dict, Tuple


# =========================================================
# CONFIG
# =========================================================

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

ROI_MAP = {
    "CL1_lf": ["Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7"],
    "CL2_lft": ["FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "C1", "CP1", "CP3", "CP5", "TP7"],
    "CL3_lpo": ["P7", "P5", "P3", "P1", "PO7", "PO3", "O1"],

    "CL4_rf": ["Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8"],
    "CL5_rft": ["FT8", "FC6", "FC4", "FC2", "T8", "C6", "C4", "C2", "CP2", "CP4", "CP6", "TP8"],
    "CL6_rpo": ["P8", "P6", "P4", "P2", "PO8", "PO4", "O2"],
}

ROI_PAIRS = [
    ("CL1_lf", "CL4_rf"),
    ("CL2_lft", "CL5_rft"),
    ("CL3_lpo", "CL6_rpo"),
]

RATIO_PAIRS = {
    "theta_beta": ("theta", "beta"),
    "alpha_beta": ("alpha", "beta"),
    "theta_alpha": ("theta", "alpha"),
    "delta_beta": ("delta", "beta"),
    "gamma_beta": ("gamma", "beta"),
}

EPS = 1e-10


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
LOG_DIR = (BASE_DIR / "../../logs/fetrs").resolve()

FETRS_DIR = DATA_DIR / "features"
EXTRCT_FETRS = DATA_DIR / "intrmd_data/raw_label"

for d in [FETRS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOGGER
# =========================================================

def setup_logger(log_file: str, level=logging.INFO):
    logger = logging.getLogger("ROIFeatureLogger")
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

logger = setup_logger(str(LOG_DIR / "roi_features1.log"))


# =========================================================
# CORE FUNCTIONS
# =========================================================

def pick_eeg_channels(epochs: mne.Epochs) -> mne.Epochs:
    return epochs.copy().pick_types(eeg=True)


def get_time_windows(tmin: float, tmax: float, window_size: float, step: float):
    windows = []
    t = tmin
    while t + window_size <= tmax:
        windows.append(t)
        t += step
    return np.array(windows)


def compute_psd(epochs: mne.Epochs):
    psd = epochs.compute_psd(
        method="welch",
        fmin=0.5,
        fmax=45,
        n_fft=1024,
        n_overlap=512,
        verbose=False
    )
    return psd.get_data(), psd.freqs


def extract_band_power(psds, freqs, bands=FREQ_BANDS):
    band_powers = []

    for _, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        power = psds[:, :, idx].mean(axis=2)
        band_powers.append(power)

    return np.stack(band_powers, axis=-1)  # (epochs, channels, bands)


def aggregate_to_roi(abs_power, ch_names, roi_map=ROI_MAP):
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    roi_features = []
    roi_names = []

    for roi, channels in roi_map.items():
        valid_idx = [ch_idx[ch] for ch in channels if ch in ch_idx]

        if len(valid_idx) == 0:
            continue

        roi_power = abs_power[:, valid_idx, :].mean(axis=1)
        roi_features.append(roi_power)
        roi_names.append(roi)

    return np.stack(roi_features, axis=1), roi_names  # (epochs, rois, bands)


def compute_relative_bandpower(roi_power):
    total = roi_power.sum(axis=-1, keepdims=True) + EPS
    return roi_power / total


def compute_band_ratios(roi_power, bands=FREQ_BANDS):
    band_names = list(bands.keys())
    band_idx = {b: i for i, b in enumerate(band_names)}

    ratio_features = []
    ratio_names = []

    for name, (num_band, den_band) in RATIO_PAIRS.items():
        num = roi_power[:, :, band_idx[num_band]]
        den = roi_power[:, :, band_idx[den_band]] + EPS
        ratio = num / den
        ratio_features.append(ratio[..., np.newaxis])
        ratio_names.append(name)

    return np.concatenate(ratio_features, axis=-1), ratio_names  # (epochs, rois, ratios)


def compute_roi_asymmetry(roi_power, roi_names, bands=FREQ_BANDS):
    roi_idx = {r: i for i, r in enumerate(roi_names)}
    asym_features = []
    asym_names = []

    for left, right in ROI_PAIRS:
        if left not in roi_idx or right not in roi_idx:
            continue

        li, ri = roi_idx[left], roi_idx[right]

        L = roi_power[:, li, :]
        R = roi_power[:, ri, :]

        asym = (L - R) / (L + R + EPS)
        asym_features.append(asym)

        for band in bands.keys():
            asym_names.append(f"asym_{band}_{left}_{right}")

    if not asym_features:
        return np.empty((roi_power.shape[0], 0)), []

    return np.concatenate(asym_features, axis=1), asym_names


def compute_windowed_roi_features(
    epochs: mne.Epochs,
    tmin: float,
    tmax: float,
    window_size: float = 2.0,
    overlap: float = 0.5,
):
    step = window_size - overlap
    windows = get_time_windows(tmin, tmax, window_size, step)

    all_abs = []
    all_rel = []
    all_ratio = []
    all_asym = []

    epochs = pick_eeg_channels(epochs)
    ch_names = epochs.ch_names

    for start in windows:
        ep = epochs.copy().crop(tmin=start, tmax=start + window_size)

        psds, freqs = compute_psd(ep)
        abs_power = extract_band_power(psds, freqs)
        roi_abs, roi_names = aggregate_to_roi(abs_power, ch_names)

        roi_rel = compute_relative_bandpower(roi_abs)
        roi_ratio, ratio_names = compute_band_ratios(roi_abs)
        roi_asym, asym_names = compute_roi_asymmetry(roi_abs, roi_names)

        all_abs.append(roi_abs)
        all_rel.append(roi_rel)
        all_ratio.append(roi_ratio)
        all_asym.append(roi_asym)

    return (
        np.stack(all_abs),
        np.stack(all_rel),
        np.stack(all_ratio),
        np.stack(all_asym),
        roi_names,
        ratio_names,
        asym_names,
    )


def log_baseline_normalize(power, baseline):
    return np.log((power + EPS) / (baseline + EPS))


# =========================================================
# MAIN FEATURE EXTRACTION
# =========================================================

def extract_roi_features(epochs: mne.Epochs) -> pd.DataFrame:
    logger.info("Starting ROI-first feature extraction")

    (
        base_abs, base_rel, base_ratio, _, roi_names, ratio_names, asym_names
    ) = compute_windowed_roi_features(epochs, -4, 0)

    (
        early_abs, early_rel, early_ratio, early_asym, _, _, _
    ) = compute_windowed_roi_features(epochs, 0, 5)

    (
        late_abs, late_rel, late_ratio, late_asym, _, _, _
    ) = compute_windowed_roi_features(epochs, 5, 10)

    # average across windows
    base_abs = base_abs.mean(axis=0)
    early_abs = early_abs.mean(axis=0)
    late_abs = late_abs.mean(axis=0)

    base_rel = base_rel.mean(axis=0)
    early_rel = early_rel.mean(axis=0)
    late_rel = late_rel.mean(axis=0)

    base_ratio = base_ratio.mean(axis=0)
    early_ratio = early_ratio.mean(axis=0)
    late_ratio = late_ratio.mean(axis=0)

    early_asym = early_asym.mean(axis=0)
    late_asym = late_asym.mean(axis=0)

    # baseline normalize
    early_abs = log_baseline_normalize(early_abs, base_abs)
    late_abs = log_baseline_normalize(late_abs, base_abs)

    early_rel = log_baseline_normalize(early_rel, base_rel)
    late_rel = log_baseline_normalize(late_rel, base_rel)

    early_ratio = log_baseline_normalize(early_ratio, base_ratio)
    late_ratio = log_baseline_normalize(late_ratio, base_ratio)

    # feature names
    abs_early_names = [f"early_abs_{band}_{roi}" for roi in roi_names for band in FREQ_BANDS.keys()]
    abs_late_names  = [f"late_abs_{band}_{roi}" for roi in roi_names for band in FREQ_BANDS.keys()]

    rel_early_names = [f"early_rel_{band}_{roi}" for roi in roi_names for band in FREQ_BANDS.keys()]
    rel_late_names  = [f"late_rel_{band}_{roi}" for roi in roi_names for band in FREQ_BANDS.keys()]

    ratio_early_names = [f"early_ratio_{ratio}_{roi}" for roi in roi_names for ratio in ratio_names]
    ratio_late_names  = [f"late_ratio_{ratio}_{roi}" for roi in roi_names for ratio in ratio_names]

    asym_early_names = [f"early_{name}" for name in asym_names]
    asym_late_names  = [f"late_{name}" for name in asym_names]

    # flatten
    early_abs_flat = early_abs.reshape(early_abs.shape[0], -1)
    late_abs_flat = late_abs.reshape(late_abs.shape[0], -1)

    early_rel_flat = early_rel.reshape(early_rel.shape[0], -1)
    late_rel_flat = late_rel.reshape(late_rel.shape[0], -1)

    early_ratio_flat = early_ratio.reshape(early_ratio.shape[0], -1)
    late_ratio_flat = late_ratio.reshape(late_ratio.shape[0], -1)

    early_asym_flat = early_asym
    late_asym_flat = late_asym

    # concatenate
    features = np.concatenate([
        early_abs_flat,
        late_abs_flat,
        early_rel_flat,
        late_rel_flat,
        early_ratio_flat,
        late_ratio_flat,
        early_asym_flat,
        late_asym_flat,
    ], axis=1)

    feature_names = (
        abs_early_names + abs_late_names +
        rel_early_names + rel_late_names +
        ratio_early_names + ratio_late_names +
        asym_early_names + asym_late_names
    )

    assert features.shape[1] == len(feature_names), "Feature mismatch"

    df = pd.DataFrame(features, columns=feature_names)

    if epochs.metadata is not None:
        for col in epochs.metadata.columns:
            df[col] = epochs.metadata[col].values

    logger.info(f"Feature extraction completed | Shape: {df.shape}")
    return df


# =========================================================
# IO
# =========================================================

def save_features(df: pd.DataFrame, path: str):
    df.to_parquet(path, compression="snappy")
    logger.info(f"Saved features -> {path}")


def process_all_files(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "sub-*_whole_epo.fif")))
    logger.info(f"Found {len(files)} files")

    all_dfs = []

    for file_path in files:
        subject = os.path.basename(file_path).split("_")[0]
        logger.info(f"Processing subject: {subject}")

        try:
            epochs = mne.read_epochs(file_path, preload=True)

            df = extract_roi_features(epochs)
            df["subject"] = subject

            save_path = os.path.join(output_dir, f"{subject}_roi_features.parquet")
            save_features(df, save_path)

            all_dfs.append(df)

        except Exception as e:
            logger.error(f"Error in {subject}: {e}", exc_info=True)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_path = os.path.join(output_dir, "all_subjects_roifeatures.parquet")
        save_features(final_df, final_path)
        logger.info(f"Final dataset shape: {final_df.shape}")


# =========================================================
# ENTRY
# =========================================================

def main():
    INPUT_DIR = str(EXTRCT_FETRS)
    OUTPUT_DIR = str(FETRS_DIR / "roi_ftrs1")
    process_all_files(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()