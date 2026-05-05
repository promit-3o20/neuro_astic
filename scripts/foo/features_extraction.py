"""
Band Power Features Extraction Script

Author : Pramit Biswas
Version : v0.0

===========================================================
Algorithm
-----------------------------------------------------------
1. Select EEG channels
2. Segment epochs into sliding windows
3. Compute PSD using Welch method
4. Extract band power (delta, theta, alpha, beta, gamma)
5. Average across time windows (baseline, early, late)
6. Apply log baseline normalization
7. Flatten features
8. Attach metadata
9. Save as parquet
===========================================================
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import mne
from typing import Dict, Tuple
from pathlib import Path

# ==============================
# CONFIG
# ==============================

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
LOG_DIR = (BASE_DIR / "../../logs/fetrs").resolve()

FETRS_DIR = DATA_DIR / "features/bp0"
EXTRCT_FETRS = DATA_DIR/ "intrmd_data/raw_label"

# create all directories (including logs)
for d in [
    FETRS_DIR,
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
# ==============================
# LOGGER SETUP
# ==============================


def setup_logger(log_file: str , level=logging.INFO):
    """
    Initialize and configure a logger.

    Parameters
    ----------
    log_file : str
        Path to log file.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.

    Notes
    -----
    - Logs are written to both console and file.
    - Existing handlers are cleared to avoid duplication.
    """
    logger = logging.getLogger("BandPowerLogger")
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


logger = setup_logger(str(LOG_DIR / "bandpower.log"))


# ==============================
# CORE FUNCTIONS
# ==============================


def pick_eeg_channels(epochs: mne.Epochs) -> mne.Epochs:
    """
    Select only EEG channels from epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs object containing multiple channel types.

    Returns
    -------
    mne.Epochs
        Copy of epochs containing only EEG channels.

    Notes
    -----
    - Other channel types (EOG, ECG, etc.) are removed.
    - Ensures consistency in feature extraction.
    """
    return epochs.copy().pick_types(eeg=True)


def get_time_windows(tmin: float, tmax: float, window_size: float, step: float):
    """
    Generate sliding window start times.

    Parameters
    ----------
    tmin : float
        Start time (in seconds).
    tmax : float
        End time (in seconds).
    window_size : float
        Length of each window (seconds).
    step : float
        Step size between windows (seconds).

    Returns
    -------
    np.ndarray
        Array of window start times.

    Notes
    -----
    - Ensures windows stay within [tmin, tmax].
    - Used for temporal feature extraction.
    """
    windows = []
    t = tmin
    while t + window_size <= tmax:
        windows.append(t)
        t += step
    return np.array(windows)


def compute_psd(epochs: mne.Epochs):
    """
    Compute Power Spectral Density (PSD) using Welch method.

    Parameters
    ----------
    epochs : mne.Epochs
        Input EEG epochs.

    Returns
    -------
    psds : np.ndarray
        PSD values of shape (n_epochs, n_channels, n_freqs).
    freqs : np.ndarray
        Frequency vector corresponding to PSD.

    Notes
    -----
    - Frequency range: 0.5–45 Hz.
    - Uses MNE's modern `compute_psd` API.
    """
    psd = epochs.compute_psd(
        method="welch", fmin=0.5, fmax=45, n_fft=1024, n_overlap=512, verbose=False
    )
    return psd.get_data(), psd.freqs


def extract_band_power(psds, freqs, bands):
    """
    Convert PSD into band power features.

    Parameters
    ----------
    psds : np.ndarray
        PSD array of shape (n_epochs, n_channels, n_freqs).
    freqs : np.ndarray
        Frequency vector.
    bands : dict
        Dictionary mapping band name → (fmin, fmax).

    Returns
    -------
    np.ndarray
        Band power array of shape (n_epochs, n_channels, n_bands).

    Notes
    -----
    - Power is computed as mean PSD within frequency band.
    - Band order follows dictionary insertion order.
    """
    band_powers = []

    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        power = psds[:, :, idx].mean(axis=2)
        band_powers.append(power)

    return np.stack(band_powers, axis=-1)


def compute_windowed_bandpower(
    epochs: mne.Epochs,
    tmin: float,
    tmax: float,
    window_size: float = 2.0,
    overlap: float = 0.5,
    bands=FREQ_BANDS,
):
    """
    Compute band power across sliding time windows.

    Parameters
    ----------
    epochs : mne.Epochs
        Input EEG epochs.
    tmin : float
        Start time of segment.
    tmax : float
        End time of segment.
    window_size : float
        Length of each window (seconds).
    overlap : float
        Overlap between consecutive windows (seconds).
    bands : dict
        Frequency bands definition.

    Returns
    -------
    np.ndarray
        Shape: (n_windows, n_epochs, n_channels, n_bands)

    Notes
    -----
    - Uses overlapping sliding windows.
    - Each window is processed independently.
    """
    step = window_size - overlap
    windows = get_time_windows(tmin, tmax, window_size, step)

    all_band_power = []

    for start in windows:
        logger.debug(f"Window: {start:.2f} -> {start + window_size:.2f}")

        ep = epochs.copy().crop(tmin=start, tmax=start + window_size)

        psds, freqs = compute_psd(ep)
        band_power = extract_band_power(psds, freqs, bands)

        all_band_power.append(band_power)

    return np.stack(all_band_power)


def log_baseline_normalize(power, baseline, eps=1e-10):
    """
    Apply log baseline normalization.

    Parameters
    ----------
    power : np.ndarray
        Power values (condition).
    baseline : np.ndarray
        Baseline power values.
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    np.ndarray
        Log-normalized power.

    Formula
    -------
    log((power + eps) / (baseline + eps))

    Notes
    -----
    - Stabilizes variance.
    - Common in EEG spectral analysis.
    """
    return np.log((power + eps) / (baseline + eps))


def flatten_features(data, ch_names, bands, prefix):
    """
    Flatten feature tensor into 2D matrix.

    Parameters
    ----------
    data : np.ndarray
        Shape: (n_epochs, n_channels, n_bands)
    ch_names : list
        Channel names.
    bands : dict
        Frequency bands.
    prefix : str
        Prefix for feature names (e.g., 'early', 'late').

    Returns
    -------
    flat : np.ndarray
        Shape: (n_epochs, n_features)
    feature_names : list
        Names of flattened features.

    Notes
    -----
    - Order: channel → band
    - Ensures correct mapping of features to names.
    """
    n_epochs, n_channels, n_bands = data.shape
    flat = data.reshape(n_epochs, -1)

    feature_names = []
    for ch in ch_names:
        for band in bands.keys():
            feature_names.append(f"{prefix}_{band}_{ch}")

    return flat, feature_names


def extract_bandpower_features(epochs: mne.Epochs, bands=FREQ_BANDS) -> pd.DataFrame:
    """
    Full band power feature extraction pipeline.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed EEG epochs.
    bands : dict
        Frequency band definitions.

    Returns
    -------
    pd.DataFrame
        Feature matrix with metadata.

    Notes
    -----
    Pipeline includes:
    - Baseline (-4 to 0 sec)
    - Early (0 to 5 sec)
    - Late (5 to 10 sec)
    - Log normalization
    """
    logger.info("Starting feature extraction")

    epochs = pick_eeg_channels(epochs)
    ch_names = epochs.ch_names

    baseline = compute_windowed_bandpower(epochs, -4, 0)
    baseline_mean = baseline.mean(axis=0)

    early = compute_windowed_bandpower(epochs, 0, 5)
    early_mean = early.mean(axis=0)

    late = compute_windowed_bandpower(epochs, 5, 10)
    late_mean = late.mean(axis=0)

    early_norm = log_baseline_normalize(early_mean, baseline_mean)
    late_norm = log_baseline_normalize(late_mean, baseline_mean)

    early_flat, early_names = flatten_features(early_norm, ch_names, bands, "early")
    late_flat, late_names = flatten_features(late_norm, ch_names, bands, "late")

    features = np.concatenate([early_flat, late_flat], axis=1)
    feature_names = early_names + late_names

    df = pd.DataFrame(features, columns=feature_names)

    if epochs.metadata is not None:
        for col in epochs.metadata.columns:
            df[col] = epochs.metadata[col].values

    logger.info("Feature extraction completed")

    return df

# ==============================
# IO FUNCTIONS
# ==============================


def save_features(df: pd.DataFrame, path: str):
    """
    Save extracted features to a parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame containing extracted EEG features.
    path : str
        Output file path.

    Returns
    -------
    None

    Notes
    -----
    - Uses Snappy compression for fast read/write.
    - Parquet format is efficient for ML pipelines.
    - Overwrites file if it already exists.
    """
    df.to_parquet(path, compression="snappy")
    logger.info(f"Saved features -> {path}")


def process_single_file(file_path: str, save_path: str = None):
    """
    Process a single EEG epochs file and extract features.

    Parameters
    ----------
    file_path : str
        Path to the input .fif epochs file.
    save_path : str, optional
        Path to save extracted features. If None, features are not saved.

    Returns
    -------
    pd.DataFrame
        Extracted feature DataFrame.

    Workflow
    --------
    1. Load epochs file
    2. Extract band power features
    3. Perform data quality checks (NaN, Inf)
    4. Optionally save features

    Notes
    -----
    - Intended for debugging and testing pipeline on a single subject.
    - Logs intermediate information for inspection.
    """
    logger.info(f"Processing file: {file_path}")

    epochs = mne.read_epochs(file_path, preload=True)
    logger.info(f"Epoch shape: {epochs.get_data().shape}")

    df = extract_bandpower_features(epochs)

    logger.info(f"Feature shape: {df.shape}")

    # Quality check
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

    logger.info(f"NaN count: {nan_count}")
    logger.info(f"Inf count: {inf_count}")

    if save_path:
        save_features(df, save_path)

    return df


def process_all_files(input_dir: str, output_dir: str):
    """
    Batch process multiple EEG epochs files.

    Parameters
    ----------
    input_dir : str
        Directory containing input .fif files.
    output_dir : str
        Directory where feature files will be saved.

    Returns
    -------
    None

    Workflow
    --------
    1. Locate all subject files (pattern: sub-*_whole_epo.fif)
    2. Loop through each subject
    3. Extract features
    4. Save individual subject features
    5. Combine all subjects into a single dataset

    Output
    ------
    - Individual files:
        <subject>_features.parquet
    - Combined file:
        all_subjects_features.parquet

    Notes
    -----
    - Errors in individual subjects do not stop the pipeline.
    - Failed subjects are logged and skipped.
    - Adds "subject" column for identification.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "sub-*_whole_epo.fif")))
    logger.info(f"Found {len(files)} files")

    all_dfs = []

    for file_path in files:
        subject = os.path.basename(file_path).split("_")[0]

        logger.info(f"Processing subject: {subject}")

        try:
            epochs = mne.read_epochs(file_path, preload=True)

            df = extract_bandpower_features(epochs)
            df["subject"] = subject

            save_path = os.path.join(output_dir, f"{subject}_bpfeatures.parquet")
            save_features(df, save_path)

            all_dfs.append(df)

        except Exception as e:
            logger.error(f"Error in {subject}: {e}", exc_info=True)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)

        final_path = os.path.join(output_dir, "all_subjects_features.parquet")
        save_features(final_df, final_path)

        logger.info(f"Final dataset shape: {final_df.shape}")
        logger.info("All subjects processed successfully")


# ==============================
# ENTRY POINT
# ==============================


def main():
    """
    Main execution entry point.

    Modes
    -----
    DEBUG = True
        → Process a single file (for testing/debugging)

    DEBUG = False
        → Process all files in batch mode

    Configuration
    -------------
    INPUT_DIR : str
        Directory containing input epoch files
    OUTPUT_DIR : str
        Directory to store extracted features
    DEBUG_FILE : str
        Path to a single test file
    DEBUG_SAVE : str
        Output path for debug features

    Notes
    -----
    - Modify paths before running.
    - Designed for flexible switching between debug and batch modes.
    """
    DEBUG = False

    INPUT_DIR = str(EXTRCT_FETRS)
    OUTPUT_DIR = str(FETRS_DIR)

    DEBUG_FILE = os.path.join(INPUT_DIR, "sub-021_whole_epo.fif")
    DEBUG_SAVE = os.path.join(OUTPUT_DIR, "sub-021_bpfeatures.parquet")

    if DEBUG:
        process_single_file(DEBUG_FILE, DEBUG_SAVE)
    else:
        process_all_files(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
