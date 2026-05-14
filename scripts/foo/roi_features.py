"""
ROI Band Power Feature Extraction Script

Author  : Pramit Biswas
Version : v1.0

Description
-----------
Extract baseline-normalised band power features and average them into
6 ROI clusters, following the exact signal-processing specification
provided in the clarification document (Query 3 & Query 4).

Algorithm
---------
1.  Load preprocessed epochs and keep only EEG channels.
2.  Crop three fixed temporal segments per epoch:
        Baseline  : -4 to  0 s  (samples 1 : 512×4)
        Early     :  0 to +5 s  (samples 512×4+1 : 512×9)
        Late      : +5 to +10 s (samples 512×9+1 : 512×14)
3.  Compute PSD with Welch on each segment independently:
        fs        = 512 Hz
        window    = 512 samples  (1 s)
        overlap   = 256 samples  (50 %)
        nfft      = 512 points
        → 257 one-sided frequency bins, 1 Hz resolution
4.  Apply log-ratio baseline correction AT THE SPECTRAL LEVEL
    (before band-averaging):
        pow_EP_BL = 10·log10(pow_EP) − 10·log10(pow_BL)
        pow_LP_BL = 10·log10(pow_LP) − 10·log10(pow_BL)
5.  Band-average the normalised spectra across the six bands:
        Delta   :  0– 4 Hz
        Theta   :  4– 8 Hz
        Alpha   :  8–12 Hz
        Beta    : 12–30 Hz
        Gamma1  : 30–48 Hz   (used in paper figures)
        Gamma2  : 52–80 Hz   (computed but excluded from paper)
6.  Average band power across the 6 predefined ROI electrode clusters.
7.  Flatten, attach metadata, save as Snappy-compressed Parquet.

ROI Clusters (from Addendum)
-----------------------------
CL1  Left Frontal           : AF7, AF3, F1, F3, F5, F7
CL2  Left Fronto-Temporal   : FT7, FC3, C1, CP3, TP7, T7
CL3  Left Parieto-Occipital : P7, P5, P3, P1, PO3, PO7
CL4  Right Frontal          : AF8, AF4, F2, F4, F6, F8
CL5  Right Fronto-Temporal  : FT8, FC4, C2, CP4, TP8, T8
CL6  Right Parieto-Occipital: P8, P6, P4, P2, PO4, PO8

Important ordering (Query 4)
-----------------------------
  CORRECT  : normalise spectra → band-average → ROI-average
  INCORRECT: band-average → normalise           (common replication error)

Input
-----
  sub-*_whole_epo.fif  (preprocessed MNE Epochs files)

Output
------
  sub-*_roi_bpfeatures.parquet   per-subject ROI feature files
  all_subjects_roi_bpfeatures.parquet  combined dataset
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
LOG_DIR = (BASE_DIR / "../../logs/fetrs").resolve()

INPUT_DIR = DATA_DIR / "intrmd_data/raw_label"
OUTPUT_DIR = DATA_DIR / "features/roi_bp"
BP_DIR = DATA_DIR / "features/bp"  # per-channel bandpower (pre-ROI)

for d in [OUTPUT_DIR, BP_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOGGER
# =========================================================
def setup_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Initialise a logger that writes to both console and file."""
    logger = logging.getLogger("ROIBandPowerLogger")
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


logger = setup_logger(str(LOG_DIR / "roi_bp_features.log"))


# =========================================================
# CONSTANTS  (Query 3)
# =========================================================
SFREQ: int = 512  # sampling rate (Hz)

# Welch parameters (match SpectralPower.m)
WELCH_N_FFT: int = 512  # 1 second window
WELCH_N_OVERLAP: int = 256  # 50 % overlap

# Fixed temporal segments (samples relative to epoch start)
# Epoch spans -4 to +11 s, so sample 1 = -4 s in MATLAB indexing.
# MNE uses seconds, so we pass time values directly.
SEGMENTS: Dict[str, Tuple[float, float]] = {
    "baseline": (-4.0, 0.0),
    "early": (0.0, 5.0),
    "late": (5.0, 10.0),
}

# Six frequency bands (Query 3)
# Gamma2 is computed but deliberately excluded from the feature matrix
# to match the paper figures (notch at 50 Hz ± 2 Hz).
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma1": (30.0, 48.0),
    "gamma2": (52.0, 80.0),
}

# Bands included in the output feature matrix (gamma2 excluded)
BANDS_FOR_FEATURES: List[str] = ["delta", "theta", "alpha", "beta", "gamma1"]

# ROI cluster definitions (Addendum — electrode order from paper)
ROI_MAP: Dict[str, List[str]] = {
    "CL1_lf": ["AF7", "AF3", "F1", "F3", "F5", "F7"],
    "CL2_lft": ["FT7", "FC3", "C1", "CP3", "TP7", "T7"],
    "CL3_lpo": ["P7", "P5", "P3", "P1", "PO3", "PO7"],
    "CL4_rf": ["AF8", "AF4", "F2", "F4", "F6", "F8"],
    "CL5_rft": ["FT8", "FC4", "C2", "CP4", "TP8", "T8"],
    "CL6_rpo": ["P8", "P6", "P4", "P2", "PO4", "PO8"],
}

EPS: float = 1e-30  # guard against log(0)


# =========================================================
# STEP 1 – CHANNEL SELECTION
# =========================================================
def pick_eeg_channels(epochs: mne.Epochs) -> mne.Epochs:
    """
    Keep only scalp EEG channels (drop EXG / EOG / ECG).

    Parameters
    ----------
    epochs : mne.Epochs
        Raw epochs with all channel types.

    Returns
    -------
    mne.Epochs
        Copy restricted to EEG channels.
    """
    return epochs.copy().pick_types(eeg=True)


# =========================================================
# STEP 2 – PSD COMPUTATION  (Query 3)
# =========================================================
def compute_segment_psd(
    epochs: mne.Epochs,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a fixed segment and compute its one-sided Welch PSD.

    Welch parameters match SpectralPower.m:
        window   = 512 samples  (1 s at 512 Hz)
        overlap  = 256 samples  (50 %)
        nfft     = 512 points
        → frequency resolution = 1 Hz
        → 257 one-sided frequency bins

    Parameters
    ----------
    epochs : mne.Epochs
        Full-length epochs (should span at least tmin–tmax).
    tmin, tmax : float
        Segment boundaries in seconds.

    Returns
    -------
    psds : np.ndarray
        Shape (n_epochs, n_channels, n_freqs).
    freqs : np.ndarray
        Frequency axis, length 257.
    """
    seg = epochs.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

    psd_obj = seg.compute_psd(
        method="welch",
        fmin=0.0,
        fmax=SFREQ / 2,  # full spectrum up to Nyquist
        n_fft=WELCH_N_FFT,
        n_overlap=WELCH_N_OVERLAP,
        verbose=False,
    )

    # psds : (n_epochs, n_channels, n_freqs)
    psds = psd_obj.get_data()
    freqs = psd_obj.freqs

    return psds, freqs


# =========================================================
# STEP 3 – BASELINE NORMALISATION  (Query 4)
# =========================================================
def log_baseline_normalise(
    pow_cond: np.ndarray,
    pow_bl: np.ndarray,
) -> np.ndarray:
    """
    Apply log-ratio baseline correction at the spectral level.

    Formula (from SpectralPower.m)
    --------------------------------
        pow_norm = 10·log10(pow_cond) − 10·log10(pow_bl)

    This is equivalent to 10·log10(pow_cond / pow_bl) but is computed
    in two separate log10 calls to match the MATLAB implementation
    exactly and avoid numerical differences when one operand is zero.

    Parameters
    ----------
    pow_cond : np.ndarray
        Condition PSD — shape (n_epochs, n_channels, n_freqs).
    pow_bl : np.ndarray
        Baseline PSD  — shape (n_epochs, n_channels, n_freqs).

    Returns
    -------
    np.ndarray
        Normalised power in dB, same shape as inputs.

    Notes
    -----
    EPS is added before taking log to prevent log(0).  The value
    (1e-30) is small enough not to bias physiologically plausible
    power values.
    """
    return 10.0 * np.log10(pow_cond + EPS) - 10.0 * np.log10(pow_bl + EPS)


# =========================================================
# STEP 4 – BAND AVERAGING  (Query 4 — after normalisation)
# =========================================================
def band_average(
    psds_norm: np.ndarray,
    freqs: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Average normalised PSD within each frequency band.

    Band-averaging is performed AFTER baseline normalisation,
    matching the ordering in SpectralPower.m.  Reversing these two
    steps produces different results (common replication error noted
    in Query 4).

    Parameters
    ----------
    psds_norm : np.ndarray
        Baseline-normalised PSD — shape (n_epochs, n_channels, n_freqs).
    freqs : np.ndarray
        Frequency axis.
    bands : dict
        Band name → (fmin, fmax) in Hz.

    Returns
    -------
    dict
        Maps band name → array of shape (n_epochs, n_channels).
    """
    band_powers: Dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)

        if not idx.any():
            logger.warning(
                f"No frequency bins found for band '{band_name}' "
                f"({fmin}–{fmax} Hz). Skipping."
            )
            continue

        # mean across frequency bins → (n_epochs, n_channels)
        band_powers[band_name] = psds_norm[:, :, idx].mean(axis=2)

    return band_powers


# =========================================================
# STEP 5 – ROI AVERAGING
# =========================================================
def roi_average(
    band_powers: Dict[str, np.ndarray],
    ch_names: List[str],
    roi_map: Dict[str, List[str]],
    phase: str,
    bands_for_features: List[str],
) -> Dict[str, np.ndarray]:
    """
    Average channel-wise band power across ROI electrode clusters.

    ROI averaging is the final spatial reduction step, applied after
    band-averaging of the baseline-normalised spectra.

    Parameters
    ----------
    band_powers : dict
        Band name → array (n_epochs, n_channels).
    ch_names : list
        Channel names corresponding to axis-1 of each array.
    roi_map : dict
        ROI name → list of electrode names in the cluster.
    phase : str
        'early' or 'late' — used to build column names.
    bands_for_features : list
        Bands to include in output (gamma2 excluded from paper figures).

    Returns
    -------
    dict
        Column name → array (n_epochs,) ready for a DataFrame.
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
            missing = [e for e in electrodes if e not in ch_index]

            if missing:
                logger.warning(
                    f"[{phase}|{band_name}|{roi_name}] "
                    f"Missing electrodes: {missing}. "
                    f"Averaging over {len(available)}/{len(electrodes)} channels."
                )

            if not available:
                logger.error(
                    f"[{phase}|{band_name}|{roi_name}] "
                    "No electrodes found — ROI omitted from output."
                )
                continue

            indices = [ch_index[e] for e in available]
            col = f"{phase}_{band_name}_{roi_name}"
            roi_features[col] = bp[:, indices].mean(axis=1)  # (n_epochs,)

    return roi_features


# =========================================================
# PER-CHANNEL FEATURE FLATTENING
# =========================================================
def flatten_channel_features(
    early_bands: Dict[str, np.ndarray],
    late_bands: Dict[str, np.ndarray],
    ch_names: List[str],
    bands_for_features: List[str],
) -> pd.DataFrame:
    """
    Flatten per-channel band power into a 2-D feature DataFrame.

    Called after band-averaging but BEFORE ROI averaging, so each
    column represents one phase × band × channel combination.

    Column naming: {phase}_{band}_{channel}
    Example      : 'early_alpha_AF7'

    Parameters
    ----------
    early_bands, late_bands : dict
        Band name → array (n_epochs, n_channels) for each phase.
    ch_names : list
        Channel names (length = n_channels).
    bands_for_features : list
        Bands to include (gamma2 excluded to match paper figures).

    Returns
    -------
    pd.DataFrame
        Shape (n_epochs, n_phases × n_bands × n_channels).
    """
    cols: Dict[str, np.ndarray] = {}

    for phase, band_powers in [("early", early_bands), ("late", late_bands)]:
        for band_name in bands_for_features:
            if band_name not in band_powers:
                logger.warning(
                    f"[flatten_channel] Band '{band_name}' missing — skipped."
                )
                continue

            bp = band_powers[band_name]  # (n_epochs, n_channels)

            for ch_idx, ch in enumerate(ch_names):
                cols[f"{phase}_{band_name}_{ch}"] = bp[:, ch_idx]

    return pd.DataFrame(cols)


# =========================================================
# MAIN EXTRACTION PIPELINE
# =========================================================
def extract_roi_bandpower_features(
    epochs: mne.Epochs,
    bands: Dict[str, Tuple[float, float]] = FREQ_BANDS,
    bands_for_features: List[str] = BANDS_FOR_FEATURES,
    roi_map: Dict[str, List[str]] = ROI_MAP,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full ROI band power feature extraction pipeline.

    Processing order (critical — see Query 4)
    ------------------------------------------
    1.  Compute Welch PSD for each fixed temporal segment.
    2.  Apply log-ratio baseline normalisation at the spectral level.
    3.  Average normalised spectra within each frequency band.
    4.  Flatten per-channel band power (saved to bp/ before ROI step).
    5.  Average channel-wise band power within each ROI cluster.
    6.  Attach metadata to both DataFrames and return them.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed EEG epochs spanning at least -4 to +10 s.
    bands : dict
        Frequency band definitions (all six, including gamma2).
    bands_for_features : list
        Subset of bands written to output DataFrames.
    roi_map : dict
        ROI cluster → electrode list mapping.

    Returns
    -------
    bp_df : pd.DataFrame
        Per-channel band power.  Columns = {phase}_{band}_{ch} + metadata.
        Example column: 'early_alpha_AF7'
    roi_df : pd.DataFrame
        ROI-averaged band power.  Columns = {phase}_{band}_{roi} + metadata.
        Example column: 'early_alpha_CL1_lf'
    """
    logger.info("Starting ROI band power feature extraction")

    # ── Channel selection ──────────────────────────────────────────────
    epochs = pick_eeg_channels(epochs)
    ch_names = epochs.ch_names
    logger.info(f"EEG channels selected: {len(ch_names)}")

    # ── Step 2: PSD per segment  (Query 3) ────────────────────────────
    logger.info("Computing Welch PSD for each temporal segment ...")

    pow_bl, freqs = compute_segment_psd(epochs, *SEGMENTS["baseline"])
    pow_early, _ = compute_segment_psd(epochs, *SEGMENTS["early"])
    pow_late, _ = compute_segment_psd(epochs, *SEGMENTS["late"])

    logger.info(
        f"PSD shape per segment: {pow_bl.shape}  "
        f"(n_epochs × n_channels × {len(freqs)} freq bins)"
    )

    # ── Step 3: Log-ratio baseline normalisation  (Query 4) ────────────
    logger.info(
        "Applying log-ratio baseline correction at spectral level "
        "(normalise BEFORE band-averaging) ..."
    )
    pow_early_norm = log_baseline_normalise(pow_early, pow_bl)
    pow_late_norm = log_baseline_normalise(pow_late, pow_bl)

    # ── Step 4: Band-averaging AFTER normalisation  (Query 4) ──────────
    logger.info("Band-averaging normalised spectra ...")
    early_bands = band_average(pow_early_norm, freqs, bands)
    late_bands = band_average(pow_late_norm, freqs, bands)

    # ── Step 4b: Per-channel DataFrame (saved to bp/ before ROI step) ──
    bp_df = flatten_channel_features(
        early_bands, late_bands, ch_names, bands_for_features
    )
    logger.info(f"Per-channel feature matrix: {bp_df.shape}")

    # ── Step 5: ROI spatial averaging ──────────────────────────────────
    logger.info("Averaging band power within ROI clusters ...")
    early_roi = roi_average(early_bands, ch_names, roi_map, "early", bands_for_features)
    late_roi = roi_average(late_bands, ch_names, roi_map, "late", bands_for_features)

    roi_df = pd.DataFrame({**early_roi, **late_roi})

    n_expected_roi_cols = len(bands_for_features) * len(roi_map) * 2
    logger.info(
        f"ROI feature matrix: {roi_df.shape[0]} epochs × {roi_df.shape[1]} features "
        f"(expected {n_expected_roi_cols} feature columns)"
    )

    # ── Attach epoch metadata to both DataFrames ───────────────────────
    if epochs.metadata is not None:
        for col in epochs.metadata.columns:
            bp_df[col] = epochs.metadata[col].values
            roi_df[col] = epochs.metadata[col].values
        logger.info(f"Metadata columns attached: {list(epochs.metadata.columns)}")

    # ── Quality check ──────────────────────────────────────────────────
    for label, df in [("bp", bp_df), ("roi", roi_df)]:
        nan_count = df.isna().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if nan_count > 0:
            logger.warning(f"[{label}] NaN values detected: {nan_count}")
        if inf_count > 0:
            logger.warning(f"[{label}] Inf values detected: {inf_count}")

    logger.info("ROI band power feature extraction completed.")
    return bp_df, roi_df


# =========================================================
# IO HELPERS
# =========================================================
def save_features(df: pd.DataFrame, path: str) -> None:
    """
    Save feature DataFrame to Snappy-compressed Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix.
    path : str
        Output file path.
    """
    df.to_parquet(path, compression="snappy")
    logger.info(f"Saved features → {path}")


def already_processed(subject: str, output_dir: str) -> bool:
    """
    Check whether a subject's ROI feature file already exists.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g. 'sub-021').
    output_dir : str
        ROI output directory to check.

    Returns
    -------
    bool
    """
    output_file = os.path.join(output_dir, f"{subject}_roi_bpfeatures.parquet")
    return os.path.exists(output_file)


# =========================================================
# PROCESSING FUNCTIONS
# =========================================================
def process_single_file(
    file_path: str,
    roi_save_path: str | None = None,
    bp_save_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract per-channel and ROI band power features from a single subject file.

    Parameters
    ----------
    file_path : str
        Path to a preprocessed .fif epochs file.
    roi_save_path : str, optional
        Where to save the ROI feature Parquet.  If None, not saved.
    bp_save_path : str, optional
        Where to save the per-channel feature Parquet.  If None, not saved.

    Returns
    -------
    bp_df : pd.DataFrame
        Per-channel band power feature matrix.
    roi_df : pd.DataFrame
        ROI-averaged band power feature matrix.
    """
    logger.info(f"Processing single file: {file_path}")

    epochs = mne.read_epochs(file_path, preload=True)
    logger.info(f"Epoch shape: {epochs.get_data().shape}")

    bp_df, roi_df = extract_roi_bandpower_features(epochs)

    if bp_save_path:
        save_features(bp_df, bp_save_path)

    if roi_save_path:
        save_features(roi_df, roi_save_path)

    return bp_df, roi_df


def process_all_files(input_dir: str, output_dir: str, bp_dir: str) -> None:
    """
    Batch-process all subject epoch files and save both feature sets.

    Parameters
    ----------
    input_dir : str
        Directory containing sub-*_whole_epo.fif files.
    output_dir : str
        Directory where ROI feature Parquet files are saved.
    bp_dir : str
        Directory where per-channel band power Parquet files are saved.

    Output
    ------
    Per-channel (bp/)  : {bp_dir}/sub-*_bpfeatures.parquet
    ROI (roi_bp/)      : {output_dir}/sub-*_roi_bpfeatures.parquet
    Combined ROI       : {output_dir}/all_subjects_roi_bpfeatures.parquet

    Notes
    -----
    - Resume-safe: subjects whose ROI file already exists are skipped.
    - Errors in individual subjects are logged and do not stop the batch.
    - The combined file covers only subjects processed in this run.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bp_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "sub-*_whole_epo.fif")))
    logger.info(f"Found {len(files)} epoch files in {input_dir}")

    all_roi_dfs: List[pd.DataFrame] = []

    for file_path in files:
        subject = os.path.basename(file_path).split("_")[0]

        if already_processed(subject, output_dir):
            logger.info(f"Skipping {subject} (already processed)")
            continue

        logger.info(f"Processing subject: {subject}")

        try:
            epochs = mne.read_epochs(file_path, preload=True)

            bp_df, roi_df = extract_roi_bandpower_features(epochs)
            bp_df["subject"] = subject
            roi_df["subject"] = subject

            # ── Save per-channel features to bp/ ──────────────────────
            bp_save_path = os.path.join(bp_dir, f"{subject}_bpfeatures.parquet")
            save_features(bp_df, bp_save_path)

            # ── Save ROI features to roi_bp/ ──────────────────────────
            roi_save_path = os.path.join(
                output_dir, f"{subject}_roi_bpfeatures.parquet"
            )
            save_features(roi_df, roi_save_path)

            all_roi_dfs.append(roi_df)

        except Exception as exc:
            logger.error(f"Error processing {subject}: {exc}", exc_info=True)

    # ── Combine ROI DataFrames across newly processed subjects ─────────
    if all_roi_dfs:
        final_df = pd.concat(all_roi_dfs, ignore_index=True)
        final_path = os.path.join(output_dir, "all_subjects_roi_bpfeatures.parquet")
        save_features(final_df, final_path)
        logger.info(f"Combined ROI dataset shape: {final_df.shape}")
        logger.info("Batch processing complete.")
    else:
        logger.info("No new subjects to process.")


# =========================================================
# ENTRY POINT
# =========================================================
def main() -> None:
    """
    Entry point.

    Set DEBUG = True to run a single subject (useful for validation).
    Set DEBUG = False for full batch processing.
    """
    DEBUG = True

    INPUT_DIR_STR = str(INPUT_DIR)
    OUTPUT_DIR_STR = str(OUTPUT_DIR)
    BP_DIR_STR = str(BP_DIR)

    DEBUG_FILE = os.path.join(INPUT_DIR_STR, "sub-021_whole_epo.fif")
    DEBUG_ROI_SAVE = os.path.join(OUTPUT_DIR_STR, "sub-021_roi_bpfeatures.parquet")
    DEBUG_BP_SAVE = os.path.join(BP_DIR_STR, "sub-021_bpfeatures.parquet")

    if DEBUG:
        process_single_file(DEBUG_FILE, DEBUG_ROI_SAVE, DEBUG_BP_SAVE)
    else:
        process_all_files(INPUT_DIR_STR, OUTPUT_DIR_STR, BP_DIR_STR)


if __name__ == "__main__":
    main()
