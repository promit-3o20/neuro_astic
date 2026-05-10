"""
Enhanced Band Power Features Extraction with ROI Clustering

Extracts:
1. Bandpower features per electrode
2. Ratio features per electrode  
3. Hemispheric asymmetry features
4. ROI-aggregated features
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import mne
from typing import Dict, Tuple, List
from pathlib import Path

# ==============================
# CONFIG
# ==============================

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 48),
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

EPS = 1e-30

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
LOG_DIR = (BASE_DIR / "../../logs/fetrs").resolve()

FETRS_DIR = DATA_DIR / "features/roi_ftrs2"
EXTRCT_FETRS = DATA_DIR / "intrmd_data/raw_label"

# create directories
for d in [FETRS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================
# LOGGER SETUP
# ==============================

def setup_logger(log_file: str, level=logging.INFO):
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

logger = setup_logger(str(LOG_DIR / "bandpower_roi.log"))

# ==============================
# ROI CLUSTERING FUNCTIONS
# ==============================

def aggregate_roi_features(features: np.ndarray, 
                          ch_names: List[str], 
                          roi_map: Dict[str, List[str]],
                          agg_func='mean') -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate electrode-level features into ROI clusters.
    
    Parameters
    ----------
    features : np.ndarray
        Shape (n_epochs, n_channels, n_features) or (n_epochs, n_channels)
    ch_names : List[str]
        Channel names corresponding to feature dimensions
    roi_map : Dict[str, List[str]]
        Mapping from ROI name to list of channel names
    agg_func : str
        Aggregation function ('mean', 'std', 'max', 'min')
    
    Returns
    -------
    roi_features : np.ndarray
        Shape (n_epochs, n_rois, n_features)
    roi_names : List[str]
        Names of ROIs in order
    """
    n_epochs = features.shape[0]
    n_features = features.shape[2] if features.ndim == 3 else 1
    
    if features.ndim == 2:
        features = features[:, :, np.newaxis]
        n_features = 1
    
    # Create channel to index mapping
    ch_to_idx = {ch: i for i, ch in enumerate(ch_names)}
    
    roi_features = []
    roi_names = list(roi_map.keys())
    
    for roi_name, roi_chs in roi_map.items():
        # Get indices for this ROI
        roi_indices = [ch_to_idx[ch] for ch in roi_chs if ch in ch_to_idx]
        
        if not roi_indices:
            logger.warning(f"No channels found for ROI: {roi_name}")
            roi_features.append(np.zeros((n_epochs, n_features)))
            continue
        
        # Extract and aggregate
        roi_data = features[:, roi_indices, :]
        
        if agg_func == 'mean':
            roi_agg = roi_data.mean(axis=1)
        elif agg_func == 'std':
            roi_agg = roi_data.std(axis=1)
        elif agg_func == 'max':
            roi_agg = roi_data.max(axis=1)
        elif agg_func == 'min':
            roi_agg = roi_data.min(axis=1)
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
        
        roi_features.append(roi_agg)
    
    # Stack: (n_rois, n_epochs, n_features) -> (n_epochs, n_rois, n_features)
    roi_features = np.stack(roi_features, axis=1)
    
    return roi_features, roi_names

def compute_roi_asymmetry(roi_features_left: np.ndarray,
                          roi_features_right: np.ndarray,
                          roi_pairs: List[Tuple[str, str]],
                          feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute asymmetry between paired ROIs (left vs right hemisphere).
    
    Parameters
    ----------
    roi_features_left : np.ndarray
        Features for left ROIs, shape (n_epochs, n_left_rois, n_features)
    roi_features_right : np.ndarray
        Features for right ROIs, shape (n_epochs, n_right_rois, n_features)
    roi_pairs : List[Tuple[str, str]]
        List of (left_roi, right_roi) pairs
    feature_names : List[str]
        Names of features (e.g., band names)
    
    Returns
    -------
    asym_features : np.ndarray
        Asymmetry features, shape (n_epochs, n_pairs * n_features)
    asym_names : List[str]
        Names for asymmetry features
    """
    asym_list = []
    asym_names_list = []
    
    for left_roi, right_roi in roi_pairs:
        # Get indices for this pair
        left_idx = list(roi_pairs).index((left_roi, right_roi)) if left_roi in [p[0] for p in roi_pairs] else None
        right_idx = list(roi_pairs).index((left_roi, right_roi)) if right_roi in [p[1] for p in roi_pairs] else None
        
        # Simplified: assuming features are in same order as roi_pairs
        # For full implementation, you'd need proper indexing based on ROI names
        
        left_data = roi_features_left[:, left_idx, :] if left_idx is not None else None
        right_data = roi_features_right[:, right_idx, :] if right_idx is not None else None
        
        if left_data is not None and right_data is not None:
            # Calculate asymmetry: (L - R) / (L + R)
            asym = (left_data - right_data) / (left_data + right_data + EPS)
            asym_list.append(asym)
            
            # Create names for each feature
            for feat_name in feature_names:
                asym_names_list.append(f"asym_{feat_name}_{left_roi}_{right_roi}")
    
    if not asym_list:
        return np.empty((roi_features_left.shape[0], 0)), []
    
    return np.concatenate(asym_list, axis=1), asym_names_list

# ==============================
# CORE FUNCTIONS (from original)
# ==============================

def pick_eeg_channels(epochs: mne.Epochs) -> mne.Epochs:
    """Select only EEG channels from epochs."""
    return epochs.copy().pick_types(eeg=True)

def get_time_windows(tmin: float, tmax: float, window_size: float, step: float):
    """Generate sliding window start times."""
    windows = []
    t = tmin
    while t + window_size <= tmax:
        windows.append(t)
        t += step
    return np.array(windows)

def compute_psd(epochs: mne.Epochs):
    """Compute Power Spectral Density using Welch method."""
    psd = epochs.compute_psd(
        method="welch", fmin=0.5, fmax=45, n_fft=1024, n_overlap=512, verbose=False
    )
    return psd.get_data(), psd.freqs

def extract_band_power(psds, freqs, bands):
    """Convert PSD into band power features."""
    band_powers = []
    
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        power = psds[:, :, idx].mean(axis=2)
        band_powers.append(power)
    
    return np.stack(band_powers, axis=-1)

def compute_relative_bandpower(abs_power):
    """Compute relative bandpower."""
    total_power = abs_power.sum(axis=-1, keepdims=True) + EPS
    return abs_power / total_power

def compute_band_ratios(abs_power, bands=FREQ_BANDS, ratio_pairs=RATIO_PAIRS):
    """Compute bandpower ratios."""
    band_names = list(bands.keys())
    band_idx = {b: i for i, b in enumerate(band_names)}
    
    ratio_features = []
    ratio_names = []
    
    for ratio_name, (num_band, den_band) in ratio_pairs.items():
        num = abs_power[:, :, band_idx[num_band]]
        den = abs_power[:, :, band_idx[den_band]] + EPS
        ratio = num / den
        
        ratio_features.append(ratio[..., np.newaxis])
        ratio_names.append(ratio_name)
    
    return np.concatenate(ratio_features, axis=-1), ratio_names

def compute_windowed_bandpower(
    epochs: mne.Epochs,
    tmin: float,
    tmax: float,
    window_size: float = 2.0,
    overlap: float = 0.5,
    bands=FREQ_BANDS,
):
    """Compute band power across sliding time windows."""
    step = window_size - overlap
    windows = get_time_windows(tmin, tmax, window_size, step)
    
    all_abs = []
    all_rel = []
    all_ratio = []
    
    for start in windows:
        ep = epochs.copy().crop(tmin=start, tmax=start + window_size)
        psds, freqs = compute_psd(ep)
        
        abs_power = extract_band_power(psds, freqs, bands)
        rel_power = compute_relative_bandpower(abs_power)
        ratio_power, ratio_names = compute_band_ratios(abs_power)
        
        all_abs.append(abs_power)
        all_rel.append(rel_power)
        all_ratio.append(ratio_power)
    
    return (
        np.stack(all_abs),
        np.stack(all_rel),
        np.stack(all_ratio),
        epochs.ch_names,
        ratio_names,
    )

def extract_bandpower_features_with_roi(epochs: mne.Epochs, 
                                        bands=FREQ_BANDS,
                                        roi_map=ROI_MAP,
                                        roi_pairs=ROI_PAIRS) -> pd.DataFrame:
    """
    Full band power feature extraction pipeline with ROI clustering.
    
    Extracts:
    1. Electrode-level features (absolute, relative, ratio)
    2. ROI-aggregated features
    3. ROI asymmetry features
    """
    logger.info("Starting ROI-based feature extraction")
    
    epochs = pick_eeg_channels(epochs)
    ch_names = epochs.ch_names
    
    # Compute windowed features
    base_abs, base_rel, base_ratio, ch_names, ratio_names = compute_windowed_bandpower(epochs, -4, 0)
    early_abs, early_rel, early_ratio, _, _ = compute_windowed_bandpower(epochs, 0, 5)
    late_abs, late_rel, late_ratio, _, _ = compute_windowed_bandpower(epochs, 5, 10)
    
    # Average across windows
    base_abs = base_abs.mean(axis=0)
    early_abs = early_abs.mean(axis=0)
    late_abs = late_abs.mean(axis=0)
    
    base_rel = base_rel.mean(axis=0)
    early_rel = early_rel.mean(axis=0)
    late_rel = late_rel.mean(axis=0)
    
    base_ratio = base_ratio.mean(axis=0)
    early_ratio = early_ratio.mean(axis=0)
    late_ratio = late_ratio.mean(axis=0)
    
    # Apply log baseline normalization
    early_abs = np.log10((early_abs + EPS) / (base_abs + EPS))
    late_abs = np.log10((late_abs + EPS) / (base_abs + EPS))
    
    early_rel = np.log10((early_rel + EPS) / (base_rel + EPS))
    late_rel = np.log10((late_rel + EPS) / (base_rel + EPS))
    
    early_ratio = np.log10((early_ratio + EPS) / (base_ratio + EPS))
    late_ratio = np.log10((late_ratio + EPS) / (base_ratio + EPS))
    
    # ============================================
    # ROI AGGREGATION
    # ============================================
    
    # Get band names
    band_names = list(bands.keys())
    
    # Aggregate absolute power to ROIs
    early_abs_roi, roi_names = aggregate_roi_features(early_abs, ch_names, roi_map, 'mean')
    late_abs_roi, _ = aggregate_roi_features(late_abs, ch_names, roi_map, 'mean')
    
    # Aggregate relative power to ROIs
    early_rel_roi, _ = aggregate_roi_features(early_rel, ch_names, roi_map, 'mean')
    late_rel_roi, _ = aggregate_roi_features(late_rel, ch_names, roi_map, 'mean')
    
    # Aggregate ratio features to ROIs
    early_ratio_roi, _ = aggregate_roi_features(early_ratio, ch_names, roi_map, 'mean')
    late_ratio_roi, _ = aggregate_roi_features(late_ratio, ch_names, roi_map, 'mean')
    
    # ============================================
    # ROI ASYMMETRY
    # ============================================
    
    # Split ROIs into left and right
    left_rois = [roi for roi in roi_names if 'lf' in roi or 'lft' in roi or 'lpo' in roi]
    right_rois = [roi for roi in roi_names if 'rf' in roi or 'rft' in roi or 'rpo' in roi]
    
    # Get indices for left and right ROIs
    left_indices = [i for i, roi in enumerate(roi_names) if roi in left_rois]
    right_indices = [i for i, roi in enumerate(roi_names) if roi in right_rois]
    
    # Compute asymmetry for absolute power
    early_abs_left = early_abs_roi[:, left_indices, :]
    early_abs_right = early_abs_roi[:, right_indices, :]
    late_abs_left = late_abs_roi[:, left_indices, :]
    late_abs_right = late_abs_roi[:, right_indices, :]
    
    early_abs_asym = (early_abs_left - early_abs_right) / (early_abs_left + early_abs_right + EPS)
    late_abs_asym = (late_abs_left - late_abs_right) / (late_abs_left + late_abs_right + EPS)
    
    # ============================================
    # FEATURE FLATTENING
    # ============================================
    
    # Flatten electrode-level features (optional - uncomment if needed)
    # early_abs_flat = early_abs.reshape(early_abs.shape[0], -1)
    
    # Flatten ROI features
    early_abs_roi_flat = early_abs_roi.reshape(early_abs_roi.shape[0], -1)
    late_abs_roi_flat = late_abs_roi.reshape(late_abs_roi.shape[0], -1)
    early_rel_roi_flat = early_rel_roi.reshape(early_rel_roi.shape[0], -1)
    late_rel_roi_flat = late_rel_roi.reshape(late_rel_roi.shape[0], -1)
    early_ratio_roi_flat = early_ratio_roi.reshape(early_ratio_roi.shape[0], -1)
    late_ratio_roi_flat = late_ratio_roi.reshape(late_ratio_roi.shape[0], -1)
    
    # Flatten asymmetry features
    early_abs_asym_flat = early_abs_asym.reshape(early_abs_asym.shape[0], -1)
    late_abs_asym_flat = late_abs_asym.reshape(late_abs_asym.shape[0], -1)
    
    # ============================================
    # FEATURE NAMES
    # ============================================
    
    # ROI feature names
    abs_roi_early_names = [f"early_abs_{band}_{roi}" 
                          for roi in roi_names for band in band_names]
    abs_roi_late_names = [f"late_abs_{band}_{roi}" 
                         for roi in roi_names for band in band_names]
    
    rel_roi_early_names = [f"early_rel_{band}_{roi}" 
                          for roi in roi_names for band in band_names]
    rel_roi_late_names = [f"late_rel_{band}_{roi}" 
                         for roi in roi_names for band in band_names]
    
    ratio_roi_early_names = [f"early_ratio_{ratio}_{roi}" 
                            for roi in roi_names for ratio in ratio_names]
    ratio_roi_late_names = [f"late_ratio_{ratio}_{roi}" 
                           for roi in roi_names for ratio in ratio_names]
    
    # Asymmetry feature names
    abs_asym_early_names = [f"early_asym_abs_{band}_{left}_{right}" 
                           for left, right in zip(left_rois, right_rois) 
                           for band in band_names]
    abs_asym_late_names = [f"late_asym_abs_{band}_{left}_{right}" 
                          for left, right in zip(left_rois, right_rois) 
                          for band in band_names]
    
    # ============================================
    # CONCATENATE FEATURES
    # ============================================
    
    features = np.concatenate([
        early_abs_roi_flat,
        late_abs_roi_flat,
        early_rel_roi_flat,
        late_rel_roi_flat,
        early_ratio_roi_flat,
        late_ratio_roi_flat,
        early_abs_asym_flat,
        late_abs_asym_flat,
    ], axis=1)
    
    feature_names = (
        abs_roi_early_names + abs_roi_late_names +
        rel_roi_early_names + rel_roi_late_names +
        ratio_roi_early_names + ratio_roi_late_names +
        abs_asym_early_names + abs_asym_late_names
    )
    
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add metadata
    if epochs.metadata is not None:
        for col in epochs.metadata.columns:
            df[col] = epochs.metadata[col].values
    
    logger.info(f"Feature extraction completed with {len(feature_names)} features")
    logger.info(f"Feature breakdown: {len(abs_roi_early_names)} ROI-abs, "
                f"{len(rel_roi_early_names)} ROI-rel, "
                f"{len(ratio_roi_early_names)} ROI-ratio, "
                f"{len(abs_asym_early_names)} asymmetry")
    
    return df

# ==============================
# IO FUNCTIONS
# ==============================

def save_features(df: pd.DataFrame, path: str):
    """Save extracted features to a parquet file."""
    df.to_parquet(path, compression="snappy")
    logger.info(f"Saved features -> {path}")

def process_single_file(file_path: str, save_path: str = None):
    """Process a single EEG epochs file and extract ROI-clustered features."""
    logger.info(f"Processing file: {file_path}")
    
    epochs = mne.read_epochs(file_path, preload=True)
    logger.info(f"Epoch shape: {epochs.get_data().shape}")
    
    df = extract_bandpower_features_with_roi(epochs)
    
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
    """Batch process multiple EEG epochs files."""
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(input_dir, "sub-*_whole_epo.fif")))
    logger.info(f"Found {len(files)} files")
    
    all_dfs = []
    
    for file_path in files:
        subject = os.path.basename(file_path).split("_")[0]
        logger.info(f"Processing subject: {subject}")
        
        try:
            epochs = mne.read_epochs(file_path, preload=True)
            df = extract_bandpower_features_with_roi(epochs)
            df["subject"] = subject
            
            save_path = os.path.join(output_dir, f"{subject}_roi_bpfeatures.parquet")
            save_features(df, save_path)
            all_dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error in {subject}: {e}", exc_info=True)
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_path = os.path.join(output_dir, "all_subjects_features_roi.parquet")
        save_features(final_df, final_path)
        logger.info(f"Final dataset shape: {final_df.shape}")

# ==============================
# ENTRY POINT
# ==============================

def main():
    """Main execution entry point."""
    DEBUG = False
    
    INPUT_DIR = str(EXTRCT_FETRS)
    OUTPUT_DIR = str(FETRS_DIR)
    
    DEBUG_FILE = os.path.join(INPUT_DIR, "sub-021_whole_epo.fif")
    DEBUG_SAVE = os.path.join(OUTPUT_DIR, "sub-021_bpfeatures_roi.parquet")
    
    if DEBUG:
        process_single_file(DEBUG_FILE, DEBUG_SAVE)
    else:
        process_all_files(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()