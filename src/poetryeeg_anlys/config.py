from pathlib import Path
from typing import Dict, Tuple

# ==============================
# PATH
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Core project paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intrmd_data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Raw + intermediate data paths
RAW_DIR = RAW_DATA_DIR / "ds006648-download"
FILTER_DIR = INTERMEDIATE_DATA_DIR / "filtered"
ICA_DIR = INTERMEDIATE_DATA_DIR / "ica_signal"
EPOCH_DIR = INTERMEDIATE_DATA_DIR / "epochs"
BEHAV_DIR = RAW_DIR / "derivatives/Behavioural_Ratings"
PARTICIPANTS_FILE = RAW_DIR / "participants.tsv"

# Epoch subfolders
START_REST_EPO = EPOCH_DIR / "strt_rst"
END_REST_EPO = EPOCH_DIR / "end_rst"
STIM_EPO = EPOCH_DIR / "stim"

G_STIM = STIM_EPO / "g_stim"
B_STIM = STIM_EPO / "b_stim"
INDICES_STIM = STIM_EPO / "indices_stim"
WHOLE_STIM_EPO = STIM_EPO / "whole_stim"

# Labels
RAW_LABEL_DIR = INTERMEDIATE_DATA_DIR / "raw_label"
LABELED_DIR = INTERMEDIATE_DATA_DIR / "labeled"

# Feature paths
FETRS_DIR = DATA_DIR / "features"
EXTRCT_FETRS = INTERMEDIATE_DATA_DIR / "raw_label"

# Log paths
PREPRS_LOG_DIR = LOGS_DIR / "preprs"
FETRS_LOG_DIR = LOGS_DIR / "fetrs"

# Create all required directories
for d in [
    FILTER_DIR,
    ICA_DIR,
    START_REST_EPO,
    END_REST_EPO,
    STIM_EPO,
    WHOLE_STIM_EPO,
    G_STIM,
    B_STIM,
    INDICES_STIM,
    FETRS_DIR,
    EXTRCT_FETRS,
    PREPRS_LOG_DIR,
    FETRS_LOG_DIR,
    RESULTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================
# CONFIG
# ==============================

CONFIG = {
    "sampling_rate": 512,
    "bandpass": (0.5, 45),
    "notch": [50, 100],
    "epoch_tmin": -4.0,
    "epoch_tmax": 11.0,
    "baseline": (-4.0, 0.0),
    "reject_threshold": 200e-6,
    "reject_tmin": 0.0,
    "reject_tmax": 10.0,
    "ica_components": 20,
    "stim_onset": 65282,
    "stim_codes": [65281, 65282, 65283, 65284],
    "rest_start": (65285, 65286),
    "rest_end": (65287, 65288),
    "rest_window": 4.0,
}

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
