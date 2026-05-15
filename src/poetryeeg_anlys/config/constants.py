"""
config/constants.py
===================
Domain constants shared across the analysis pipeline.

These values encode the experimental protocol (BioSemi ActiveTwo, 210-trial
Poetry–EEG design) and signal-processing parameters agreed upon in the
clarification document.  Do **not** scatter these values across individual
modules — import from here instead.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Hardware / recording
# ---------------------------------------------------------------------------
SFREQ: int = 512                   # sampling rate (Hz)

# EXG channel roles (BioSemi ActiveTwo)
EARLOBE_CHANNELS: List[str] = ["EXG1", "EXG2"]
EOG_CHANNELS:    List[str] = ["EXG3", "EXG4", "EXG7", "EXG8"]
ECG_CHANNELS:    List[str] = ["EXG5", "EXG6"]
EXG_ALL:         List[str] = ["EXG1", "EXG2", "EXG3", "EXG4",
                               "EXG5", "EXG6", "EXG7", "EXG8"]

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
NOTCH_FREQS:   List[float] = [50.0, 100.0]
BANDPASS:      Tuple[float, float] = (0.5, 48.0)

# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------
ICA_N_COMPONENTS: int   = 64
ICA_METHOD:       str   = "infomax"
ICA_RANDOM_STATE: int   = 97

# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------
EPOCH_TMIN:       float = -4.0
EPOCH_TMAX:       float = 11.0
BASELINE:         Tuple[float, float] = (-4.0, 0.0)
REJECT_THRESHOLD: float = 200e-6      # peak-to-peak amplitude (V)
REJECT_TMIN:      float = 0.0
REJECT_TMAX:      float = 10.0
N_PRACTICE_TRIALS: int  = 2
N_EXPECTED_TRIALS: int  = 210

# ---------------------------------------------------------------------------
# Event / marker codes
# ---------------------------------------------------------------------------
STIM_ONSET:    int               = 65282
STIM_CODES:    List[int]         = [65281, 65282, 65283, 65284]
REST_START:    Tuple[int, int]   = (65285, 65286)
REST_END:      Tuple[int, int]   = (65287, 65288)
REST_WINDOW:   float             = 4.0    # seconds per rest segment

# ---------------------------------------------------------------------------
# Feature extraction – Welch PSD
# ---------------------------------------------------------------------------
WELCH_N_FFT:     int = 512    # 1-second window @ 512 Hz
WELCH_N_OVERLAP: int = 256    # 50 % overlap
EPS:             float = 1e-30  # guard against log(0)

# Temporal segments (seconds, relative to epoch onset)
SEGMENTS: Dict[str, Tuple[float, float]] = {
    "baseline": (-4.0,  0.0),
    "early":    ( 0.0,  5.0),
    "late":     ( 5.0, 10.0),
}

# Frequency bands
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta":  ( 0.0,  4.0),
    "theta":  ( 4.0,  8.0),
    "alpha":  ( 8.0, 12.0),
    "beta":   (12.0, 30.0),
    "gamma1": (30.0, 48.0),
    "gamma2": (52.0, 80.0),   # computed but excluded from paper figures
}

# Bands written to the output feature matrix (gamma2 excluded)
BANDS_FOR_FEATURES: List[str] = ["delta", "theta", "alpha", "beta", "gamma1"]

# ---------------------------------------------------------------------------
# ROI electrode clusters (from Addendum)
# ---------------------------------------------------------------------------
ROI_MAP: Dict[str, List[str]] = {
    "CL1_lf":  ["AF7", "AF3", "F1",  "F3",  "F5",  "F7"],
    "CL2_lft": ["FT7", "FC3", "C1",  "CP3", "TP7", "T7"],
    "CL3_lpo": ["P7",  "P5",  "P3",  "P1",  "PO3", "PO7"],
    "CL4_rf":  ["AF8", "AF4", "F2",  "F4",  "F6",  "F8"],
    "CL5_rft": ["FT8", "FC4", "C2",  "CP4", "TP8", "T8"],
    "CL6_rpo": ["P8",  "P6",  "P4",  "P2",  "PO4", "PO8"],
}

# ---------------------------------------------------------------------------
# Labeling – behavioral columns
# ---------------------------------------------------------------------------
BEHAV_COLUMNS: List[str] = [
    "PoemType", "Block", "AA", "Imagery", "Moved", "Originality", "Creativity"
]
CATEGORICAL_COLS: List[str] = ["PoemType", "Block"]
