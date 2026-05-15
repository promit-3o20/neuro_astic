"""
config/paths.py
===============
Centralised, environment-aware path definitions.

All paths are derived from the DATA_ROOT environment variable (default: two
levels above the package root).  Override DATA_ROOT to redirect the entire
pipeline to a different storage location without touching any other file.

Usage
-----
    from poetryeeg_anlys.config.paths import Paths
    raw_dir = Paths.RAW_DIR
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parents[2]          # src/poetryeeg_anlys
_REPO_ROOT = _PKG_ROOT.parent.parent                     # repository root

# Allow override via environment variable
DATA_ROOT = Path(os.environ.get("EEG_DATA_ROOT", _REPO_ROOT / "data"))
LOG_ROOT  = Path(os.environ.get("EEG_LOG_ROOT",  _REPO_ROOT / "logs"))
RESULTS_DIR = Path(os.environ.get("EEG_RESULTS_ROOT", _REPO_ROOT / "results"))

class Paths:
    """Namespace for all project paths (all are pathlib.Path objects)."""

    # ── Raw data ──────────────────────────────────────────────────────────
    RAW_DIR         = DATA_ROOT / "raw_data" / "ds006648-download"
    PARTICIPANTS    = RAW_DIR / "participants.tsv"
    BEHAV_DIR       = RAW_DIR / "derivatives" / "Behavioural_Ratings"

    # ── Intermediate outputs ──────────────────────────────────────────────
    INTRMD_DIR      = DATA_ROOT / "intrmd_data"
    FILTER_DIR      = INTRMD_DIR / "filtered"
    ICA_DIR         = INTRMD_DIR / "ica_signal"
    EPOCH_DIR       = INTRMD_DIR / "epochs"

    # Epoch sub-directories
    STIM_EPO        = EPOCH_DIR / "stim"
    WHOLE_STIM_EPO  = STIM_EPO / "whole_stim"
    G_STIM          = STIM_EPO / "g_stim"
    B_STIM          = STIM_EPO / "b_stim"
    INDICES_STIM    = STIM_EPO / "indices_stim"
    START_REST_EPO  = EPOCH_DIR / "strt_rst"
    END_REST_EPO    = EPOCH_DIR / "end_rst"

    # Labeled epochs
    RAW_LABEL_DIR   = INTRMD_DIR / "raw_label"
    LABELED_DIR     = INTRMD_DIR / "labeled"

    # ── Features ──────────────────────────────────────────────────────────
    FEATURES_DIR    = DATA_ROOT / "features"
    ROI_BP_DIR      = FEATURES_DIR / "roi_bp"
    BP_DIR          = FEATURES_DIR / "bp"

    # ── Logs ──────────────────────────────────────────────────────────────
    LOG_PREPROCESS  = LOG_ROOT / "preprs"
    LOG_FEATURES    = LOG_ROOT / "fetrs"
    LOG_LABELING    = LOG_ROOT / "labeling"
    LOG_ML          = LOG_ROOT / "ml"
    LOG_DL          = LOG_ROOT / "dl"
    
    # ── Results ──────────────────────────────────────────────────────────────
    REPORTS_DIR = RESULTS_DIR / "reports"
    DESCRIPTIVE_DIR = RESULTS_DIR / "descriptive"

    ML_DIR = RESULTS_DIR / "ml_pipeline"
    ML_BINARY_DIR = ML_DIR / "ml_pipeline_binary"
    # ML_BINARY2_DIR = ML_DIR / "ml_pipeline_binary_2"

    DL_DIR = RESULTS_DIR / "dl_results"
    EEGNET_DIR = DL_DIR / "eegnet"
    CNNLSTM_DIR = DL_DIR / "cnnlstm"

    PLOTS_DIR = RESULTS_DIR / "plots"
    MODELS_DIR = RESULTS_DIR / "saved_models"

    @classmethod
    def make_all(cls) -> None:
        """Create every output directory that does not yet exist."""
        dirs = [
            cls.FILTER_DIR, cls.ICA_DIR,
            cls.WHOLE_STIM_EPO, cls.G_STIM, cls.B_STIM, cls.INDICES_STIM,
            cls.START_REST_EPO, cls.END_REST_EPO,
            cls.RAW_LABEL_DIR, cls.LABELED_DIR,
            cls.ROI_BP_DIR, cls.BP_DIR,
            cls.LOG_PREPROCESS, cls.LOG_FEATURES, cls.LOG_LABELING,
            cls.LOG_ML, cls.LOG_DL,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
