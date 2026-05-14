"""
EEG Preprocessing Pipeline (Batch + Single Mode)
Author: Pramit Biswas (MSc CS)

Version: v1.0

Description:
------------
Automated EEG preprocessing using MNE-Python.

Experimental design (from protocol):
- 210 trials total (70 Haiku + 70 Senryu + 70 Control), 7 blocks x 30 trials
- Per-trial epoch window: -4 to +11 s (4s baseline, 10s stimulus, 1s buffer)
- Stimulus codes: 65281-65284 (multiple markers per trial)
- Rest codes: 65285, 65286 (start) and 65287, 65288 (end)

Changes in v1.0:
----------------
- Re-referencing: EXG1 & EXG2 set as earlobe reference channels; average of both
  used as the re-reference signal before dropping EXG channels.
- ICA: fixed to exactly 64 components (Infomax/runica equivalent), fitted ONLY
  on scalp EEG channels (picks="eeg") after EXG channels are dropped.

Pipeline Steps:
1.  Load EEGLAB data
2.  Set channel types & montage
3.  Re-reference to average of EXG1 & EXG2 (earlobes)
4.  Drop all EXG channels — retain only 64 scalp EEG channels
5.  Notch + bandpass filtering
6.  ICA artifact removal (64 components, scalp channels only)
7.  Event extraction + event id naming + DIAGNOSIS + onset-code selection
8.  Epoching + baseline correction
9.  Bad epoch rejection
10. Save outputs
"""

import os
import mne
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# ==============================
# CONFIG
# ==============================

CONFIG = {
    "sampling_rate": 512,
    "bandpass": (0.5, 48),
    "notch": [50, 100],
    "epoch_tmin": -4.0,
    "epoch_tmax": 11.0,
    "baseline": (-4.0, 0.0),
    "reject_threshold": 200e-6,
    "reject_tmin": 0.0,
    "reject_tmax": 10.0,
    # ICA: exactly 64 components (rank of 64-channel scalp data, no PCA reduction)
    "ica_components": 64,
    "stim_onset": 65282,
    "stim_codes": [65281, 65282, 65283, 65284],
    "rest_start": (65285, 65286),
    "rest_end": (65287, 65288),
    "rest_window": 4.0,
    # EXG channel roles (BioSemi ActiveTwo)
    # EXG1 & EXG2 → earlobe reference electrodes (used for re-referencing)
    # EXG3 & EXG4 → vertical EOG (above & below right eye)
    # EXG7 & EXG8 → horizontal EOG (outer canthi of both eyes)
    # EXG5 & EXG6 → ECG (right collarbone & left waist) — recorded, not analyzed
    "earlobe_channels": ["EXG1", "EXG2"],
    "eog_channels": ["EXG3", "EXG4", "EXG7", "EXG8"],
    "ecg_channels": ["EXG5", "EXG6"],
    # All EXG channels to drop after re-referencing
    "exg_channels": ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"],
}

# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
LOG_DIR = (BASE_DIR / "../../logs/preprs").resolve()

RAW_DIR = DATA_DIR / "raw_data/ds006648-download"
FILTER_DIR = DATA_DIR / "intrmd_data/filtered"
ICA_DIR = DATA_DIR / "intrmd_data/ica_signal"
EPOCH_DIR = DATA_DIR / "intrmd_data/epochs"

START_REST_EPO = EPOCH_DIR / "strt_rst"
END_REST_EPO = EPOCH_DIR / "end_rst"
STIM_EPO = EPOCH_DIR / "stim"
G_STIM = STIM_EPO / "g_stim"
B_STIM = STIM_EPO / "b_stim"
INDICES_STIM = STIM_EPO / "indices_stim"
WHOLE_STIM_EPO = STIM_EPO / "whole_stim"

# Create all directories (including logs)
for d in [FILTER_DIR, ICA_DIR, START_REST_EPO, END_REST_EPO, STIM_EPO, WHOLE_STIM_EPO, G_STIM, B_STIM, INDICES_STIM, LOG_DIR,]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================
# LOGGER
# ==============================

def setup_logger(subject_id: str) -> logging.Logger:
    """
    Create (or retrieve) a per-subject file logger.

    Parameters
    ----------
    subject_id : str
        Subject identifier used as logger name and log filename.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(subject_id)
    logger.setLevel(logging.INFO)

    log_file = LOG_DIR / f"{subject_id}_log.log"
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Prevent duplicate handlers on repeated calls
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# ==============================
# READ EEG
# ==============================

def read_eeg(file_path: str) -> mne.io.Raw:
    """
    Load an EEGLAB .set file into an MNE Raw object.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the .set file.

    Returns
    -------
    mne.io.Raw
        Preloaded Raw object.
    """
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    return raw


# ==============================
# SET CHANNELS
# ==============================

def set_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Assign channel types for EXG channels and apply the BioSemi 64 montage.

    Channel assignment (BioSemi ActiveTwo protocol):
    - EXG1, EXG2  → earlobe reference electrodes  (typed 'eeg' temporarily
                     so set_eeg_reference can include them; renamed after
                     re-referencing)
    - EXG3, EXG4  → vertical EOG
    - EXG5, EXG6  → ECG
    - EXG7, EXG8  → horizontal EOG

    The montage covers only the 64 scalp channels; EXG channels are ignored
    (on_missing='ignore').

    Parameters
    ----------
    raw : mne.io.Raw

    Returns
    -------
    mne.io.Raw
        Raw object with updated channel types and montage.
    """
    raw.set_channel_types(
        {
            # Earlobes: keep as 'eeg' so they participate in re-referencing
            "EXG1": "eeg",
            "EXG2": "eeg",
            # EOG channels
            "EXG3": "eog",
            "EXG4": "eog",
            "EXG7": "eog",
            "EXG8": "eog",
            # ECG channels (recorded, not analyzed)
            # "EXG5": "ecg",
            # "EXG6": "ecg",
        }
    )
    raw.set_montage("biosemi64", on_missing="ignore")
    return raw


# ==============================
# RE-REFERENCE
# ==============================

def rereference_eeg(raw: mne.io.Raw, logger: logging.Logger) -> mne.io.Raw:
    """
    Re-reference scalp EEG data to the average of EXG1 & EXG2 (earlobes),
    then drop ALL EXG channels so that only the 64 BioSemi scalp channels
    remain for subsequent processing.

    Steps
    -----
    1. Mark EXG1 & EXG2 explicitly as reference channels via
       ``add_reference_channels`` is NOT used here because they are already
       recorded. Instead we set their type to 'eeg' in set_channels() so
       MNE can compute the average.
    2. Apply ``set_eeg_reference(['EXG1', 'EXG2'])`` — subtracts the mean of
       both earlobe channels from every EEG channel.
    3. Drop all EXG channels (EXG1–EXG8) to retain exactly 64 scalp channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object after set_channels().
    logger : logging.Logger

    Returns
    -------
    mne.io.Raw
        64-channel scalp-only Raw object, re-referenced to earlobe average.
    """
    earlobe_chs = CONFIG["earlobe_channels"]   # ['EXG1', 'EXG2']
    exg_chs     = CONFIG["exg_channels"]       # all EXG1–EXG8

    logger.info(f"Re-referencing to average of {earlobe_chs}")
    raw.set_eeg_reference(ref_channels=earlobe_chs, projection=False)

    # # Drop all EXG channels — leaves exactly 64 scalp EEG channels
    # existing_exg = [ch for ch in exg_chs if ch in raw.ch_names]
    # raw.drop_channels(existing_exg)
    # logger.info(
    #     f"Dropped EXG channels: {existing_exg}. "
    #     f"Remaining channels: {len(raw.ch_names)}"
    # )
    return raw


# ==============================
# FILTER
# ==============================

def filter_eeg(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply notch filter (50 & 100 Hz) followed by bandpass filter (0.5–48 Hz).

    Parameters
    ----------
    raw : mne.io.Raw

    Returns
    -------
    mne.io.Raw
    """
    raw.notch_filter(CONFIG["notch"])
    raw.filter(CONFIG["bandpass"][0], CONFIG["bandpass"][1])
    return raw


# ==============================
# SAVE RAW
# ==============================

def save_raw(raw: mne.io.Raw, path: Path) -> None:
    """
    Save a Raw object to disk as a .fif file.

    Parameters
    ----------
    raw  : mne.io.Raw
    path : Path
        Destination file path (should end in _raw.fif).
    """
    raw.save(path, overwrite=True)


# ==============================
# ICA
# ==============================

def run_ica(raw: mne.io.Raw, logger: logging.Logger) -> mne.io.Raw:
    """
    Run Independent Component Analysis on the 64 scalp EEG channels and
    remove EOG-related components.

    ICA configuration (matching original EEGLAB preprocessing):
    - n_components = 64  (full-rank decomposition; no prior PCA reduction,
      equivalent to EEGLAB's ``runica`` with 64 channels)
    - method = 'infomax' (same algorithm as EEGLAB runica)
    - fit_params extended=True → Extended Infomax (handles super- and
      sub-Gaussian sources, recommended for EEG)
    - random_state = 97 for reproducibility
    - picks = 'eeg' → fitted only on the 64 scalp channels (EXG channels
      have already been dropped by rereference_eeg)

    EOG component detection uses find_bads_eog(). Because the EXG EOG
    channels (EXG3/4/7/8) were dropped, MNE will fall back to using frontal
    EEG channels (Fp1, Fp2) as EOG proxies — consistent with the original
    manual visual-inspection approach.

    Parameters
    ----------
    raw    : mne.io.Raw
        64-channel scalp-only Raw object.
    logger : logging.Logger

    Returns
    -------
    mne.io.Raw
        ICA-cleaned Raw object (copy; original untouched).
    """
    ica = mne.preprocessing.ICA(
        n_components=CONFIG["ica_components"],   # 64 — full rank
        method="infomax",
        fit_params={"extended": True},           # Extended Infomax
        random_state=97,
    )

    # Fit on EEG channels only (all 64 scalp channels after EXG drop)
    ica.fit(raw, picks="eeg", decim=3)
    logger.info(f"ICA fitted: {ica.n_components_} components")

    # Identify EOG artifacts — uses Fp1/Fp2 as proxy since EXG EOG dropped
    eog_inds, eog_scores = ica.find_bads_eog(
        raw, ch_name=["EXG3", "EXG4", "EXG7", "EXG8"]
    )
    ica.exclude = eog_inds
    logger.info(f"ICA excluded components (EOG): {eog_inds}")

    raw_clean = ica.apply(raw.copy())
    return raw_clean


# ==============================
# EVENT EXTRACTION
# ==============================

def extract_events(
    raw: mne.io.Raw,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict]:
    """
    Extract events from annotations and log the distribution.

    Parameters
    ----------
    raw    : mne.io.Raw
    logger : logging.Logger

    Returns
    -------
    events   : np.ndarray, shape (n_events, 3)
    event_id : dict  {str_label: int_code}
    """
    events, event_id = mne.events_from_annotations(raw)

    logger.info(f"Total events found: {len(events)}")
    logger.info(f"Event ID mapping: {event_id}")

    unique, counts = np.unique(events[:, 2], return_counts=True)
    event_dict = dict(zip(unique, counts))
    logger.info(f"Event distribution: {event_dict}")

    return events, event_id


# ==============================
# STIM EPOCHING (ONE PER TRIAL)
# ==============================

def epoch_stimulus(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    logger: logging.Logger,
) -> mne.Epochs:
    """
    Create one epoch per stimulus trial, anchored on marker 65282.

    Steps
    -----
    1. Select events with code 65282 (trial-onset marker).
    2. Epoch from tmin=-4 s to tmax=+11 s with baseline (-4, 0).
    3. Drop the first 2 practice trials.
    4. Attach metadata with a clean 0-indexed trial_index column.

    Parameters
    ----------
    raw      : mne.io.Raw
    events   : np.ndarray
    event_id : dict
    logger   : logging.Logger

    Returns
    -------
    mne.Epochs
        210 clean stimulus epochs (post-practice removal).
    """
    if "65282" not in event_id:
        raise ValueError("Marker 65282 not found in annotations — check event codes.")

    stim_code   = event_id["65282"]
    stim_events = events[events[:, 2] == stim_code]
    logger.info(f"Stim trials found (including practice): {len(stim_events)}")

    epochs = mne.Epochs(
        raw,
        stim_events,
        tmin=CONFIG["epoch_tmin"],
        tmax=CONFIG["epoch_tmax"],
        baseline=CONFIG["baseline"],
        preload=True,
    )

    # Remove first 2 practice trials
    epochs = epochs[2:]
    logger.info(f"After removing practice trials: {len(epochs)} trials (expected 210)")

    # Assign clean 0-based trial index for downstream labelling
    epochs.metadata = pd.DataFrame({"trial_index": np.arange(len(epochs))})

    return epochs


# ==============================
# REST EPOCHING (4 s WINDOWS)
# ==============================

def epoch_rest(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    rest_pair: tuple,
    label: str,
    logger: logging.Logger,
) -> mne.Epochs | None:
    """
    Segment a rest interval into non-overlapping 4 s windows.

    The rest interval is bounded by a start marker and an end marker. The
    duration is divided into floor(duration / 4) windows; each window is
    encoded as a synthetic event at sample ``start + i * 4 s``.

    Parameters
    ----------
    raw       : mne.io.Raw
    events    : np.ndarray
    event_id  : dict
    rest_pair : tuple of (int, int)
        (start_code, end_code) for the rest interval.
    label     : str
        Human-readable label used in log messages.
    logger    : logging.Logger

    Returns
    -------
    mne.Epochs or None
        None if start/end markers are missing.
    """
    start_code_str, end_code_str = map(str, rest_pair)

    if start_code_str not in event_id or end_code_str not in event_id:
        logger.warning(f"{label} rest events missing — skipping.")
        return None

    start_code   = event_id[start_code_str]
    end_code     = event_id[end_code_str]
    start_events = events[events[:, 2] == start_code]
    end_events   = events[events[:, 2] == end_code]

    if len(start_events) == 0 or len(end_events) == 0:
        logger.warning(f"{label} rest events empty — skipping.")
        return None

    sfreq    = raw.info["sfreq"]
    start_t  = start_events[0][0] / sfreq
    end_t    = end_events[0][0] / sfreq
    duration = end_t - start_t
    window   = CONFIG["rest_window"]
    n_segs   = int(duration // window)

    logger.info(f"{label} rest duration: {duration:.2f} s → {n_segs} × {window} s segments")

    synthetic_events = np.array(
        [[int((start_t + i * window) * sfreq), 0, 999] for i in range(n_segs)]
    )

    epochs = mne.Epochs(
        raw,
        synthetic_events,
        tmin=0,
        tmax=window,
        baseline=None,
        preload=True,
    )
    return epochs


# ==============================
# BAD EPOCH REJECTION
# ==============================

def reject_epochs(
    epochs: mne.Epochs,
    logger: logging.Logger,
    label: str,
) -> tuple[mne.Epochs, mne.Epochs, list, list]:
    """
    Reject epochs whose peak-to-peak amplitude exceeds the threshold
    (200 µV) and return separate good / bad epoch objects plus their
    original indices.

    Parameters
    ----------
    epochs : mne.Epochs
    logger : logging.Logger
    label  : str
        Label for log messages (e.g. 'STIM', 'START REST').

    Returns
    -------
    epochs_good : mne.Epochs
    epochs_bad  : mne.Epochs
    good_idx    : list of int
    bad_idx     : list of int
    """
    reject_criteria = dict(eeg=CONFIG["reject_threshold"])
    before       = len(epochs)
    original_idx = np.arange(before)

    epochs_copy = epochs.copy()
    epochs_copy.drop_bad(reject=reject_criteria)

    after    = len(epochs_copy)
    drop_log = epochs_copy.drop_log[:before]

    good_mask = np.array([len(log) == 0 for log in drop_log])
    bad_mask  = ~good_mask

    good_idx = original_idx[good_mask].tolist()
    bad_idx  = original_idx[bad_mask].tolist()

    logger.info(f"{label} | before: {before}, after: {after}")
    logger.info(f"{label} | good: {len(good_idx)}, bad: {len(bad_idx)}")
    logger.info(f"{label} | first 10 good_idx: {good_idx[:10]}")
    logger.info(f"{label} | first 10  bad_idx: {bad_idx[:10]}")

    epochs_good = epochs[good_idx]
    epochs_bad  = epochs[bad_idx]

    return epochs_good, epochs_bad, good_idx, bad_idx


# ==============================
# SAVE EPOCHS
# ==============================

def save_epochs(epochs: mne.Epochs | None, path: Path) -> None:
    """
    Save an Epochs object to disk. No-op if epochs is None.

    Parameters
    ----------
    epochs : mne.Epochs or None
    path   : Path
    """
    if epochs is not None:
        epochs.save(path, overwrite=True)


# ==============================
# SINGLE SUBJECT PIPELINE
# ==============================

def preprocessing_pipeline(file_path: str) -> None:
    """
    Run the full preprocessing pipeline for a single subject.

    Pipeline order
    --------------
    1.  Read raw EEGLAB data.
    2.  Set channel types & montage (EXG1/2 kept as 'eeg' for referencing).
    3.  Re-reference to average of EXG1 & EXG2 (earlobes); drop all EXG channels.
    4.  Notch + bandpass filtering on 64 scalp channels.
    5.  ICA (64 components, Extended Infomax) on scalp channels; remove EOG ICs.
    6.  Extract events.
    7.  Epoch stimulus trials (marker 65282); drop 2 practice trials.
    8.  Save all 210 epochs (whole_stim).
    [Steps 9–11 commented out — uncomment to enable rejection & rest epoching]
    9.  Reject bad epochs; save good / bad + index arrays.
    10. Epoch start-rest & end-rest intervals (4 s windows).
    11. Reject bad rest epochs; save.

    Parameters
    ----------
    file_path : str
        Path to the subject's .set file.
    """
    subject_id = Path(file_path).parts[-3]
    logger     = setup_logger(subject_id)

    try:
        logger.info(f"=== Processing {subject_id} ===")

        # Step 1 — Load
        raw = read_eeg(file_path)

        # Step 2 — Assign channel types & montage
        raw = set_channels(raw)

        # Step 3 — Re-reference to earlobes; drop EXG channels
        raw = rereference_eeg(raw, logger)

        # Step 4 — Filter (notch + bandpass)
        raw = filter_eeg(raw)
        save_raw(raw, FILTER_DIR / f"{subject_id}_filtered_raw.fif")

        # Step 5 — ICA (64 components, scalp channels only)
        raw = run_ica(raw, logger)
        save_raw(raw, ICA_DIR / f"{subject_id}_ica_raw.fif")
        
        # Drop EXG channels after ICA artifact detection
        existing_exg = [ch for ch in CONFIG["exg_channels"] if ch in raw.ch_names]
        raw.drop_channels(existing_exg)
        logger.info(f"Dropped EXG channels after ICA: {existing_exg}")

        # Step 6 — Events
        events, event_id = extract_events(raw, logger)

        # ---- STIMULUS EPOCHS ----

        # Step 7 — Epoch (210 trials, practice removed)
        stim_epochs = epoch_stimulus(raw, events, event_id, logger)

        # Step 8 — Save all 210 epochs
        save_epochs(stim_epochs, WHOLE_STIM_EPO / f"{subject_id}_allstim_epo.fif")

        # Step 9 — Reject bad epochs (uncomment to activate)
        # stim_good, stim_bad, good_idx, bad_idx = reject_epochs(stim_epochs, logger, "STIM")
        # save_epochs(stim_good, G_STIM / f"{subject_id}_stim_epo.fif")
        # save_epochs(stim_bad,  B_STIM / f"{subject_id}_badstim_epo.fif")
        # np.savez(
        #     INDICES_STIM / f"{subject_id}_indices.npz",
        #     good_idx=good_idx,
        #     bad_idx=bad_idx,
        # )

        # ---- REST EPOCHS ----

        # Step 10 — Epoch rest intervals (uncomment to activate)
        # start_rest = epoch_rest(raw, events, event_id, CONFIG["rest_start"], "START", logger)
        # end_rest   = epoch_rest(raw, events, event_id, CONFIG["rest_end"],   "END",   logger)

        # Step 11 — Reject rest bad epochs (uncomment to activate)
        # if start_rest:
        #     start_good, _, _, _ = reject_epochs(start_rest, logger, "START REST")
        #     save_epochs(start_good, START_REST_EPO / f"{subject_id}_strt_rest_epo.fif")
        # if end_rest:
        #     end_good, _, _, _ = reject_epochs(end_rest, logger, "END REST")
        #     save_epochs(end_good, END_REST_EPO / f"{subject_id}_end_rest_epo.fif")

        logger.info(f"=== {subject_id} completed successfully ===")

    except Exception as e:
        logger.error(f"Error processing {subject_id}: {e}", exc_info=True)


# ==============================
# BATCH PROCESSING
# ==============================

def already_processed(subject_id: str) -> bool:
    """
    Check whether all expected output files for a subject already exist.

    Parameters
    ----------
    subject_id : str

    Returns
    -------
    bool
        True if all required output files are present on disk.
    """
    required_files = [
        FILTER_DIR    / f"{subject_id}_filtered_raw.fif",
        ICA_DIR       / f"{subject_id}_ica_raw.fif",
        WHOLE_STIM_EPO / f"{subject_id}_allstim_epo.fif",
    ]
    return all(f.exists() for f in required_files)


def batch_process() -> None:
    """
    Run the preprocessing pipeline on all subjects found under RAW_DIR,
    skipping any that have already been processed.
    """
    files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))
    print(f"Found {len(files)} subject file(s)")

    for file in files:
        subject_id = Path(file).parts[-3]
        if already_processed(subject_id):
            print(f"Skipping {subject_id} (already processed)")
            continue
        print(f"Processing {subject_id} ...")
        preprocessing_pipeline(str(file))


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))

    if len(files) == 0:
        print("No .set files found under RAW_DIR.")
    else:
        # DEBUG — single subject (uncomment and point to the desired file)
        preprocessing_pipeline(str(files[0]))

        # BATCH — all subjects
        # batch_process()
