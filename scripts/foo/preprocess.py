"""
EEG Preprocessing Pipeline (Batch + Single Mode)
Author: Pramit Biswas (MSc CS)

Version: v0.0

Description:
------------
Automated EEG preprocessing using MNE-Python.

Experimental design (from protocol):
- 210 trials total (70 Haiku + 70 Senryu + 70 Control), 7 blocks x 30 trials
- Per-trial epoch window: -4 to +11 s (4s baseline, 10s stimulus, 1s buffer)
- Stimulus codes: 65281-65284 (multiple markers per trial - see note below)
- Rest codes: 65285, 65286 (start) and 65287, 65288 (end)

Steps:
1. Load EEGLAB data
2. Set channel types & montage
3. Notch + bandpass filtering
4. ICA artifact removal
5. Event extraction + event id naming + DIAGNOSIS + onset-code selection
6. Epoching + baseline correction
7. Bad epoch rejection
8. Save outputs
"""

import os
import mne
import numpy as np
import logging
from pathlib import Path

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

# create all directories (including logs)
for d in [FILTER_DIR, ICA_DIR, START_REST_EPO, END_REST_EPO, STIM_EPO, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================
# LOGGER
# ==============================


def setup_logger(subject_id):
    logger = logging.getLogger(subject_id)
    logger.setLevel(logging.INFO)

    log_file = LOG_DIR / f"{subject_id}_log.log"  # ✅ use Path

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # ✅ prevent duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# ==============================
# READ EEG
# ==============================


def read_eeg(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    return raw


# ==============================
# SET CHANNELS
# ==============================


def set_channels(raw):
    raw.set_channel_types(
        {
            "EXG1": "eog",
            "EXG2": "eog",
            "EXG3": "eog",
            "EXG4": "eog",
            "EXG7": "ecg",
            "EXG8": "ecg",
        }
    )
    raw.set_montage("biosemi64", on_missing="ignore")
    return raw


# ==============================
# FILTER
# ==============================


def filter_eeg(raw):
    raw.notch_filter(CONFIG["notch"])
    raw.filter(CONFIG["bandpass"][0], CONFIG["bandpass"][1])
    return raw


# ==============================
# SAVE RAW
# ==============================


def save_raw(raw, path):
    raw.save(path, overwrite=True)


# ==============================
# ICA
# ==============================


def run_ica(raw, logger):
    ica = mne.preprocessing.ICA(n_components=CONFIG["ica_components"], random_state=97)
    ica.fit(raw, decim=3)

    eog_inds, _ = ica.find_bads_eog(raw)
    ica.exclude = eog_inds

    logger.info(f"ICA removed components: {eog_inds}")

    raw_clean = ica.apply(raw.copy())
    return raw_clean


# ==============================
# EVENT EXTRACTION
# ==============================


def extract_events(raw, logger):
    events, event_id = mne.events_from_annotations(raw)

    logger.info(f"Total events found: {len(events)}")
    logger.info(f"Event ID mapping: {event_id}")

    # count events
    unique, counts = np.unique(events[:, 2], return_counts=True)
    event_dict = dict(zip(unique, counts))
    logger.info(f"Event distribution: {event_dict}")

    return events, event_id


# ==============================
# STIM EPOCHING (ONE PER TRIAL)
# ==============================


def epoch_stimulus(raw, events, event_id, logger):
    if "65282" not in event_id:
        raise ValueError("65281 not found in annotations")

    stim_code = event_id["65282"]

    stim_events = events[events[:, 2] == stim_code]

    logger.info(f"Stim trials found: {len(stim_events)}")

    epochs = mne.Epochs(
        raw,
        stim_events,
        tmin=CONFIG["epoch_tmin"],
        tmax=CONFIG["epoch_tmax"],
        baseline=CONFIG["baseline"],
        # baseline=None,
        preload=True,
    )
    # epochs.apply_baseline(CONFIG["baseline"])
    print("Baseline inside function:", epochs.baseline)
    return epochs


# ==============================
# REST EPOCHING (4s windows)
# ==============================


def epoch_rest(raw, events, event_id, rest_pair, label, logger):
    start_code_str, end_code_str = map(str, rest_pair)

    if start_code_str not in event_id or end_code_str not in event_id:
        logger.warning(f"{label} rest events missing")
        return None

    start_code = event_id[start_code_str]
    end_code = event_id[end_code_str]

    start_events = events[events[:, 2] == start_code]
    end_events = events[events[:, 2] == end_code]

    if len(start_events) == 0 or len(end_events) == 0:
        logger.warning(f"{label} rest events missing")
        return None

    start = start_events[0][0] / raw.info["sfreq"]
    end = end_events[0][0] / raw.info["sfreq"]

    duration = end - start
    window = CONFIG["rest_window"]

    n_segments = int(duration // window)

    logger.info(f"{label} rest duration: {duration}s → {n_segments} segments")

    events_new = []
    for i in range(n_segments):
        t = start + i * window
        sample = int(t * raw.info["sfreq"])
        events_new.append([sample, 0, 999])

    events_new = np.array(events_new)

    epochs = mne.Epochs(
        raw,
        events_new,
        tmin=0,
        tmax=window,
        baseline=None,
        preload=True,
    )
    # epochs.apply_baseline(CONFIG["baseline"])
    return epochs


# ==============================
# BAD EPOCH REJECTION
# ==============================


def reject_epochs(epochs, logger, label):
    reject_criteria = dict(eeg=CONFIG["reject_threshold"])

    before = len(epochs)

    epochs.drop_bad(reject=reject_criteria)

    after = len(epochs)

    logger.info(f"{label} epochs before: {before}, after rejection: {after}")

    return epochs


# ==============================
# SAVE EPOCHS
# ==============================


def save_epochs(epochs, path):
    if epochs is not None:
        epochs.save(path, overwrite=True)


# ==============================
# SINGLE SUBJECT PIPELINE
# ==============================


def preprocessing_pipeline(file_path):
    subject_id = Path(file_path).parts[-3]
    logger = setup_logger(subject_id)
    
    try:
        logger.info(f"Processing {subject_id}")

        raw = read_eeg(file_path)
        raw = set_channels(raw)

        raw = filter_eeg(raw)
        save_raw(raw, FILTER_DIR / f"{subject_id}_filtered_raw.fif")

        raw = run_ica(raw, logger)
        save_raw(raw, ICA_DIR / f"{subject_id}_ica_raw.fif")

        events, event_id = extract_events(raw, logger)

        # ---- STIM ----
        stim_epochs = epoch_stimulus(raw, events, event_id, logger)
        # stim_epochs = reject_epochs(stim_epochs, logger, "STIM")
        save_epochs(stim_epochs, STIM_EPO / f"{subject_id}_stim_epo.fif")

        # ---- REST ----
        start_rest = epoch_rest(raw, events, event_id, CONFIG["rest_start"], "START", logger)
        end_rest = epoch_rest(raw, events, event_id, CONFIG["rest_end"], "END", logger)

        if start_rest:
            start_rest = reject_epochs(start_rest, logger, "START REST")
            save_epochs(start_rest, START_REST_EPO / f"{subject_id}_strt_rest_epo.fif")

        if end_rest:
            end_rest = reject_epochs(end_rest, logger, "END REST")
            save_epochs(end_rest, END_REST_EPO / f"{subject_id}_end_rest_epo.fif")

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Error processing {subject_id}: {str(e)}")


# ==============================
# BATCH PROCESS
# ==============================


def batch_process():
    files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))

    print(f"Found {len(files)} files")

    for file in files:
        preprocessing_pipeline(str(file))


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))

    if len(files) == 0:
        print("No files found")
    else:
        # 👉 DEBUG MODE (IMPORTANT FOR YOU)
        preprocessing_pipeline(str(files[0]))

        # 👉 AFTER VALIDATION
        # batch_process()
