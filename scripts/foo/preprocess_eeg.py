"""
EEG Preprocessing Pipeline (Batch + Single Mode)

Author: <your_name>

Description:
------------
Automated EEG preprocessing using MNE-Python.

Steps:
1. Load EEGLAB data
2. Set channel types & montage
3. Notch + bandpass filtering
4. ICA artifact removal
5. Event extraction
6. Epoching + baseline correction
7. Bad epoch rejection
8. Save outputs

Modes:
------
- single: process one file
- batch: process all files
"""

import os
import mne
import numpy as np
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

CONFIG = {
    "sampling_rate": 512,
    "filter_type": "highpass",  # "highpass" or "bandpass"
    "highpass": 0.5,
    "bandpass": (0.5, 40),
    "epoch_window": (-0.4, 10),
    "baseline": (-0.4, 0),
    "reject_threshold": 100e-6,
    "ica_components": 20,
    "plot": False,  # Disable plots in batch mode
}

# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

RAW_DIR = DATA_DIR / "raw_data/ds006648-download"
FILTER_DIR = DATA_DIR / "intrmd_data/filtered"
EPOCH_DIR = DATA_DIR / "intrmd_data/epochs"

FILTER_DIR.mkdir(parents=True, exist_ok=True)
EPOCH_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# PREPROCESS FUNCTION
# ==============================

def preprocess_file(filepath):
    try:
        subject = next(p for p in Path(filepath).parts if p.startswith("sub-"))
        print(f"\n===== Processing {subject} =====")

        # ------------------------------
        # 1. Load raw EEG
        # ------------------------------
        raw = mne.io.read_raw_eeglab(filepath, preload=True)

        raw.set_channel_types({
            "EXG1": "eog", "EXG2": "eog",
            "EXG3": "eog", "EXG4": "eog",
            "EXG7": "ecg", "EXG8": "ecg",
        })

        raw.set_montage("biosemi64", on_missing="ignore")

        if CONFIG["plot"]:
            raw.plot_sensors(show_names=True)
            raw.plot(duration=10)

        print("Loaded raw data")

        # ------------------------------
        # 2. Filtering
        # ------------------------------
        raw.notch_filter(freqs=[50, 100], notch_widths=4)

        if CONFIG["filter_type"] == "highpass":
            raw.filter(l_freq=CONFIG["highpass"], h_freq=None, picks="eeg")

        elif CONFIG["filter_type"] == "bandpass":
            raw.filter(
                l_freq=CONFIG["bandpass"][0],
                h_freq=CONFIG["bandpass"][1],
                picks="eeg"
            )

        print("Filtering done")

        # Save filtered data
        filtered_path = FILTER_DIR / f"{subject}_filtered.fif"
        raw.save(filtered_path, overwrite=True)

        if CONFIG["plot"]:
            raw.plot_psd(fmax=60)

        # ------------------------------
        # 3. ICA
        # ------------------------------
        print("Running ICA...")

        ica = mne.preprocessing.ICA(
            n_components=CONFIG["ica_components"],
            random_state=97,
            max_iter="auto"
        )

        ica.fit(raw, decim=3)

        eog_indices, _ = ica.find_bads_eog(raw)
        ica.exclude = eog_indices

        raw_clean = ica.apply(raw.copy())

        print(f"Removed ICA components: {ica.exclude}")

        if CONFIG["plot"]:
            raw_clean.plot(duration=10)

        # ------------------------------
        # 4. Events
        # ------------------------------
        events, event_id = mne.events_from_annotations(raw_clean)

        print(f"Found {len(events)} events")

        if CONFIG["plot"]:
            mne.viz.plot_events(events)

        # ------------------------------
        # 5. Epoching
        # ------------------------------
        epochs = mne.Epochs(
            raw_clean,
            events,
            event_id=event_id,
            tmin=CONFIG["epoch_window"][0],
            tmax=CONFIG["epoch_window"][1],
            baseline=CONFIG["baseline"],
            preload=True
        )

        print(f"Epochs created: {len(epochs)}")

        if CONFIG["plot"]:
            epochs.average().plot()

        # ------------------------------
        # 6. Reject bad epochs
        # ------------------------------
        epochs.drop_bad(reject=dict(eeg=CONFIG["reject_threshold"]))

        print(f"Remaining epochs after rejection: {len(epochs)}")

        # ------------------------------
        # 7. Save epochs
        # ------------------------------
        save_path = EPOCH_DIR / f"{subject}_epo.fif"
        epochs.save(save_path, overwrite=True)

        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"❌ Error processing {filepath}")
        print(e)


# ==============================
# RUN MODES
# ==============================

def run_batch():
    eeg_files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))
    print(f"Found {len(eeg_files)} files")

    for file in eeg_files:
        preprocess_file(str(file))


def run_single():
    eeg_files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))

    if len(eeg_files) == 0:
        print("No EEG files found")
        return

    preprocess_file(str(eeg_files[0]))


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    MODE = "batch"   # change to "single" for testing

    if MODE == "single":
        CONFIG["plot"] = True
        run_single()

    elif MODE == "batch":
        CONFIG["plot"] = False
        run_batch()