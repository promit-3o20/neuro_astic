import os
import glob
import mne
import numpy as np
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

CONFIG = {
    "sampling_rate": 512,
    "filter_type": "bandpass",  # "highpass" or "bandpass"
    "highpass": 0.5,
    "bandpass": (0.5, 40),
    "epoch_window": (-0.2, 1.0),
    "baseline": (-0.2, 0),
    "reject_threshold": 100e-6,
    "ica_components": 20,
}

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

    # subject = os.path.basename(filepath).split(".")[0]
    subject = next(p for p in Path(filepath).parts if p.startswith("sub-"))

    print(f"\nProcessing {subject}")

    # ------------------------------
    # 1 Load raw EEG
    # ------------------------------

    raw = mne.io.read_raw_eeglab(filepath, preload=True)

    # Set channel types for external electrodes
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
    print("Loaded raw signal")
    raw.set_montage("biosemi64", on_missing="ignore")
    raw.plot_sensors(show_names= True)
    # Visualize raw
    raw.plot(duration=10);
    # ------------------------------
    # 2 filter
    # ------------------------------
    # Notch filter
    raw.notch_filter(freqs=[50, 100], notch_widths=4)

    print("Notch filtering done")

    # Flexible filtering
    if CONFIG["filter_type"] == "highpass":
        raw.filter(l_freq=CONFIG["highpass"], h_freq=None, picks="eeg")
        print(f"High-pass filter applied at {CONFIG['highpass']} Hz")

    elif CONFIG["filter_type"] == "bandpass":
        raw.filter(l_freq=CONFIG["bandpass"][0], h_freq=CONFIG["bandpass"][1], picks="eeg")
        print(f"Band-pass filter {CONFIG['bandpass']} Hz")

    print("Bandpass filtering done")

    # save filtered version
    filtered_path = os.path.join(
        FILTER_DIR, f"{subject}_filtered.fif"
    )
    raw.save(filtered_path, overwrite=True)

    # After filtering
    raw.plot_psd(fmax=60)

    # ------------------------------
    # 3 ICA artifact removal
    # ------------------------------

    print("Running ICA")

    ica = mne.preprocessing.ICA(
        n_components=CONFIG["ica_components"],
        random_state=97,
        max_iter="auto"
    )

    ica.fit(raw, decim=3)

    # automatic artifact detection
    eog_indices, _ = ica.find_bads_eog(raw)

    ica.exclude = eog_indices

    raw_clean = ica.apply(raw.copy())

    print(f"Removed ICA components: {ica.exclude}")

    # # ICA components
    # ica.plot_components();
    # Clean signal
    raw_clean.plot(duration=10);
    # ------------------------------
    # 4 Event detection
    # ------------------------------

    events, event_id = mne.events_from_annotations(raw_clean)

    print(f"Found {len(events)} events")
    print("Event mapping:", event_id)
    # Events
    mne.viz.plot_events(events);
    # ------------------------------
    # 5 Epoch extraction
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
    print("Epoching done")
    # Epochs
    print(len(epochs))
    print(epochs.selection.shape)
    epochs.average().plot();
    # ------------------------------
    # 6 Reject bad epochs
    # ------------------------------

    epochs.drop_bad(
        reject=dict(eeg=CONFIG["reject_threshold"])
    )

    print(f"Remaining epochs: {len(epochs)}")
    # ERP
    # epochs.average().plot()
    # ------------------------------
    # 7 Save epochs
    # ------------------------------

    save_path = os.path.join(
        EPOCH_DIR, f"{subject}_epo.fif"
    )

    epochs.save(save_path, overwrite=True)

    print("Saved cleaned epochs")


# ==============================
# SINGLE MODE
# ==============================

def run_batch():

    eeg_files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))

    print(f"Found {len(eeg_files)} subjects")

    for file in eeg_files:
        preprocess_file(str(file))


# ==============================
# BATCH MODE
# ==============================


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

    MODE = "single"
    # change to "batch" later

    if MODE == "single":
        run_single()

    if MODE == "batch":
        run_batch()
