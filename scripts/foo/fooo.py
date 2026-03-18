import os
from pathlib import Path
import mne
import numpy as np

# ==============================
# CONFIGURATION
# ==============================

CONFIG = {
    "sampling_rate": 512,
    "filter_type": "bandpass",
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
# UTILITY FUNCTIONS
# ==============================


def get_subject(filepath):
    return next(p for p in Path(filepath).parts if p.startswith("sub-"))


# ==============================
# VISUALIZATION FUNCTIONS
# ==============================


def plot_raw(raw, title="Raw Signal"):
    raw.plot(duration=10, scalings=dict(eeg=50e-6), title=title)


def plot_psd(raw, title="PSD"):
    raw.plot_psd(fmax=60, spatial_colors=True)


def plot_montage(raw):
    raw.plot_sensors(ch_type="eeg", kind="topomap", show_names=True, sphere=0.1)


def plot_clean_vs_raw(raw_before, raw_after):
    raw_before.plot(duration=10, title="Before Cleaning")
    raw_after.plot(duration=10, title="After Cleaning")


def plot_epochs(epochs):
    epochs.plot(n_epochs=10, n_channels=20, scalings="auto")
    epochs.plot_image(picks="eeg")
    epochs.average().plot_joint()
    epochs.average().plot_topomap(times=[0.1, 0.2, 0.3])


# ==============================
# PREPROCESSING STEPS
# ==============================


def load_raw(filepath):
    raw = mne.io.read_raw_eeglab(filepath, preload=True)

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


def apply_filter(raw):
    raw.notch_filter(freqs=[50, 100], notch_widths=4)

    if CONFIG["filter_type"] == "highpass":
        raw.filter(l_freq=CONFIG["highpass"], h_freq=None, picks="eeg")

    elif CONFIG["filter_type"] == "bandpass":
        raw.filter(
            l_freq=CONFIG["bandpass"][0], h_freq=CONFIG["bandpass"][1], picks="eeg"
        )

    return raw


def run_ica(raw):
    ica = mne.preprocessing.ICA(
        n_components=CONFIG["ica_components"], random_state=97, max_iter="auto"
    )

    ica.fit(raw, decim=3)

    eog_indices, _ = ica.find_bads_eog(raw)
    ica.exclude = eog_indices

    raw_clean = ica.apply(raw.copy())

    return raw_clean, ica


def extract_events(raw):
    events, event_id = mne.events_from_annotations(raw)
    return events, event_id


def create_epochs(raw, events, event_id):
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=CONFIG["epoch_window"][0],
        tmax=CONFIG["epoch_window"][1],
        baseline=CONFIG["baseline"],
        preload=True,
    )
    return epochs


def reject_bad_epochs(epochs):
    epochs.drop_bad(reject=dict(eeg=CONFIG["reject_threshold"]))
    return epochs


# ==============================
# MAIN PIPELINE
# ==============================


def preprocess_file(filepath):

    subject = get_subject(filepath)
    print(f"\nProcessing {subject}")

    # --------------------------
    # 1 Load
    # --------------------------
    raw = load_raw(filepath)

    plot_montage(raw)
    # plot_raw(raw)

    raw_before = raw.copy()

    # --------------------------
    # 2 Filtering
    # --------------------------
    raw = apply_filter(raw)
    plot_psd(raw)

    # Save filtered
    raw.save(FILTER_DIR / f"{subject}_filtered.fif", overwrite=True)

    # --------------------------
    # 3 ICA
    # --------------------------
    raw_clean, ica = run_ica(raw)

    print(f"Removed ICA components: {ica.exclude}")

    plot_clean_vs_raw(raw_before, raw_clean)

    # --------------------------
    # 4 Events
    # --------------------------
    events, event_id = extract_events(raw_clean)

    print(f"Found {len(events)} events")
    print("Event mapping:", event_id)

    mne.viz.plot_events(events)

    # --------------------------
    # 5 Epoching
    # --------------------------
    epochs = create_epochs(raw_clean, events, event_id)

    print(f"Total epochs: {len(epochs)}")

    plot_epochs(epochs)

    # --------------------------
    # 6 Reject bad epochs
    # --------------------------
    epochs = reject_bad_epochs(epochs)

    print(f"Remaining epochs: {len(epochs)}")

    # --------------------------
    # 7 Save
    # --------------------------
    save_path = EPOCH_DIR / f"{subject}_epo.fif"
    epochs.save(save_path, overwrite=True)

    print("Saved cleaned epochs")


# ==============================
# RUN MODES
# ==============================


def run_single():
    eeg_files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))

    if not eeg_files:
        print("No EEG files found")
        return

    preprocess_file(str(eeg_files[0]))


def run_batch():
    eeg_files = list(RAW_DIR.rglob("sub-*/eeg/*.set"))

    print(f"Found {len(eeg_files)} subjects")

    for file in eeg_files:
        preprocess_file(str(file))


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":

    MODE = "single"  # change to "batch"

    if MODE == "single":
        run_single()

    elif MODE == "batch":
        run_batch()
