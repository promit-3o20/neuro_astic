"""
This script is used to label EEG epochs with their corresponding behavioral ratings.

Author : Pramit Biswas
Version : v0.0

Pipeline
------------------------------
1. Initialize project paths and create output directories.
2. Load participant mapping from participants.tsv.
3. Match each EEG subject ID with its corresponding behavioral rating file.
4. Load behavioral ratings and validate trial consistency (210 trials).
5. Load preprocessed EEG epoch files (whole stimulus and good stimulus).
6. Align EEG epochs with behavioral trials using trial_index metadata.
7. Attach behavioral labels (PoemType, Block, AA, Imagery, Moved,
   Originality, Creativity) to epoch metadata.
8. Encode categorical labels (PoemType, Block) into numerical format.
9. Save labeled EEG epochs in FIF format for EEG analysis.
10. Flatten epoch-wise EEG signals and combine them with metadata.
11. Save machine-learning-ready labeled datasets in Parquet format.
12. Repeat the process for all available subjects.

"""

import mne
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

EPOCH_DIR = DATA_DIR / "intrmd_data/epochs"
WHOLE_STIM = EPOCH_DIR / "stim/whole_stim"
GOOD_STIM = EPOCH_DIR / "stim/g_stim"

BEHAV_DIR = DATA_DIR / "raw_data/ds006648-download/derivatives/Behavioural_Ratings"
PARTICIPANTS_FILE = DATA_DIR / "raw_data/ds006648-download/participants.tsv"

RAW_LABEL_DIR = DATA_DIR / "intrmd_data/raw_label"
LABELED_DIR = DATA_DIR / "intrmd_data/labeled"

RAW_LABEL_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD PARTICIPANTS TSV
# ==============================


def load_participants():
    df = pd.read_csv(PARTICIPANTS_FILE, sep="\t", header=None)
    df.columns = ["subject_id", "behav_id"]
    return df


def create_subject_mapping(df):
    return dict(zip(df["subject_id"], df["behav_id"]))


# ==============================
# LOAD BEHAVIOR FILE
# ==============================


def get_behavior_file(subject_id, mapping):
    if subject_id not in mapping:
        raise ValueError(f"{subject_id} not found in participants.tsv")

    behav_id = mapping[subject_id]  # e.g., P101
    behav_file = BEHAV_DIR / f"{behav_id}.csv"

    return behav_file


def load_behavior(subject_id, mapping):
    behav_file = get_behavior_file(subject_id, mapping)

    print(f"Loading behavior: {behav_file}")

    if not behav_file.exists():
        raise FileNotFoundError(f"{behav_file} not found")

    try:
        df = pd.read_csv(behav_file)
    except UnicodeDecodeError:
        print("UTF-8 failed, trying latin1...")
        df = pd.read_csv(behav_file, encoding="latin1")

    if len(df) != 210:
        raise ValueError(f"{subject_id}: Expected 210 trials, got {len(df)}")

    return df


# ==============================
# ATTACH LABELS
# ==============================


def attach_labels(epochs, behav_df):
    trial_idx = epochs.metadata["trial_index"].values

    columns = [
        "PoemType",
        "Block",
        "AA",
        "Imagery",
        "Moved",
        "Originality",
        "Creativity",
    ]

    for col in columns:
        if col not in behav_df.columns:
            raise ValueError(f"{col} not found in behavioral file")

        epochs.metadata[col] = behav_df.iloc[trial_idx][col].values

    return epochs


# ==============================
# ENCODE LABELS
# ==============================


def encode_labels(metadata):
    metadata = metadata.copy()
    encoders = {}

    categorical_cols = ["PoemType", "Block"]

    for col in categorical_cols:
        le = LabelEncoder()
        metadata[col + "_enc"] = le.fit_transform(metadata[col])
        encoders[col] = le

    return metadata, encoders


# ==============================
# SAVE LABELED FIF
# ==============================


def save_labeled_fif(epochs, subject_id, suffix):
    out_file = RAW_LABEL_DIR / f"{subject_id}_{suffix}_epo.fif"
    epochs.save(out_file, overwrite=True)
    print(f"Saved FIF: {out_file}")


# ==============================
# SAVE PARQUET (ML READY)
# ==============================


def epochs_to_parquet(epochs, subject_id, suffix):
    data = epochs.get_data()
    n_epochs, n_ch, n_t = data.shape

    # Flatten EEG
    X = data.reshape(n_epochs, n_ch * n_t)
    X_df = pd.DataFrame(X)

    # Metadata
    metadata = epochs.metadata.reset_index(drop=True)

    # Encode labels
    metadata_enc, _ = encode_labels(metadata)

    # Combine
    df = pd.concat([X_df, metadata_enc], axis=1)

    out_file = LABELED_DIR / f"{subject_id}_{suffix}.parquet"
    df.to_parquet(out_file)

    print(f"Saved Parquet: {out_file}")


# ==============================
# PROCESS SUBJECT
# ==============================


def process_subject(subject_id, mapping):
    print(f"\nProcessing {subject_id}")

    whole_file = WHOLE_STIM / f"{subject_id}_allstim_epo.fif"
    good_file = GOOD_STIM / f"{subject_id}_stim_epo.fif"

    if not whole_file.exists():
        print(f"Missing: {whole_file}")
        return

    if not good_file.exists():
        print(f"Missing: {good_file}")
        return

    whole_epochs = mne.read_epochs(whole_file, preload=True)
    good_epochs = mne.read_epochs(good_file, preload=True)

    # Load behavior using TSV mapping
    behav_df = load_behavior(subject_id, mapping)

    # Attach labels
    whole_epochs = attach_labels(whole_epochs, behav_df)
    good_epochs = attach_labels(good_epochs, behav_df)

    # Save labeled FIF
    save_labeled_fif(whole_epochs, subject_id, "whole")
    save_labeled_fif(good_epochs, subject_id, "good")

    # Save parquet
    epochs_to_parquet(whole_epochs, subject_id, "whole")
    epochs_to_parquet(good_epochs, subject_id, "good")


# ==============================
# MAIN
# ==============================


def main():
    participants_df = load_participants()
    mapping = create_subject_mapping(participants_df)

    files = list(WHOLE_STIM.glob("*_allstim_epo.fif"))
    subject_ids = [f.stem.split("_")[0] for f in files]

    print(f"Found {len(subject_ids)} subjects")

    for subject_id in subject_ids:
        print(f"{subject_id} → {mapping.get(subject_id, 'NOT FOUND')}")
        process_subject(subject_id, mapping)


if __name__ == "__main__":
    main()
