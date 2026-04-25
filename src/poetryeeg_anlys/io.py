import pandas as pd
from .config import PARTICIPANTS_FILE, RAW_LABEL_DIR, LABELED_DIR


def save_raw(raw, path):
    raw.save(path, overwrite=True)


def save_epochs(epochs, path):
    if epochs is not None:
        epochs.save(path, overwrite=True)


def load_participants():
    df = pd.read_csv(PARTICIPANTS_FILE, sep="\t", header=None)
    df.columns = ["subject_id", "behav_id"]
    return df


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


def save_labeled_fif(epochs, subject_id, suffix):
    out_file = RAW_LABEL_DIR / f"{subject_id}_{suffix}_epo.fif"
    epochs.save(out_file, overwrite=True)
    print(f"Saved FIF: {out_file}")


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


def save_features(df: pd.DataFrame, path: str):
    """
    Save extracted features to a parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame containing extracted EEG features.
    path : str
        Output file path.

    Returns
    -------
    None

    Notes
    -----
    - Uses Snappy compression for fast read/write.
    - Parquet format is efficient for ML pipelines.
    - Overwrites file if it already exists.
    """
    df.to_parquet(path, compression="snappy")
    logger.info(f"Saved features → {path}")
