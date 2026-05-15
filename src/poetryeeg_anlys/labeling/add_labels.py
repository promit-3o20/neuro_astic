"""
labeling/add_labels.py
======================
Attach behavioral ratings to EEG epoch metadata and encode categorical labels.

Refactored from ``add_label.py`` — all logic preserved exactly.

Pipeline (per subject)
----------------------
1. Load participants mapping.
2. Load behavioral CSV.
3. Align epochs ↔ behavioral trials via ``trial_index`` metadata.
4. Attach label columns to ``epochs.metadata``.
5. Encode categorical columns (PoemType, Block) with :class:`LabelEncoder`.
6. Save labeled epochs as ``.fif`` (raw_label/).
7. Optionally flatten and save ML-ready Parquet (labeled/).
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from poetryeeg_anlys.config.constants import BEHAV_COLUMNS, CATEGORICAL_COLS
from poetryeeg_anlys.config.paths import Paths
from poetryeeg_anlys.utils.io import load_epochs, save_epochs, save_parquet
from poetryeeg_anlys.utils.logger import get_logger
from poetryeeg_anlys.utils.validation import require_columns
from poetryeeg_anlys.utils.helpers import check_resume
from .behavior import load_participants, create_subject_mapping, load_behavior


# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------

def attach_labels(epochs: mne.Epochs, behav_df: pd.DataFrame) -> mne.Epochs:
    """
    Align behavioral ratings with EEG epochs and attach them to
    ``epochs.metadata``.

    Alignment uses the ``trial_index`` column in ``epochs.metadata``,
    which must be a 0-based integer index into *behav_df*.

    Parameters
    ----------
    epochs   : mne.Epochs  (must have a ``trial_index`` metadata column)
    behav_df : pd.DataFrame  (210-row behavioral table)

    Returns
    -------
    mne.Epochs
        Same object with enriched metadata.

    Raises
    ------
    ValueError
        If any required column is absent from *behav_df*.
    """
    require_columns(behav_df, BEHAV_COLUMNS, source="behavioral file")

    trial_idx = epochs.metadata["trial_index"].values

    for col in BEHAV_COLUMNS:
        epochs.metadata[col] = behav_df.iloc[trial_idx][col].values

    return epochs


def encode_labels(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode categorical columns (PoemType, Block) into integer codes.

    Parameters
    ----------
    metadata : pd.DataFrame

    Returns
    -------
    metadata_enc : pd.DataFrame
        Copy of *metadata* with additional ``{col}_enc`` columns.
    encoders : dict[str, LabelEncoder]
        Fitted encoders keyed by column name.
    """
    metadata = metadata.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        metadata[f"{col}_enc"] = le.fit_transform(metadata[col])
        encoders[col] = le

    return metadata, encoders


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_labeled_fif(
    epochs: mne.Epochs,
    subject_id: str,
    suffix: str = "whole",
) -> None:
    """Save labeled epochs to ``raw_label/{subject_id}_{suffix}_epo.fif``."""
    out_path = Paths.RAW_LABEL_DIR / f"{subject_id}_{suffix}_epo.fif"
    save_epochs(epochs, out_path)


def epochs_to_parquet(
    epochs: mne.Epochs,
    subject_id: str,
    suffix: str = "whole",
) -> None:
    """
    Flatten epoch data + metadata and save as a ML-ready Parquet file.

    Column layout: [EEG flat features] + [metadata columns (encoded)].
    Output path: ``labeled/{subject_id}_{suffix}.parquet``.

    Parameters
    ----------
    epochs     : mne.Epochs
    subject_id : str
    suffix     : str
    """
    data = epochs.get_data()                          # (n_ep, n_ch, n_t)
    n_epochs, n_ch, n_t = data.shape
    X_df = pd.DataFrame(data.reshape(n_epochs, n_ch * n_t))

    metadata_enc, _ = encode_labels(epochs.metadata.reset_index(drop=True))
    df = pd.concat([X_df, metadata_enc], axis=1)

    out_path = Paths.LABELED_DIR / f"{subject_id}_{suffix}.parquet"
    save_parquet(df, out_path)


# ---------------------------------------------------------------------------
# Per-subject orchestrator
# ---------------------------------------------------------------------------

def process_subject(
    subject_id: str,
    mapping: dict[str, str],
    logger: logging.Logger | None = None,
) -> None:
    """
    Run the full labeling pipeline for a single subject.

    Parameters
    ----------
    subject_id : str
    mapping    : dict[str, str]  (output of :func:`create_subject_mapping`)
    logger     : logging.Logger, optional
    """
    if logger is None:
        logger = get_logger(subject_id, log_dir=Paths.LOG_LABELING)

    whole_file = Paths.WHOLE_STIM_EPO / f"{subject_id}_allstim_epo.fif"

    if not whole_file.exists():
        logger.warning(f"Missing epoch file: {whole_file} — skipping.")
        return

    logger.info(f"Loading epochs: {whole_file}")
    whole_epochs = load_epochs(whole_file)

    behav_df = load_behavior(subject_id, mapping)

    whole_epochs = attach_labels(whole_epochs, behav_df)
    save_labeled_fif(whole_epochs, subject_id, suffix="whole")

    # Uncomment to also save ML-ready Parquet:
    # epochs_to_parquet(whole_epochs, subject_id, suffix="whole")

    logger.info(f"{subject_id} labeling complete.")


def already_processed(subject_id: str) -> bool:
    """Return True if the labeled .fif output already exists (resume-safe)."""
    return check_resume(
        subject_id,
        [Paths.RAW_LABEL_DIR / f"{subject_id}_whole_epo.fif"],
    )


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def run_labeling(logger: logging.Logger | None = None) -> None:
    """
    Batch-label all subjects found in ``WHOLE_STIM_EPO``.

    Resume-safe: subjects whose output .fif already exists are skipped.
    """
    if logger is None:
        logger = get_logger("labeling", log_dir=Paths.LOG_LABELING)

    participants_df = load_participants()
    mapping = create_subject_mapping(participants_df)

    files = list(Paths.WHOLE_STIM_EPO.glob("*_allstim_epo.fif"))
    subject_ids = [f.stem.split("_")[0] for f in files]
    logger.info(f"Found {len(subject_ids)} subjects")

    for subject_id in subject_ids:
        if already_processed(subject_id):
            logger.info(f"Skipping {subject_id} (already processed)")
            continue
        try:
            subject_logger = get_logger(subject_id, log_dir=Paths.LOG_LABELING)
            process_subject(subject_id, mapping, logger=subject_logger)
        except Exception as exc:
            logger.error(f"Error labeling {subject_id}: {exc}", exc_info=True)


def main() -> None:
    """CLI entry point — run batch labeling."""
    Paths.make_all()
    run_labeling()


if __name__ == "__main__":
    main()
