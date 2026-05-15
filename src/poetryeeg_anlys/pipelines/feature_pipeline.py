"""
pipelines/feature_pipeline.py
==============================
Batch ROI band-power feature extraction orchestrator.

Imports all logic from :mod:`poetryeeg_anlys.features`; contains no
signal-processing code.
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from poetryeeg_anlys.config.paths import Paths
from poetryeeg_anlys.config.settings import Settings
from poetryeeg_anlys.utils.logger import get_logger
from poetryeeg_anlys.utils.io import load_epochs, save_parquet
from poetryeeg_anlys.utils.helpers import find_fif_files
from poetryeeg_anlys.features import extract_roi_bandpower_features
from poetryeeg_anlys.features.feature_utils import already_processed


def run_features(debug: bool = False) -> None:
    """
    Batch-extract ROI band-power features for all labeled subjects.

    Parameters
    ----------
    debug : bool
        If True (or ``Settings.DEBUG``), process only ``Settings.DEBUG_SUBJECT``.
    """
    Paths.make_all()
    logger = get_logger("features", log_dir=Paths.LOG_FEATURES)

    files = find_fif_files(Paths.RAW_LABEL_DIR, pattern="*_whole_epo.fif")
    logger.info(f"Found {len(files)} labeled epoch files")

    if debug or Settings.DEBUG:
        files = [
            f for f in files
            if Settings.DEBUG_SUBJECT in f.name
        ] or files[:1]
        logger.info(f"DEBUG mode — {[f.name for f in files]}")

    all_roi: list[pd.DataFrame] = []

    for fif_path in files:
        subject = fif_path.stem.split("_")[0]

        if already_processed(subject, Paths.ROI_BP_DIR):
            logger.info(f"Skipping {subject} (already processed)")
            continue

        logger.info(f"Extracting features: {subject}")
        try:
            epochs = load_epochs(fif_path)

            bp_df, roi_df = extract_roi_bandpower_features(
                epochs, logger=get_logger(subject, log_dir=Paths.LOG_FEATURES)
            )
            bp_df["subject"]  = subject
            roi_df["subject"] = subject

            save_parquet(
                bp_df,
                Paths.BP_DIR / f"{subject}_bpfeatures.parquet",
                logger=logger,
            )
            save_parquet(
                roi_df,
                Paths.ROI_BP_DIR / f"{subject}_roi_bpfeatures.parquet",
                logger=logger,
            )
            all_roi.append(roi_df)

        except Exception as exc:
            logger.error(f"Error extracting {subject}: {exc}", exc_info=True)

    if all_roi:
        combined = pd.concat(all_roi, ignore_index=True)
        save_parquet(
            combined,
            Paths.ROI_BP_DIR / "all_subjects_roi_bpfeatures.parquet",
            logger=logger,
        )
        logger.info(f"Combined dataset shape: {combined.shape}")

    logger.info("Feature extraction batch complete.")


def main() -> None:
    """CLI entry point."""
    run_features()


if __name__ == "__main__":
    main()
