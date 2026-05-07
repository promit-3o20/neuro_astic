"""
ROI Feature Grouping Script

Author : Pramit Biswas
Version : v0.0

Description
-----------
Convert channel-wise bandpower features into 6 ROI-averaged features.

Input
-----
Channel-wise extracted bandpower parquet:
    early/late × band × channel

Output
------
ROI-averaged parquet:
    early/late × band × ROI

Notes
-----
- Keeps metadata columns unchanged
- Reduces dimensionality
- Keeps interpretation easier for ML/statistics
"""

from pathlib import Path
import glob
import logging
import os
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
LOG_DIR = (BASE_DIR / "../../logs/fetrs").resolve()

INPUT_DIR = DATA_DIR / "features/bp0"
OUTPUT_DIR = DATA_DIR / "features/roi_ftrs"

for d in [OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOGGER
# =========================================================
def setup_logger(log_file: str, level=logging.INFO):
    """
    Initialize and configure logger.
    """
    logger = logging.getLogger("ROIFeatureLogger")
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


logger = setup_logger(str(LOG_DIR / "roi_features.log"))


# =========================================================
# ROI DEFINITIONS
# =========================================================
ROI_MAP = {
    "CL1_lf": ["Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7"],
    "CL2_lft": ["FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "C1",  "CP1", "CP3", "CP5", "TP7"],
    "CL3_lpo": ["P7", "P5", "P3", "P1", "PO7", "PO3", "O1"],

    "CL4_rf": ["Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8"],
    "CL5_rft": ["FT8", "FC6", "FC4", "FC2", "T8", "C6", "C4", "C2", "CP2", "CP4", "CP6", "TP8"],
    "CL6_rpo": ["P8", "P6", "P4", "P2", "PO8", "PO4", "O2"],
}

TIME_WINDOWS = ["early", "late"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


# =========================================================
# CORE
# =========================================================
def group_roi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert channel-wise bandpower features into ROI-wise averaged features.

    Parameters
    ----------
    df : pd.DataFrame
        Channel-wise feature dataframe.

    Returns
    -------
    pd.DataFrame
        ROI-averaged feature dataframe with metadata preserved.
    """

    roi_features = {}

    feature_prefixes = tuple(f"{t}_{b}_" for t in TIME_WINDOWS for b in BANDS)
    metadata_cols = [c for c in df.columns if not c.startswith(feature_prefixes)]

    for time in TIME_WINDOWS:
        for band in BANDS:
            for roi_name, channels in ROI_MAP.items():
                cols = [f"{time}_{band}_{ch}" for ch in channels if f"{time}_{band}_{ch}" in df.columns]

                if len(cols) == 0:
                    logger.warning(f"Missing channels for {time}_{band}_{roi_name}")
                    continue

                roi_col = f"{time}_{band}_{roi_name}"
                roi_features[roi_col] = df[cols].mean(axis=1)

    roi_df = pd.DataFrame(roi_features)
    roi_df = pd.concat([roi_df, df[metadata_cols].reset_index(drop=True)], axis=1)

    return roi_df


# =========================================================
# IO
# =========================================================
def process_roi_file(input_path: str, output_path: str):
    """
    Load channel-wise features, group into ROI features, save parquet.
    """
    logger.info(f"Processing file: {input_path}")

    df = pd.read_parquet(input_path)
    logger.info(f"Input shape: {df.shape}")

    roi_df = group_roi_features(df)
    logger.info(f"Output shape: {roi_df.shape}")

    roi_df.to_parquet(output_path, compression="snappy")
    logger.info(f"Saved ROI features -> {output_path}")

def already_processed(subject: str, output_dir: str):
    """
    Check whether ROI feature file already exists.
    """

    output_file = os.path.join(
        output_dir,
        f"{subject}_roi_bpfeatures.parquet"
    )

    return os.path.exists(output_file)

def process_all_roi_files(input_dir: str, output_dir: str):
    """
    Batch process all subject feature parquet files into ROI features.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "sub-*_bpfeatures.parquet")))
    logger.info(f"Found {len(files)} feature files")

    all_dfs = []

    for file_path in files:

        subject = os.path.basename(file_path).replace(
            "_bpfeatures.parquet",
            ""
        )

        # ✅ Skip already processed subjects
        if already_processed(subject, output_dir):
            logger.info(f"Skipping {subject} (already processed)")
            continue

        logger.info(f"Processing subject: {subject}")

        try:
            df = pd.read_parquet(file_path)

            roi_df = group_roi_features(df)
            roi_df["subject"] = subject

            save_path = os.path.join(
                output_dir,
                f"{subject}_roi_bpfeatures.parquet"
            )

            roi_df.to_parquet(save_path, compression="snappy")

            logger.info(f"Saved ROI file -> {save_path}")

            all_dfs.append(roi_df)

        except Exception as e:
            logger.error(
                f"Error processing {subject}: {e}",
                exc_info=True
            )

    # Combine only newly processed files
    if all_dfs:

        final_df = pd.concat(all_dfs, ignore_index=True)

        final_path = os.path.join(
            output_dir,
            "all_subjects_roibpfeatures.parquet"
        )

        final_df.to_parquet(final_path, compression="snappy")

        logger.info(f"Saved combined ROI dataset -> {final_path}")
        logger.info(f"Final dataset shape: {final_df.shape}")


# =========================================================
# MAIN
# =========================================================
def main():
    DEBUG = False

    debug_in = INPUT_DIR / "sub-021_bpfeatures.parquet"
    debug_out = OUTPUT_DIR / "sub-021_roi_bpfeatures.parquet"

    if DEBUG:
        process_roi_file(str(debug_in), str(debug_out))
    else:
        process_all_roi_files(str(INPUT_DIR), str(OUTPUT_DIR))


if __name__ == "__main__":
    main()