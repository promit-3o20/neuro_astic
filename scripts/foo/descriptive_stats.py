"""
ROI Descriptive Statistics Script

Author : Pramit Biswas
Version : v0.0

Description
-----------
Compute subject-wise descriptive statistics from ROI-averaged bandpower features.

Input
-----
ROI feature parquet:
    early/late × band × ROI + metadata + behavioral ratings

Output
------
Subject-wise descriptive statistics CSV:
    descriptive stats for EEG ROI features only

Notes
-----
- Splits EEG features, metadata, and behavioral ratings first
- Computes descriptive statistics only on EEG ROI features
- Saves one descriptive CSV per subject
"""

from pathlib import Path
import glob
import logging
import os
import pandas as pd


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
LOG_DIR = (BASE_DIR / "../../logs/fetrs").resolve()

INPUT_DIR = DATA_DIR / "features/roi_ftrs"
OUTPUT_DIR = BASE_DIR / "../../results/descriptive"

for d in [OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOGGER
# =========================================================
def setup_logger(log_file: str, level=logging.INFO):
    """
    Initialize and configure logger.
    """
    logger = logging.getLogger("ROIDescriptiveLogger")
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


logger = setup_logger(str(LOG_DIR / "roi_descriptive.log"))


# =========================================================
# COLUMN GROUPS
# =========================================================
METADATA_COLS = ["trial_index", "PoemType", "Block", "subject"]
BEHAVIOR_COLS = ["AA", "Imagery", "Moved", "Originality", "Creativity"]


# =========================================================
# CORE
# =========================================================
def split_columns(df: pd.DataFrame):
    """
    Split dataframe into EEG features, metadata, and behavioral ratings.
    """
    metadata_cols = [c for c in METADATA_COLS if c in df.columns]
    behavior_cols = [c for c in BEHAVIOR_COLS if c in df.columns]

    metadata_df = df[metadata_cols].copy()
    behavior_df = df[behavior_cols].copy()

    feature_cols = [c for c in df.columns if c not in metadata_cols + behavior_cols]
    feature_df = df[feature_cols].copy()

    return feature_df, metadata_df, behavior_df


def compute_descriptive_stats(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for EEG ROI feature columns.
    """
    desc = feature_df.describe().T

    desc["median"] = feature_df.median()
    desc["variance"] = feature_df.var()
    desc["skew"] = feature_df.skew()
    desc["kurtosis"] = feature_df.kurtosis()
    desc["missing"] = feature_df.isna().sum()
    desc["missing_pct"] = (feature_df.isna().mean() * 100).round(2)

    desc = desc[
        [
            "count", "missing", "missing_pct",
            "mean", "std", "variance",
            "min", "25%", "50%", "75%", "max",
            "median", "skew", "kurtosis"
        ]
    ]

    desc.index.name = "feature"
    return desc.reset_index()


# =========================================================
# IO
# =========================================================
def process_subject_file(input_path: str, output_path: str):
    """
    Load ROI feature parquet, compute descriptive stats, save CSV.
    """
    logger.info(f"Processing file: {input_path}")

    df = pd.read_parquet(input_path)
    logger.info(f"Input shape: {df.shape}")

    feature_df, metadata_df, behavior_df = split_columns(df)
    desc_df = compute_descriptive_stats(feature_df)

    logger.info(f"Descriptive stats shape: {desc_df.shape}")

    desc_df.to_csv(output_path, index=False)
    logger.info(f"Saved descriptive stats -> {output_path}")


def process_all_subjects(input_dir: str, output_dir: str):
    """
    Batch process all ROI subject parquet files into descriptive stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "sub-*_roi_bpfeatures.parquet")))
    logger.info(f"Found {len(files)} ROI feature files")

    for file_path in files:
        subject = os.path.basename(file_path).replace("_roi_bpfeatures.parquet", "")
        logger.info(f"Processing subject: {subject}")

        try:
            save_path = os.path.join(output_dir, f"{subject}_descriptive_stats.csv")
            process_subject_file(file_path, save_path)

        except Exception as e:
            logger.error(f"Error processing {subject}: {e}", exc_info=True)


# =========================================================
# MAIN
# =========================================================
def main():
    DEBUG = True

    debug_in = INPUT_DIR / "sub-021_roi_bpfeatures.parquet"
    debug_out = OUTPUT_DIR / "sub-021_descriptive_stats.csv"

    if DEBUG:
        process_subject_file(str(debug_in), str(debug_out))
    else:
        process_all_subjects(str(INPUT_DIR), str(OUTPUT_DIR))


if __name__ == "__main__":
    main()