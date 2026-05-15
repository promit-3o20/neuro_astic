"""
labeling/behavior.py
====================
Participants mapping and behavioral rating file loading.

Refactored from ``add_label.py`` — all logic preserved exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from poetryeeg_anlys.config.paths import Paths
from poetryeeg_anlys.config.constants import N_EXPECTED_TRIALS
from poetryeeg_anlys.utils.validation import require_trial_count


def load_participants() -> pd.DataFrame:
    """
    Load ``participants.tsv`` and return a two-column DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``subject_id`` (e.g. ``'sub-021'``),
                 ``behav_id``  (e.g. ``'P101'``).
    """
    df = pd.read_csv(Paths.PARTICIPANTS, sep="\t", header=None)
    df.columns = ["subject_id", "behav_id"]
    return df


def create_subject_mapping(df: pd.DataFrame) -> dict[str, str]:
    """
    Build a ``{subject_id: behav_id}`` lookup from the participants DataFrame.

    Parameters
    ----------
    df : pd.DataFrame  (output of :func:`load_participants`)

    Returns
    -------
    dict[str, str]
    """
    return dict(zip(df["subject_id"], df["behav_id"]))


def get_behavior_file(subject_id: str, mapping: dict[str, str]) -> Path:
    """
    Resolve the behavioral CSV path for *subject_id*.

    Parameters
    ----------
    subject_id : str
    mapping    : dict[str, str]  (output of :func:`create_subject_mapping`)

    Returns
    -------
    Path

    Raises
    ------
    ValueError
        If *subject_id* is not in *mapping*.
    """
    if subject_id not in mapping:
        raise ValueError(f"{subject_id} not found in participants.tsv")
    behav_id = mapping[subject_id]
    return Paths.BEHAV_DIR / f"{behav_id}.csv"


def load_behavior(subject_id: str, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Load and validate the behavioral rating CSV for *subject_id*.

    Block column forward-fills missing values (blocks are labelled only on
    the first trial of each block in the raw CSV).

    Parameters
    ----------
    subject_id : str
    mapping    : dict[str, str]

    Returns
    -------
    pd.DataFrame
        210-row behavioral table.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist on disk.
    ValueError
        If the row count differs from 210.
    """
    behav_file = get_behavior_file(subject_id, mapping)

    if not behav_file.exists():
        raise FileNotFoundError(f"Behavioral file not found: {behav_file}")

    try:
        df = pd.read_csv(behav_file)
    except UnicodeDecodeError:
        df = pd.read_csv(behav_file, encoding="latin1")

    require_trial_count(df, N_EXPECTED_TRIALS, subject_id)

    # Forward-fill block labels (block name appears only on the first trial
    # of each block in the raw CSV)
    df["Block"] = df["Block"].ffill()

    return df
