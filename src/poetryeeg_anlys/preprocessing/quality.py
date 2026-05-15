"""
preprocessing/quality.py
========================
Epoch quality control: peak-to-peak amplitude rejection (Step 9).
"""

from __future__ import annotations

import logging

import mne
import numpy as np

from poetryeeg_anlys.config.constants import REJECT_THRESHOLD


def reject_epochs(
    epochs: mne.Epochs,
    logger: logging.Logger,
    label: str = "EPOCHS",
) -> tuple[mne.Epochs, mne.Epochs, list[int], list[int]]:
    """
    Reject epochs whose peak-to-peak amplitude exceeds 200 µV.

    Parameters
    ----------
    epochs : mne.Epochs
    logger : logging.Logger
    label  : str
        Label for log messages (e.g. ``'STIM'``, ``'START REST'``).

    Returns
    -------
    epochs_good : mne.Epochs
    epochs_bad  : mne.Epochs
    good_idx    : list[int]
    bad_idx     : list[int]
    """
    reject_criteria = {"eeg": REJECT_THRESHOLD}
    before       = len(epochs)
    original_idx = np.arange(before)

    epochs_copy = epochs.copy()
    epochs_copy.drop_bad(reject=reject_criteria)

    drop_log = epochs_copy.drop_log[:before]
    good_mask = np.array([len(log) == 0 for log in drop_log])
    bad_mask  = ~good_mask

    good_idx = original_idx[good_mask].tolist()
    bad_idx  = original_idx[bad_mask].tolist()

    logger.info(f"{label} | before={before}  good={len(good_idx)}  bad={len(bad_idx)}")
    logger.info(f"{label} | first 10 good_idx: {good_idx[:10]}")
    logger.info(f"{label} | first 10  bad_idx: {bad_idx[:10]}")

    epochs_good = epochs[good_idx]
    epochs_bad  = epochs[bad_idx]

    return epochs_good, epochs_bad, good_idx, bad_idx
