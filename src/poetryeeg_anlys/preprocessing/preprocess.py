"""
preprocessing/preprocess.py
============================
Raw EEG loading and channel configuration.

Covers Steps 1–2 of the preprocessing pipeline:
    1. Load EEGLAB .set file into an MNE Raw object.
    2. Assign channel types and apply the BioSemi 64 montage.
"""

from __future__ import annotations

import mne

from poetryeeg_anlys.config.constants import (
    EARLOBE_CHANNELS, EOG_CHANNELS,
)


def read_eeg(file_path: str) -> mne.io.Raw:
    """
    Load an EEGLAB .set file into an MNE Raw object (preloaded).

    Parameters
    ----------
    file_path : str
        Absolute path to the ``.set`` file.

    Returns
    -------
    mne.io.Raw
    """
    return mne.io.read_raw_eeglab(file_path, preload=True)


def set_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Assign channel types for EXG channels and apply the BioSemi 64 montage.

    Channel assignment (BioSemi ActiveTwo protocol):

    - EXG1, EXG2  → typed ``'eeg'`` temporarily so MNE's
      ``set_eeg_reference`` can include them in the average; dropped later.
    - EXG3, EXG4, EXG7, EXG8  → vertical / horizontal EOG.
    - EXG5, EXG6  → ECG (recorded; not used in analysis).

    Parameters
    ----------
    raw : mne.io.Raw

    Returns
    -------
    mne.io.Raw
        Raw object with updated channel types and BioSemi 64 montage applied.
    """
    channel_types: dict[str, str] = {
        # Earlobes kept as 'eeg' for re-referencing
        "EXG1": "eeg",
        "EXG2": "eeg",
        # EOG channels
        "EXG3": "eog",
        "EXG4": "eog",
        "EXG7": "eog",
        "EXG8": "eog",
        # ECG channels (not analysed; leave as default misc / skip typing)
    }
    raw.set_channel_types(channel_types)
    raw.set_montage("biosemi64", on_missing="ignore")
    return raw
