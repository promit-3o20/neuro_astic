"""
preprocessing/filtering.py
===========================
Re-referencing and spectral filtering (Step 3–4 of the pipeline).

Step 3 — Re-reference to the average of EXG1 & EXG2 (earlobe electrodes).
Step 4 — Notch filter at 50 & 100 Hz; bandpass filter at 0.5–48 Hz.
"""

from __future__ import annotations

import logging

import mne

from poetryeeg_anlys.config.constants import (
    EARLOBE_CHANNELS, EXG_ALL, NOTCH_FREQS, BANDPASS,
)


def rereference_eeg(raw: mne.io.Raw, logger: logging.Logger) -> mne.io.Raw:
    """
    Re-reference all scalp EEG channels to the average of EXG1 & EXG2.

    EXG channels are **not** dropped here; they are removed after ICA so that
    ``find_bads_eog`` can still access EXG3/4/7/8 during artifact detection.

    Parameters
    ----------
    raw    : mne.io.Raw  (after :func:`set_channels`)
    logger : logging.Logger

    Returns
    -------
    mne.io.Raw
        Re-referenced Raw object (EXG channels still present).
    """
    logger.info(f"Re-referencing to average of {EARLOBE_CHANNELS}")
    raw.set_eeg_reference(ref_channels=EARLOBE_CHANNELS, projection=False)
    return raw


def filter_eeg(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply notch filter (50, 100 Hz) then bandpass filter (0.5–48 Hz).

    Parameters
    ----------
    raw : mne.io.Raw

    Returns
    -------
    mne.io.Raw
        Filtered Raw object (in-place modification; same object returned).
    """
    raw.notch_filter(NOTCH_FREQS)
    raw.filter(BANDPASS[0], BANDPASS[1])
    return raw
