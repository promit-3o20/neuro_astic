"""
preprocessing/epoching.py
==========================
Event extraction and epoch creation (Steps 6–8 of the pipeline).

Step 6 — Extract events from annotations.
Step 7 — Epoch stimulus trials (marker 65282); remove 2 practice trials.
Step 8 — (Optional) Epoch rest intervals into non-overlapping 4-second windows.
"""

from __future__ import annotations

import logging
from typing import Optional

import mne
import numpy as np
import pandas as pd

from poetryeeg_anlys.config.constants import (
    EPOCH_TMIN, EPOCH_TMAX, BASELINE,
    N_PRACTICE_TRIALS, REST_WINDOW,
)
from poetryeeg_anlys.utils.validation import require_marker


def extract_events(
    raw: mne.io.Raw,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict]:
    """
    Extract events from MNE annotations and log the distribution.

    Parameters
    ----------
    raw    : mne.io.Raw
    logger : logging.Logger

    Returns
    -------
    events : np.ndarray, shape (n_events, 3)
    event_id : dict  {label: code}
    """
    events, event_id = mne.events_from_annotations(raw)

    logger.info(f"Total events found: {len(events)}")
    logger.info(f"Event ID mapping: {event_id}")

    unique, counts = np.unique(events[:, 2], return_counts=True)
    logger.info(f"Event distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return events, event_id


def epoch_stimulus(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    logger: logging.Logger,
) -> mne.Epochs:
    """
    Create one epoch per stimulus trial, anchored on marker 65282.

    Steps
    -----
    1. Select events with code 65282.
    2. Epoch from ``EPOCH_TMIN`` to ``EPOCH_TMAX`` with baseline correction.
    3. Drop the first two practice trials.
    4. Attach metadata with a 0-indexed ``trial_index`` column.

    Parameters
    ----------
    raw      : mne.io.Raw
    events   : np.ndarray
    event_id : dict
    logger   : logging.Logger

    Returns
    -------
    mne.Epochs
        210 stimulus epochs (practice trials removed).
    """
    require_marker(event_id, "65282")

    stim_code   = event_id["65282"]
    stim_events = events[events[:, 2] == stim_code]
    logger.info(f"Stim events found (including practice): {len(stim_events)}")

    epochs = mne.Epochs(
        raw,
        stim_events,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        preload=True,
    )

    # Remove practice trials
    epochs = epochs[N_PRACTICE_TRIALS:]
    logger.info(
        f"After removing {N_PRACTICE_TRIALS} practice trials: "
        f"{len(epochs)} epochs (expected 210)"
    )

    # Clean 0-based trial index for downstream labelling
    epochs.metadata = pd.DataFrame({"trial_index": np.arange(len(epochs))})

    return epochs


def epoch_rest(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    rest_pair: tuple[int, int],
    label: str,
    logger: logging.Logger,
) -> Optional[mne.Epochs]:
    """
    Segment a rest interval into non-overlapping 4-second windows.

    Parameters
    ----------
    raw       : mne.io.Raw
    events    : np.ndarray
    event_id  : dict
    rest_pair : (start_code, end_code)
    label     : str
        Human-readable label for log messages (e.g. ``'START'``).
    logger    : logging.Logger

    Returns
    -------
    mne.Epochs or None
        ``None`` if the start / end markers are absent.
    """
    start_str, end_str = str(rest_pair[0]), str(rest_pair[1])

    if start_str not in event_id or end_str not in event_id:
        logger.warning(f"{label} rest events missing — skipping.")
        return None

    start_code   = event_id[start_str]
    end_code     = event_id[end_str]
    start_events = events[events[:, 2] == start_code]
    end_events   = events[events[:, 2] == end_code]

    if len(start_events) == 0 or len(end_events) == 0:
        logger.warning(f"{label} rest event arrays empty — skipping.")
        return None

    sfreq    = raw.info["sfreq"]
    start_t  = start_events[0][0] / sfreq
    end_t    = end_events[0][0] / sfreq
    duration = end_t - start_t
    n_segs   = int(duration // REST_WINDOW)

    logger.info(
        f"{label} rest: {duration:.2f} s → "
        f"{n_segs} × {REST_WINDOW} s segments"
    )

    synthetic_events = np.array(
        [
            [int((start_t + i * REST_WINDOW) * sfreq), 0, 999]
            for i in range(n_segs)
        ]
    )

    return mne.Epochs(
        raw,
        synthetic_events,
        tmin=0,
        tmax=REST_WINDOW,
        baseline=None,
        preload=True,
    )
