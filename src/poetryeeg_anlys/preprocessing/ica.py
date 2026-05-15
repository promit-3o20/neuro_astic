"""
preprocessing/ica.py
====================
ICA-based EOG artifact removal (Step 5 of the pipeline).

Configuration mirrors the original EEGLAB preprocessing:
- 64 components (full-rank on 64-channel scalp data, no PCA reduction)
- Extended Infomax (``method='infomax'``, ``extended=True``)
- EOG detection via EXG3/4/7/8; falls back to Fp1/Fp2 if those are absent
"""

from __future__ import annotations

import logging

import mne

from poetryeeg_anlys.config.constants import (
    ICA_N_COMPONENTS, ICA_METHOD, ICA_RANDOM_STATE,
    EOG_CHANNELS, EXG_ALL,
)


def run_ica(raw: mne.io.Raw, logger: logging.Logger) -> mne.io.Raw:
    """
    Fit ICA, identify EOG components, and return a cleaned Raw copy.

    ICA is fitted on scalp EEG channels only (``picks='eeg'``).
    EOG component detection uses ``find_bads_eog`` with EXG3/4/7/8;
    because those channels are still present at this stage the detection
    is equivalent to the original manual inspection approach.

    After ICA application all EXG channels are dropped so that only the
    64 BioSemi scalp channels remain.

    Parameters
    ----------
    raw    : mne.io.Raw
        Raw object after re-referencing and filtering.  EXG channels
        must still be present.
    logger : logging.Logger

    Returns
    -------
    mne.io.Raw
        ICA-cleaned, 64-channel scalp-only Raw object.
    """
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        fit_params={"extended": True},
        random_state=ICA_RANDOM_STATE,
    )

    ica.fit(raw, picks="eeg", decim=3)
    logger.info(f"ICA fitted: {ica.n_components_} components")

    # EOG artifact detection
    eog_inds, _ = ica.find_bads_eog(raw, ch_name=EOG_CHANNELS)
    ica.exclude = eog_inds
    logger.info(f"ICA excluded EOG components: {eog_inds}")

    raw_clean = ica.apply(raw.copy())

    # Drop all EXG channels to retain exactly 64 scalp EEG channels
    existing_exg = [ch for ch in EXG_ALL if ch in raw_clean.ch_names]
    raw_clean.drop_channels(existing_exg)
    logger.info(
        f"Dropped EXG channels after ICA: {existing_exg}. "
        f"Remaining channels: {len(raw_clean.ch_names)}"
    )

    return raw_clean
