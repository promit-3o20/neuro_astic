"""
pipelines/preprocessing_pipeline.py
====================================
Orchestrator for the full preprocessing pipeline (Steps 1–11).

Imports all logic from sub-packages; contains no signal-processing code.
"""

from __future__ import annotations

import numpy as np

from poetryeeg_anlys.config.paths import Paths
from poetryeeg_anlys.config.constants import REST_START, REST_END
from poetryeeg_anlys.config.settings import Settings
from poetryeeg_anlys.utils.logger import get_logger
from poetryeeg_anlys.utils.io import save_raw, save_epochs
from poetryeeg_anlys.utils.helpers import (
    subject_id_from_path, find_set_files, check_resume,
)
from poetryeeg_anlys.preprocessing import (
    read_eeg, set_channels,
    rereference_eeg, filter_eeg,
    run_ica,
    extract_events, epoch_stimulus, epoch_rest,
    reject_epochs,
)


# ---------------------------------------------------------------------------
# Resume guard
# ---------------------------------------------------------------------------

def already_processed(subject_id: str) -> bool:
    """Return True if all required preprocessing outputs already exist."""
    return check_resume(subject_id, [
        Paths.FILTER_DIR    / f"{subject_id}_filtered_raw.fif",
        Paths.ICA_DIR       / f"{subject_id}_ica_raw.fif",
        Paths.WHOLE_STIM_EPO / f"{subject_id}_allstim_epo.fif",
    ])


# ---------------------------------------------------------------------------
# Single-subject pipeline
# ---------------------------------------------------------------------------

def run_subject(file_path: str) -> None:
    """
    Execute the complete preprocessing pipeline for one subject.

    Pipeline order
    --------------
    1.  Read raw EEGLAB data.
    2.  Set channel types & BioSemi 64 montage.
    3.  Re-reference to average of EXG1 & EXG2 (earlobes).
    4.  Notch + bandpass filtering.
    5.  ICA (64 components, Extended Infomax); drop EXG channels.
    6.  Extract events.
    7.  Epoch stimulus trials (marker 65282); remove 2 practice trials.
    8.  Save all 210 epochs (whole_stim).
    [9–11 optional — toggle in Settings]
    9.  Reject bad epochs; save good / bad + index arrays.
    10. Epoch start-rest & end-rest intervals (4 s windows).
    11. Reject bad rest epochs; save.

    Parameters
    ----------
    file_path : str
        Path to the subject's ``.set`` file.
    """
    subject_id = subject_id_from_path(file_path)
    logger = get_logger(subject_id, log_dir=Paths.LOG_PREPROCESS)

    try:
        logger.info(f"=== Processing {subject_id} ===")

        # Steps 1–2
        raw = read_eeg(file_path)
        raw = set_channels(raw)

        # Step 3 – re-reference
        raw = rereference_eeg(raw, logger)

        # Step 4 – filter
        raw = filter_eeg(raw)
        save_raw(raw, Paths.FILTER_DIR / f"{subject_id}_filtered_raw.fif")

        # Step 5 – ICA + drop EXG
        raw = run_ica(raw, logger)
        save_raw(raw, Paths.ICA_DIR / f"{subject_id}_ica_raw.fif")

        # Step 6 – events
        events, event_id = extract_events(raw, logger)

        # Step 7 – stimulus epoching (210 trials)
        stim_epochs = epoch_stimulus(raw, events, event_id, logger)

        # Step 8 – save whole-stimulus epochs
        save_epochs(
            stim_epochs,
            Paths.WHOLE_STIM_EPO / f"{subject_id}_allstim_epo.fif",
        )

        # Step 9 – bad-epoch rejection (optional)
        # if Settings.ENABLE_EPOCH_REJECTION:
        #     stim_good, stim_bad, good_idx, bad_idx = reject_epochs(
        #         stim_epochs, logger, "STIM"
        #     )
        #     save_epochs(stim_good, Paths.G_STIM / f"{subject_id}_stim_epo.fif")
        #     save_epochs(stim_bad,  Paths.B_STIM / f"{subject_id}_badstim_epo.fif")
        #     np.savez(
        #         Paths.INDICES_STIM / f"{subject_id}_indices.npz",
        #         good_idx=good_idx,
        #         bad_idx=bad_idx,
        #     )

        # Steps 10–11 – rest epoching (optional)
        # if Settings.ENABLE_REST_EPOCHING:
        #     for rest_pair, label, out_dir in [
        #         (REST_START, "START", Paths.START_REST_EPO),
        #         (REST_END,   "END",   Paths.END_REST_EPO),
        #     ]:
        #         rest_epo = epoch_rest(raw, events, event_id, rest_pair, label, logger)
        #         if rest_epo is not None:
        #             good, _, _, _ = reject_epochs(rest_epo, logger, f"{label} REST")
        #             suffix = "strt" if label == "START" else "end"
        #             save_epochs(good, out_dir / f"{subject_id}_{suffix}_rest_epo.fif")

        logger.info(f"=== {subject_id} completed successfully ===")

    except Exception as exc:
        logger.error(f"Error processing {subject_id}: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def run_preprocessing(debug: bool = False) -> None:
    """
    Batch-preprocess all subjects in the raw data directory.

    Parameters
    ----------
    debug : bool
        If True (or ``Settings.DEBUG``), process only the first subject.
    """
    Paths.make_all()
    logger = get_logger("preprocessing", log_dir=Paths.LOG_PREPROCESS)

    files = find_set_files(Paths.RAW_DIR)
    logger.info(f"Found {len(files)} .set file(s)")

    if not files:
        logger.warning("No .set files found. Check Paths.RAW_DIR.")
        return

    if debug or Settings.DEBUG:
        files = [files[0]]
        logger.info(f"DEBUG mode — processing single file: {files[0]}")

    for file in files:
        sid = subject_id_from_path(file)
        if already_processed(sid):
            logger.info(f"Skipping {sid} (already processed)")
            continue
        logger.info(f"Processing {sid} …")
        run_subject(str(file))


def main() -> None:
    """CLI entry point."""
    run_preprocessing()


if __name__ == "__main__":
    main()
