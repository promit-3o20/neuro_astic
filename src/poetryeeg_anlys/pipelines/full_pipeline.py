"""
pipelines/full_pipeline.py
==========================
Master orchestrator — runs the entire analysis pipeline in order:

    1. Preprocessing  (raw → filtered → ICA → epochs)
    2. Labeling       (epochs + behavioral CSV → labeled epochs)
    3. Feature extraction  (labeled epochs → ROI band-power parquet)
    4. ML pipeline    (features → trained classifiers + evaluation)
    5. DL training    (epochs → EEGNet / CNN-LSTM)

Each stage is independently resume-safe; re-running the full pipeline
skips any subject / stage whose outputs already exist on disk.
"""

from __future__ import annotations

from poetryeeg_anlys.config.paths import Paths
from poetryeeg_anlys.config.settings import Settings
from poetryeeg_anlys.utils.logger import get_logger

from .preprocessing_pipeline import run_preprocessing
from .feature_pipeline import run_features
from .ml_pipeline import run_ml


def run_pipeline(debug: bool | None = None) -> None:
    """
    Execute all pipeline stages in sequence.

    Parameters
    ----------
    debug : bool, optional
        Override ``Settings.DEBUG``.  Pass ``True`` to run on a single
        subject end-to-end (useful for validation).
    """
    Paths.make_all()

    _debug = Settings.DEBUG if debug is None else debug
    logger = get_logger("full_pipeline", log_dir=Paths.LOG_PREPROCESS)

    logger.info("=" * 60)
    logger.info("Poetry-EEG Analysis Pipeline — START")
    logger.info(f"  DEBUG mode: {_debug}")
    logger.info("=" * 60)

    # ── Stage 1: Preprocessing ──────────────────────────────────────────
    logger.info("Stage 1 / 4 — Preprocessing")
    run_preprocessing(debug=_debug)

    # ── Stage 2: Labeling ───────────────────────────────────────────────
    logger.info("Stage 2 / 4 — Labeling")
    from poetryeeg_anlys.labeling.add_labels import run_labeling
    run_labeling()

    # ── Stage 3: Feature extraction ─────────────────────────────────────
    logger.info("Stage 3 / 4 — Feature Extraction")
    run_features(debug=_debug)

    # ── Stage 4: ML pipeline ────────────────────────────────────────────
    logger.info("Stage 4 / 4 — ML Pipeline")
    run_ml()

    logger.info("=" * 60)
    logger.info("Poetry-EEG Analysis Pipeline — COMPLETE")
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    run_pipeline()


if __name__ == "__main__":
    main()
