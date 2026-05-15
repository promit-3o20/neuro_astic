"""
pipelines/ml_pipeline.py
=========================
Stub ML pipeline orchestrator.

Implement model training and evaluation by importing from
:mod:`poetryeeg_anlys.ml`.
"""

from __future__ import annotations

from poetryeeg_anlys.config.paths import Paths
from poetryeeg_anlys.utils.logger import get_logger


def run_ml() -> None:
    """Placeholder — implement ML training here."""
    Paths.make_all()
    logger = get_logger("ml", log_dir=Paths.LOG_ML)
    logger.info("ML pipeline: not yet implemented.")


def main() -> None:
    run_ml()


if __name__ == "__main__":
    main()
