"""
utils/helpers.py
================
Miscellaneous helpers that do not fit neatly elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def subject_id_from_path(path: Path | str) -> str:
    """
    Extract the subject ID (e.g. 'sub-021') from a BIDS-structured file path.

    Expects the subject directory to be three levels above the file
    (``sub-XXX/eeg/sub-XXX_task-…_eeg.set``).

    Parameters
    ----------
    path : Path or str

    Returns
    -------
    str
        E.g. ``'sub-021'``.
    """
    return Path(path).parts[-3]


def find_set_files(root: Path) -> list[Path]:
    """Return all ``*.set`` files under *root* matching the BIDS pattern."""
    return sorted(root.rglob("sub-*/eeg/*.set"))


def find_fif_files(directory: Path, pattern: str = "*.fif") -> list[Path]:
    """Return all ``.fif`` epoch files matching *pattern* in *directory*."""
    return sorted(directory.glob(pattern))


def check_resume(subject_id: str, required_paths: Iterable[Path]) -> bool:
    """
    Return True if *all* required output files already exist (resume-safe).

    Parameters
    ----------
    subject_id    : str  (used only for clarity; not checked here)
    required_paths: iterable of Path

    Returns
    -------
    bool
    """
    return all(Path(p).exists() for p in required_paths)
