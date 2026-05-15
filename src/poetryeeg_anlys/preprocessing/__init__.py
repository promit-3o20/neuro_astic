"""preprocessing – raw loading, filtering, ICA, epoching, quality control."""

from .preprocess import read_eeg, set_channels
from .filtering import rereference_eeg, filter_eeg
from .ica import run_ica
from .epoching import extract_events, epoch_stimulus, epoch_rest
from .quality import reject_epochs

__all__ = [
    "read_eeg", "set_channels",
    "rereference_eeg", "filter_eeg",
    "run_ica",
    "extract_events", "epoch_stimulus", "epoch_rest",
    "reject_epochs",
]
