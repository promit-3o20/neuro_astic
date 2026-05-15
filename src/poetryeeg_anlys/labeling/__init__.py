"""labeling – behavioral alignment and label encoding."""

from .behavior import (
    load_participants, create_subject_mapping,
    get_behavior_file, load_behavior,
)
from .add_labels import (
    attach_labels, encode_labels,
    save_labeled_fif, epochs_to_parquet,
    process_subject, run_labeling,
)

__all__ = [
    "load_participants", "create_subject_mapping",
    "get_behavior_file", "load_behavior",
    "attach_labels", "encode_labels",
    "save_labeled_fif", "epochs_to_parquet",
    "process_subject", "run_labeling",
]
