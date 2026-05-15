"""Entrypoint: batch EEG preprocessing."""
from poetryeeg_anlys.pipelines.preprocessing_pipeline import run_preprocessing

if __name__ == "__main__":
    run_preprocessing(True)
