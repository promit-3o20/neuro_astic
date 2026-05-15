"""Entrypoint: ROI band-power feature extraction."""
from poetryeeg_anlys.pipelines.feature_pipeline import run_features

if __name__ == "__main__":
    run_features()
