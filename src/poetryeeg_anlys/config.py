from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intrmd_data"
RESULTS_DIR = PROJECT_ROOT / "results"
