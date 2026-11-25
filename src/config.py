"""Paths and shared settings.

The real admissions file (TU.csv) is confidential student-level data and is never committed.
If it is present at data/raw/TU.csv the pipeline uses it; otherwise it falls back to the
synthetic file in data/synthetic/ and writes results to a separate folder so the two are
never mixed up.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
REAL_DATA = ROOT / "data" / "raw" / "TU.csv"
SYNTHETIC_DATA = ROOT / "data" / "synthetic" / "tu_synthetic.csv"

RANDOM_STATE = 0
RARE_CATEGORY_MIN = 100   # categories with fewer training rows are grouped into "Other"
CV_FOLDS = 5

DEMOGRAPHIC_FEATURES = ["sex", "ethnicity", "race", "religion"]


def data_source():
    """Return (path to data, output folder, label)."""
    if REAL_DATA.exists():
        return REAL_DATA, ROOT / "outputs", "real"
    out = ROOT / "outputs" / "synthetic_run"
    return SYNTHETIC_DATA, out, "synthetic"
