"""Paths and the few constants that define the forecasting problem."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
WEB_DATA_DIR = ROOT / "web" / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# One row per store x department x week.
SERIES_KEYS = ["store_id", "department"]
DATE_COL = "date"
TARGET = "weekly_sales"

# Weight used by the Kaggle competition metric (WMAE) for holiday weeks.
HOLIDAY_WEIGHT = 5

# Holdout layout. Everything before VALID_START is used for fitting,
# VALID_START..TEST_START for early stopping / tuning, and TEST_START onwards
# is touched exactly once to report the final numbers.
VALID_START = "2024-01-01"
TEST_START = "2024-07-01"

SEED = 42
