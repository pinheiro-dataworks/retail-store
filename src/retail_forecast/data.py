"""Load the raw CSVs and build the store x department x week panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATE_COL, RAW_DIR, SERIES_KEYS

NO_HOLIDAY = "None"


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read sales, stores and features exactly as published."""
    missing = [n for n in ("sales.csv", "stores.csv", "features.csv") if not (raw_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found in {raw_dir}. "
            "Run `python scripts/download_data.py` first."
        )
    sales = pd.read_csv(raw_dir / "sales.csv", parse_dates=[DATE_COL])
    stores = pd.read_csv(raw_dir / "stores.csv")
    # holiday_name uses the literal string "None"; keep it instead of letting
    # pandas turn it into NaN.
    features = pd.read_csv(raw_dir / "features.csv", parse_dates=[DATE_COL],
                           keep_default_na=False)
    return sales, stores, features


def build_panel(sales: pd.DataFrame, stores: pd.DataFrame,
                features: pd.DataFrame) -> pd.DataFrame:
    """Merge the three tables into one row per store x department x week.

    `is_holiday` exists in both sales and features; the copies agree, so the
    one from sales is kept.
    """
    feats = features.drop(columns=["is_holiday"])
    df = (
        sales.merge(stores, on="store_id", how="left", validate="many_to_one")
        .merge(feats, on=["store_id", DATE_COL], how="left", validate="many_to_one")
    )
    df["holiday_name"] = df["holiday_name"].replace("", NO_HOLIDAY).fillna(NO_HOLIDAY)
    df = df.sort_values([*SERIES_KEYS, DATE_COL]).reset_index(drop=True)
    _check_panel(df)
    return df


def _check_panel(df: pd.DataFrame) -> None:
    if df.duplicated([*SERIES_KEYS, DATE_COL]).any():
        raise ValueError("duplicate store/department/week rows")
    if df[["store_type", "region", "temperature"]].isna().any().any():
        raise ValueError("unmatched rows after merging stores/features")
    gaps = df.groupby(SERIES_KEYS)[DATE_COL].diff().dropna().dt.days.unique()
    if len(gaps) and set(gaps) != {7}:
        raise ValueError(f"series are not weekly and gap-free (gaps found: {sorted(gaps)})")


def load_panel(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return build_panel(*load_raw(raw_dir))
