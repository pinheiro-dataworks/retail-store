"""Feature engineering.

Every history-based feature is computed from the target shifted by the
forecast horizon `h`: the row for week t only sees sales up to week t - h.
That is what makes the same code valid for evaluation and for producing real
forecasts, and it is what `tests/test_features.py` checks.
"""

from __future__ import annotations

import pandas as pd

from .config import DATE_COL, SERIES_KEYS, TARGET

ROLLING_WINDOWS = (4, 13)
SEASONAL_LAG = 52


def lag_steps(horizon: int) -> list[int]:
    """Lags available at a given horizon: the latest few weeks plus last year."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    steps = {horizon, horizon + 1, horizon + 3}
    if SEASONAL_LAG >= horizon:
        steps.add(SEASONAL_LAG)
    return sorted(steps)


def add_history_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Lags, rolling statistics and an expanding mean of past sales.

    The expanding mean is a leak-free target encoding of the store x department
    level: it only averages weeks that were already observed.
    """
    out = df.sort_values([*SERIES_KEYS, DATE_COL]).copy()
    grouped = out.groupby(SERIES_KEYS, sort=False)[TARGET]
    for k in lag_steps(horizon):
        out[f"lag_{k}"] = grouped.shift(k)

    past = grouped.shift(horizon)
    past_grouped = past.groupby([out[k] for k in SERIES_KEYS], sort=False)
    for w in ROLLING_WINDOWS:
        out[f"roll_mean_{w}"] = past_grouped.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean())
    out["roll_std_4"] = past_grouped.transform(lambda s: s.rolling(4, min_periods=2).std())
    out["expanding_mean"] = past_grouped.transform(lambda s: s.expanding().mean())
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_of_year"] = out[DATE_COL].dt.isocalendar().week.astype(int)
    out["month"] = out[DATE_COL].dt.month
    return out


# Columns known in advance for any future week (calendar, store attributes).
CATEGORICAL = ["store_id", "department", "store_type", "region", "holiday_name"]
STATIC_NUMERIC = ["store_size", "is_holiday", "week_of_year", "month"]

# Macro / promo variables. In this dataset they behave like independent draws
# week to week (see notebooks/01_eda.ipynb) and they would not be known in
# advance for a real forecast, so they are left out of the model by default
# and only used in the ablation that justifies dropping them.
EXOGENOUS = ["temperature", "fuel_price", "cpi", "unemployment",
             "markdown_1", "markdown_2", "markdown_3", "markdown_4", "markdown_5"]


def add_categoricals(df: pd.DataFrame, categories: dict[str, list] | None = None
                     ) -> tuple[pd.DataFrame, dict[str, list]]:
    """Cast identifiers to pandas categoricals with a fixed category list.

    Store and department are the grain of the forecast, so the model needs to
    see them directly; v1 only had store type, which ended up as a crude proxy
    for store identity. Categories are fixed from the full panel so train and
    test share the same codes. Unordered categoricals avoid the artificial
    ordering that LabelEncoder imposed on region/season.
    """
    out = df.copy()
    categories = categories or {c: sorted(out[c].unique().tolist()) for c in CATEGORICAL}
    for col in CATEGORICAL:
        out[col] = pd.Categorical(out[col], categories=categories[col])
    return out, categories


def feature_columns(horizon: int = 1, include_exogenous: bool = False) -> list[str]:
    history = [f"lag_{k}" for k in lag_steps(horizon)]
    history += [f"roll_mean_{w}" for w in ROLLING_WINDOWS] + ["roll_std_4", "expanding_mean"]
    cols = CATEGORICAL + STATIC_NUMERIC + history
    return cols + EXOGENOUS if include_exogenous else cols


def build_features(df: pd.DataFrame, horizon: int = 1,
                   categories: dict[str, list] | None = None) -> pd.DataFrame:
    """Full feature frame for one forecast horizon."""
    out = add_history_features(add_calendar_features(df), horizon)
    out, _ = add_categoricals(out, categories)
    return out
