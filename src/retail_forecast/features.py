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
