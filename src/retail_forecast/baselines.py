"""Naive forecasts every model has to beat.

Each baseline returns predictions aligned with `target_rows`, using only
information available `horizon` weeks before each target week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COL, SERIES_KEYS, TARGET


def _shifted(history: pd.DataFrame, target_rows: pd.DataFrame, weeks: int) -> np.ndarray:
    """Value of the same series `weeks` weeks earlier, looked up by date."""
    lookup = history[[*SERIES_KEYS, DATE_COL, TARGET]].copy()
    lookup[DATE_COL] = lookup[DATE_COL] + pd.Timedelta(weeks=weeks)
    merged = target_rows[[*SERIES_KEYS, DATE_COL]].merge(
        lookup, on=[*SERIES_KEYS, DATE_COL], how="left")
    return merged[TARGET].to_numpy()


def seasonal_naive(history: pd.DataFrame, target_rows: pd.DataFrame,
                   season: int = 52) -> np.ndarray:
    """Same week last year: y_hat(t) = y(t - 52)."""
    return _shifted(history, target_rows, season)


def last_value(history: pd.DataFrame, target_rows: pd.DataFrame,
               horizon: int = 1) -> np.ndarray:
    """Most recent observed week: y_hat(t) = y(t - h)."""
    return _shifted(history, target_rows, horizon)


def historical_mean(history: pd.DataFrame, target_rows: pd.DataFrame,
                    cutoff: str | pd.Timestamp) -> np.ndarray:
    """Mean of each store x department over all weeks before `cutoff`."""
    past = history[history[DATE_COL] < pd.Timestamp(cutoff)]
    means = past.groupby(SERIES_KEYS)[TARGET].mean().rename("pred").reset_index()
    merged = target_rows[SERIES_KEYS].merge(means, on=SERIES_KEYS, how="left")
    return merged["pred"].to_numpy()
