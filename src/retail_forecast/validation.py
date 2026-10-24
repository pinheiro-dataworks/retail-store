"""Time-based splitting.

The first version of the project used `train_test_split(shuffle=True)`, which
lets the model learn from weeks that come *after* the ones it is scored on.
Every split here is defined by calendar dates, so the training side is always
strictly in the past of the evaluation side.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from .config import DATE_COL, TEST_START, VALID_START


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def temporal_split(df: pd.DataFrame, valid_start: str = VALID_START,
                   test_start: str = TEST_START) -> Split:
    """Train < valid_start <= valid < test_start <= test."""
    v0, t0 = pd.Timestamp(valid_start), pd.Timestamp(test_start)
    if v0 >= t0:
        raise ValueError("valid_start must be before test_start")
    dates = df[DATE_COL]
    return Split(
        train=df[dates < v0],
        valid=df[(dates >= v0) & (dates < t0)],
        test=df[dates >= t0],
    )


@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def expanding_window_folds(dates: pd.Series, n_folds: int, test_weeks: int,
                           end: str | pd.Timestamp | None = None) -> list[Fold]:
    """Backtest folds with an expanding training window.

    The last fold ends at `end` (exclusive; defaults to after the last date) and
    each previous fold is shifted back by `test_weeks`. Folds are built on the
    unique calendar weeks, not on rows, so every series is cut at the same date.
    """
    weeks = pd.Series(pd.to_datetime(dates).unique()).sort_values().reset_index(drop=True)
    if end is not None:
        weeks = weeks[weeks < pd.Timestamp(end)].reset_index(drop=True)
    needed = n_folds * test_weeks
    if needed >= len(weeks):
        raise ValueError(f"{n_folds} folds x {test_weeks} weeks do not fit in {len(weeks)} weeks")

    folds = []
    for k in range(n_folds, 0, -1):
        start_idx = len(weeks) - k * test_weeks
        test_start = weeks[start_idx]
        stop_idx = start_idx + test_weeks
        if stop_idx < len(weeks):
            test_end = weeks[stop_idx]
        else:
            test_end = weeks.iloc[-1] + pd.Timedelta(days=1)
        folds.append(Fold(train_end=test_start, test_start=test_start, test_end=test_end))
    return folds


def iter_fold_frames(df: pd.DataFrame, folds: list[Fold]
                     ) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    dates = df[DATE_COL]
    for fold in folds:
        yield (fold,
               df[dates < fold.train_end],
               df[(dates >= fold.test_start) & (dates < fold.test_end)])
