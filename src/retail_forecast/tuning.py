"""Hyperparameter search with expanding-window backtesting.

Only data before the test window is used. Inside each fold the last
`inner_valid_weeks` of the training side pick the number of boosting rounds,
so the fold's own evaluation weeks never influence the model fitted on it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .config import DATE_COL, TARGET, TEST_START
from .models import DEFAULT_LGBM_PARAMS, LGBMForecaster
from .validation import expanding_window_folds, iter_fold_frames

SEARCH_SPACE = {
    "objective": ["l2", "l1"],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [50, 200],
}


@dataclass
class TuningResult:
    best_params: dict
    table: pd.DataFrame  # one row per candidate, mean/std of WMAE across folds


def candidates(space: dict[str, list] = SEARCH_SPACE) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, values, strict=True))
            for values in itertools.product(*(space[k] for k in keys))]


def backtest(df: pd.DataFrame, features: list[str], params: dict, n_folds: int = 3,
             test_weeks: int = 13, end: str = TEST_START,
             inner_valid_weeks: int = 8) -> list[float]:
    """WMAE per fold for one parameter set."""
    folds = expanding_window_folds(df[DATE_COL], n_folds, test_weeks, end=end)
    scores = []
    for fold, train, test in iter_fold_frames(df, folds):
        inner_cut = fold.train_end - pd.Timedelta(weeks=inner_valid_weeks)
        fit_part = train[train[DATE_COL] < inner_cut]
        stop_part = train[train[DATE_COL] >= inner_cut]
        model = LGBMForecaster(features, params=params).fit(fit_part, stop_part)
        test = test.dropna(subset=[TARGET])
        scores.append(metrics.wmae(test[TARGET], model.predict(test), test["is_holiday"]))
    return scores


def tune(df: pd.DataFrame, features: list[str], space: dict[str, list] = SEARCH_SPACE,
         **backtest_kwargs) -> TuningResult:
    rows = []
    for cand in candidates(space):
        params = {**DEFAULT_LGBM_PARAMS, **cand}
        scores = backtest(df, features, params, **backtest_kwargs)
        rows.append({**cand, "wmae_mean": float(np.mean(scores)),
                     "wmae_std": float(np.std(scores)), "folds": scores})
    table = pd.DataFrame(rows).sort_values("wmae_mean").reset_index(drop=True)
    best = {k: table.loc[0, k] for k in space}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return TuningResult(best_params={**DEFAULT_LGBM_PARAMS, **best}, table=table)
