"""Forecast error metrics.

All functions take plain array-likes so they can be reused by the pipeline,
the tests and the notebook without dragging pandas objects around.
"""

from __future__ import annotations

import numpy as np

from .config import HOLIDAY_WEIGHT


def _arrays(*arrays):
    out = [np.asarray(a, dtype=float) for a in arrays]
    if len({a.shape for a in out}) != 1:
        raise ValueError("inputs must have the same shape")
    return out


def mae(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wmae(y_true, y_pred, is_holiday, holiday_weight: float = HOLIDAY_WEIGHT) -> float:
    """Weighted MAE from the original Kaggle competition.

    Holiday weeks count `holiday_weight` times (5 by default) because a miss
    on Black Friday or Christmas costs far more than a miss on a quiet week.
    """
    y, p, h = _arrays(y_true, y_pred, is_holiday)
    w = np.where(h > 0, holiday_weight, 1.0)
    return float(np.sum(w * np.abs(y - p)) / np.sum(w))


def wape(y_true, y_pred) -> float:
    """Sum of absolute errors over the sum of actuals (a.k.a. MAD/mean ratio).

    Unlike MAPE it does not blow up on small weeks, which is why it is the
    usual headline number in retail planning.
    """
    y, p = _arrays(y_true, y_pred)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)))


def bias(y_true, y_pred) -> float:
    """Signed total error relative to total actuals.

    Positive means over-forecasting (excess stock), negative means
    under-forecasting (stock-outs).
    """
    y, p = _arrays(y_true, y_pred)
    return float(np.sum(p - y) / np.sum(np.abs(y)))


def r2(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def skill_score(model_error: float, baseline_error: float) -> float:
    """1 - model / baseline. 0 means no better than the baseline."""
    if baseline_error <= 0:
        raise ValueError("baseline error must be positive")
    return float(1.0 - model_error / baseline_error)


def evaluate(y_true, y_pred, is_holiday) -> dict[str, float]:
    """The full metric set reported for every model."""
    return {
        "wmae": wmae(y_true, y_pred, is_holiday),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def interval_coverage(y_true, lower, upper) -> float:
    """Share of actuals that fall inside [lower, upper]."""
    y, lo, hi = _arrays(y_true, lower, upper)
    return float(np.mean((y >= lo) & (y <= hi)))
