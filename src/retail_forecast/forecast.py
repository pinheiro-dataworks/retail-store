"""Direct multi-horizon forecasting.

One model per horizon h = 1..H, each trained on features that only use sales up
to t - h. Compared with feeding predictions back recursively, this keeps the
errors of week 1 from compounding into week 8, and it lets the holdout report
an honest error for every horizon the dashboard shows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines, metrics
from .config import DATE_COL, SERIES_KEYS, TARGET, TEST_START, VALID_START
from .features import build_features, feature_columns
from .models import LGBMForecaster

# Holiday flags for the weeks right after the data ends (2024-12-21). They
# follow the tagging used in the data: the last Saturday of December counts as
# Christmas and the first Saturday of January as New Year.
FUTURE_HOLIDAYS = {
    "2024-12-28": "Christmas",
    "2025-01-04": "New Year",
}

STATIC_COLS = ["store_type", "store_size", "region"]


def extend_calendar(panel: pd.DataFrame, n_weeks: int,
                    holidays: dict[str, str] = FUTURE_HOLIDAYS) -> pd.DataFrame:
    """Append `n_weeks` future rows per series with an empty target."""
    last = panel[DATE_COL].max()
    future_dates = pd.date_range(last + pd.Timedelta(weeks=1), periods=n_weeks, freq="7D")
    series = panel.drop_duplicates(SERIES_KEYS)[SERIES_KEYS + STATIC_COLS]
    future = series.merge(pd.DataFrame({DATE_COL: future_dates}), how="cross")
    names = future[DATE_COL].dt.strftime("%Y-%m-%d").map(holidays)
    future["holiday_name"] = names.fillna("None")
    future["is_holiday"] = names.notna().astype(int)
    future[TARGET] = np.nan
    return pd.concat([panel, future], ignore_index=True).sort_values(
        [*SERIES_KEYS, DATE_COL]).reset_index(drop=True)


def evaluate_horizons(panel: pd.DataFrame, params: dict, horizons: range,
                      valid_start: str = VALID_START, test_start: str = TEST_START
                      ) -> tuple[pd.DataFrame, dict[int, int], pd.DataFrame]:
    """Holdout metrics for each horizon.

    Returns the metric table, the number of boosting rounds chosen per horizon
    (reused when training the production models) and the weekly aggregate
    errors used to size the interval band on the total-sales chart.
    """
    rows, rounds, agg_rows = [], {}, []
    v0, t0 = pd.Timestamp(valid_start), pd.Timestamp(test_start)
    for h in horizons:
        feats = build_features(panel, h)
        cols = feature_columns(h)
        train, valid = feats[feats[DATE_COL] < v0], feats[(feats[DATE_COL] >= v0)
                                                          & (feats[DATE_COL] < t0)]
        test = feats[feats[DATE_COL] >= t0]
        rounds[h] = LGBMForecaster(cols, params=params).fit(train, valid).n_estimators
        model = LGBMForecaster(cols, params=params, n_estimators=rounds[h]).fit(
            feats[feats[DATE_COL] < t0])
        pred = model.predict(test)
        naive = baselines.seasonal_naive(panel, test)

        m = metrics.evaluate(test[TARGET], pred, test["is_holiday"])
        naive_wmae = metrics.wmae(test[TARGET], naive, test["is_holiday"])
        rows.append({"horizon": h, **m, "naive_wmae": naive_wmae,
                     "skill_vs_naive": metrics.skill_score(m["wmae"], naive_wmae)})

        weekly = test.assign(pred=pred).groupby(DATE_COL)[[TARGET, "pred"]].sum()
        agg_rows.append(pd.DataFrame({"horizon": h, "date": weekly.index,
                                      "rel_error": weekly[TARGET] / weekly["pred"] - 1}))
    return pd.DataFrame(rows), rounds, pd.concat(agg_rows, ignore_index=True)


def forecast_future(panel: pd.DataFrame, params: dict, rounds: dict[int, int],
                    holidays: dict[str, str] = FUTURE_HOLIDAYS) -> pd.DataFrame:
    """Forecast the next max(rounds) weeks for every series, trained on all history."""
    horizons = sorted(rounds)
    extended = extend_calendar(panel, max(horizons), holidays)
    last = panel[DATE_COL].max()
    out = []
    for h in horizons:
        feats = build_features(extended, h)
        model = LGBMForecaster(feature_columns(h), params=params, n_estimators=rounds[h]).fit(
            feats[feats[TARGET].notna()])
        target_rows = feats[feats[DATE_COL] == last + pd.Timedelta(weeks=h)]
        out.append(target_rows[[*SERIES_KEYS, DATE_COL, "store_type", "region", "is_holiday",
                                "holiday_name"]].assign(horizon=h,
                                                        forecast=model.predict(target_rows)))
    result = pd.concat(out, ignore_index=True)
    result["forecast"] = result["forecast"].clip(lower=0)
    return result


def aggregate_band(future: pd.DataFrame, agg_errors: pd.DataFrame,
                   lower_q: float = 0.1, upper_q: float = 0.9) -> pd.DataFrame:
    """Total weekly forecast with an empirical band from holdout aggregate errors.

    Summing per-series quantiles would overstate the uncertainty of the total,
    so the band is taken from how far the *total* was off on the holdout at
    each horizon.
    """
    totals = future.groupby(["horizon", DATE_COL], as_index=False)["forecast"].sum()
    q = agg_errors.groupby("horizon")["rel_error"].agg(
        q_lo=lambda e: min(e.quantile(lower_q), 0.0),
        q_hi=lambda e: max(e.quantile(upper_q), 0.0))
    totals = totals.merge(q, left_on="horizon", right_index=True)
    totals["lower"] = totals["forecast"] * (1 + totals["q_lo"])
    totals["upper"] = totals["forecast"] * (1 + totals["q_hi"])
    return totals.drop(columns=["q_lo", "q_hi"])
