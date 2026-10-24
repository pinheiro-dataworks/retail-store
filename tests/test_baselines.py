import numpy as np
import pandas as pd
import pytest

from retail_forecast.baselines import historical_mean, last_value, seasonal_naive


def _week(panel, offset_weeks=0):
    return panel[panel["date"] == panel["date"].max() - pd.Timedelta(weeks=offset_weeks)]


def test_seasonal_naive_uses_same_week_last_year(panel):
    pred = seasonal_naive(panel, _week(panel))
    np.testing.assert_allclose(pred, _week(panel, 52)["weekly_sales"].to_numpy())


def test_last_value_respects_horizon(panel):
    pred = last_value(panel, _week(panel), horizon=3)
    np.testing.assert_allclose(pred, _week(panel, 3)["weekly_sales"].to_numpy())


def test_historical_mean_ignores_rows_after_cutoff(panel):
    cutoff = pd.Timestamp("2023-01-01")
    target = panel[panel["date"] >= cutoff]
    tampered = panel.copy()
    tampered.loc[tampered["date"] >= cutoff, "weekly_sales"] = 1e12
    np.testing.assert_allclose(historical_mean(panel, target, cutoff),
                               historical_mean(tampered, target, cutoff))


def test_historical_mean_value(panel):
    cutoff = pd.Timestamp("2023-01-01")
    row = panel[(panel.store_id == 1) & (panel.department == 2)].iloc[[-1]]
    past = panel[(panel.store_id == 1) & (panel.department == 2) & (panel.date < cutoff)]
    assert historical_mean(panel, row, cutoff)[0] == pytest.approx(past["weekly_sales"].mean())
