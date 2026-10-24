import numpy as np
import pandas as pd
import pytest

from retail_forecast.features import build_features, feature_columns, lag_steps

HISTORY_COLS = ["roll_mean_4", "roll_mean_13", "roll_std_4", "expanding_mean"]


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_history_features_ignore_the_last_h_weeks(panel, horizon):
    """Changing sales from week t - h + 1 onwards must not change features at week t."""
    cutoff = panel["date"].sort_values().unique()[100]
    target_week = cutoff + pd.Timedelta(weeks=horizon - 1)

    base = build_features(panel, horizon)
    tampered = panel.copy()
    tampered.loc[tampered["date"] >= cutoff, "weekly_sales"] *= 1000
    changed = build_features(tampered, horizon)

    cols = [c for c in feature_columns(horizon) if c.startswith("lag_")] + HISTORY_COLS
    a = base.loc[base["date"] == target_week, cols].to_numpy(dtype=float)
    b = changed.loc[changed["date"] == target_week, cols].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_lag_values_are_exact(panel):
    feats = build_features(panel, horizon=1)
    s = feats[(feats.store_id == 1) & (feats.department == 1)].reset_index(drop=True)
    assert s.loc[10, "lag_1"] == pytest.approx(s.loc[9, "weekly_sales"])
    assert s.loc[60, "lag_52"] == pytest.approx(s.loc[8, "weekly_sales"])
    assert s.loc[10, "roll_mean_4"] == pytest.approx(s.loc[6:9, "weekly_sales"].mean())
    assert s.loc[10, "expanding_mean"] == pytest.approx(s.loc[0:9, "weekly_sales"].mean())


def test_lags_never_shorter_than_horizon():
    for h in range(1, 13):
        assert min(lag_steps(h)) == h
    with pytest.raises(ValueError):
        lag_steps(0)


def test_series_do_not_bleed_into_each_other(panel):
    feats = build_features(panel, horizon=1)
    first_weeks = feats[feats["date"] == feats["date"].min()]
    assert first_weeks["lag_1"].isna().all()


def test_categoricals_and_exogenous_switch(panel):
    feats = build_features(panel)
    assert feats["store_id"].dtype.name == "category"
    assert not feats["region"].cat.ordered
    assert "cpi" not in feature_columns()
    assert "cpi" in feature_columns(include_exogenous=True)
