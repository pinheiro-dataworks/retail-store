import pandas as pd

from retail_forecast.forecast import (
    aggregate_band,
    evaluate_horizons,
    extend_calendar,
    forecast_future,
)

PARAMS = {"objective": "l2", "num_leaves": 7, "min_child_samples": 5, "learning_rate": 0.1}


def test_extend_calendar_adds_empty_future_weeks(panel):
    last = panel["date"].max()
    ext = extend_calendar(panel, 3, {(last + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"): "X"})
    future = ext[ext["date"] > last]
    assert len(future) == 3 * panel.groupby(["store_id", "department"]).ngroups
    assert future["weekly_sales"].isna().all()
    assert future.loc[future["date"] == future["date"].min(), "is_holiday"].eq(1).all()
    assert future["store_type"].notna().all()


def test_horizon_evaluation_and_future_forecast(panel):
    table, rounds, agg = evaluate_horizons(panel, PARAMS, range(1, 3),
                                           valid_start="2023-10-01", test_start="2024-01-01")
    assert list(table["horizon"]) == [1, 2]
    assert set(rounds) == {1, 2}
    assert {"wmae", "naive_wmae", "skill_vs_naive"} <= set(table.columns)

    future = forecast_future(panel, PARAMS, rounds, holidays={})
    assert sorted(future["horizon"].unique()) == [1, 2]
    assert (future["date"] > panel["date"].max()).all()
    assert (future["forecast"] >= 0).all()

    band = aggregate_band(future, agg)
    assert len(band) == 2
    assert (band["lower"] <= band["forecast"]).all()
    assert (band["forecast"] <= band["upper"]).all()
