"""Contract tests for the JSON the dashboard and the assistant read.

They run against the committed files in web/data/, so a pipeline change that
breaks the front-end (renamed key, NaN, misaligned arrays) fails in CI.
"""

import json
import math
from pathlib import Path

import pytest

WEB_DATA = Path(__file__).resolve().parents[1] / "web" / "data"


@pytest.fixture(scope="module")
def data():
    return json.loads((WEB_DATA / "dashboard_data.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def context():
    return json.loads((WEB_DATA / "assistant_context.json").read_text(encoding="utf-8"))


def _numbers(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)


def test_top_level_keys(data):
    expected = {"schema_version", "kpis", "monthly_revenue", "store_weekly", "by_type",
                "by_region", "weekly_type_trend", "holiday_impact", "non_holiday_baseline",
                "seasonal_pattern", "feature_importance", "pearson_correlation",
                "markdown_correlation", "top_departments", "evaluation", "forecast", "meta"}
    assert expected <= set(data)
    assert data["schema_version"] == 2


def test_all_numbers_are_finite(data):
    assert all(math.isfinite(x) for x in _numbers(data))


def test_holdout_is_strictly_after_training(data):
    h = data["evaluation"]["holdout"]
    assert h["train_end"] <= h["valid"][0] < h["valid"][1] <= h["test_start"] <= h["test_end"]
    assert min(data["evaluation"]["weekly"]["dates"]) >= h["test_start"]


def test_model_table_has_baseline_and_skill(data):
    models = {m["model"]: m for m in data["evaluation"]["models"]}
    naive = models["Seasonal naive (t-52)"]
    assert naive["skill_vs_naive"] == 0
    for m in models.values():
        assert m["skill_vs_naive"] == pytest.approx(1 - m["wmae"] / naive["wmae"], abs=1e-3)
        assert {"wmae", "wape", "bias", "mae", "rmse", "r2"} <= set(m)


def test_weekly_arrays_are_aligned(data):
    w = data["evaluation"]["weekly"]
    n = len(w["dates"])
    assert n > 0
    for key in ("actual", "lightgbm", "seasonal_naive", "is_holiday", "holiday",
                "mae_lightgbm", "mae_seasonal_naive"):
        assert len(w[key]) == n, key


def test_forecast_is_in_the_future_and_banded(data):
    f = data["forecast"]
    last_date = data["meta"]["date_range"].split(" to ")[1]
    assert min(f["dates"]) > last_date
    assert f["horizon"] == list(range(1, len(f["dates"]) + 1))
    for lo, mid, hi in zip(f["lower"], f["forecast"], f["upper"], strict=True):
        assert lo <= mid <= hi


def test_interval_and_importance(data):
    iv = data["evaluation"]["interval"]
    assert 0 < iv["coverage"] <= 1 and iv["nominal"] == 0.8
    shares = [f["importance"] for f in data["feature_importance"]]
    assert sum(shares) == pytest.approx(1, abs=1e-3)


def test_assistant_context_is_consistent_with_dashboard(data, context):
    counts = {t["type"]: t["num_stores"] for t in data["by_type"]}
    assert {t: v["count"] for t, v in context["store_types"].items()} == counts
    best = next(m for m in data["evaluation"]["models"] if m["model"] == "LightGBM + lags")
    assert context["model_evaluation"]["wmae"] == round(best["wmae"])
