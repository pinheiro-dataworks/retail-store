import numpy as np
import pandas as pd
import pytest

from retail_forecast import analytics


def test_controlled_lift_matches_construction(panel):
    # The fixture multiplies holiday weeks by 1.5 on top of a per-series level.
    lift = analytics.holiday_lift(panel)
    assert lift["controlled"] == pytest.approx(0.5, abs=0.05)
    assert set(lift["by_event"]["holiday_name"]) == {"Christmas"}


def test_store_type_summary_is_computed_not_hardcoded(panel):
    summary = analytics.store_type_summary(panel).set_index("store_type")
    assert summary.loc["A", "num_stores"] == 1
    assert summary["share"].sum() == pytest.approx(1.0)
    assert summary.loc["A", "label"] == "Large-format"


def test_kpis(panel):
    k = analytics.kpis(panel, analytics.holiday_lift(panel))
    assert k["total_sales"] == pytest.approx(panel["weekly_sales"].sum())
    assert k["holiday_lift_controlled_pct"] == pytest.approx(50, abs=5)


def test_monthly_revenue_sums_to_total(panel):
    m = analytics.monthly_revenue(panel)
    assert m["revenue"].sum() == pytest.approx(panel["weekly_sales"].sum())


def test_markdown_correlations_has_both_views():
    rng = np.random.default_rng(3)
    n = 2_000
    holiday = (rng.uniform(size=n) < 0.15).astype(int)
    df = pd.DataFrame({
        "is_holiday": holiday,
        "weekly_sales": 100 + 50 * holiday + rng.normal(0, 10, n),
        # markdown_1 is larger on holidays but unrelated to sales otherwise
        **{f"markdown_{k}": rng.uniform(size=n) + (5 * holiday if k == 1 else 0)
           for k in range(1, 6)},
    })
    out = analytics.markdown_correlations(df)
    assert list(out.columns) == ["all_weeks", "non_holiday"]
    assert out.loc["markdown_1", "all_weeks"] > 0.5
    assert abs(out.loc["markdown_1", "non_holiday"]) < 0.1
