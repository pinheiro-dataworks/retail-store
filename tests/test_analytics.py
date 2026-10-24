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
