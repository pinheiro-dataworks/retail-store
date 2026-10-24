"""Descriptive aggregations shown on the dashboard.

Everything here is computed from the panel; nothing is typed in by hand (v1
hard-coded store counts and revenue shares in the assistant context).
"""

from __future__ import annotations

import pandas as pd

from .config import DATE_COL, SERIES_KEYS, TARGET
from .data import NO_HOLIDAY

SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]
STORE_TYPE_LABEL = {"A": "Large-format", "B": "Medium-format", "C": "Small-format"}
MARKDOWNS = [f"markdown_{i}" for i in range(1, 6)]
CORR_COLUMNS = ["store_size", "is_holiday", "temperature", "fuel_price", "cpi", "unemployment",
                "week_of_year", "month", *MARKDOWNS]


def holiday_lift(df: pd.DataFrame) -> dict:
    """Holiday lift measured two ways.

    raw: mean sales on holiday rows vs. non-holiday rows. It mixes the holiday
    effect with *which* stores/departments and seasons are in each group.
    controlled: every row is divided by its own store x department non-holiday
    mean before averaging, so the comparison is within the same series.
    """
    is_hol = df["is_holiday"] == 1
    raw = df.loc[is_hol, TARGET].mean() / df.loc[~is_hol, TARGET].mean() - 1

    base = df[~is_hol].groupby(SERIES_KEYS)[TARGET].mean().rename("series_base")
    rel = df.join(base, on=SERIES_KEYS)
    rel["ratio"] = rel[TARGET] / rel["series_base"]
    controlled = rel.loc[is_hol, "ratio"].mean() - 1

    by_event = (rel[is_hol].groupby("holiday_name")
                .agg(avg_sales=(TARGET, "mean"), controlled_lift=("ratio", "mean"),
                     weeks=(DATE_COL, "nunique"))
                .sort_values("avg_sales", ascending=False))
    non_hol_avg = df.loc[~is_hol, TARGET].mean()
    by_event["raw_lift"] = by_event["avg_sales"] / non_hol_avg - 1
    by_event["controlled_lift"] -= 1
    return {"raw": float(raw), "controlled": float(controlled),
            "non_holiday_avg": float(non_hol_avg), "by_event": by_event.reset_index()}


def store_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = df[TARGET].sum()
    out = (df.groupby("store_type")
           .agg(revenue=(TARGET, "sum"), avg_week=(TARGET, "mean"),
                num_stores=("store_id", "nunique"), avg_size=("store_size", "mean"))
           .sort_values("revenue", ascending=False).reset_index())
    out["share"] = out["revenue"] / total
    out["label"] = out["store_type"].map(STORE_TYPE_LABEL)
    return out


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("region")
            .agg(revenue=(TARGET, "sum"), avg_week=(TARGET, "mean"),
                 stores=("store_id", "nunique"))
            .sort_values("revenue", ascending=False).reset_index())


def kpis(df: pd.DataFrame, lift: dict) -> dict:
    return {
        "total_sales": float(df[TARGET].sum()),
        "avg_weekly_sales": float(df[TARGET].mean()),
        "holiday_lift_pct": lift["raw"] * 100,
        "holiday_lift_controlled_pct": lift["controlled"] * 100,
        "max_store_revenue": float(df.groupby("store_id")[TARGET].sum().max()),
    }


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    return (df.assign(month=df[DATE_COL].dt.strftime("%Y-%m"))
            .groupby("month", as_index=False)[TARGET].sum()
            .rename(columns={TARGET: "revenue"}))


def store_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly revenue by type/region/season, re-aggregated in the browser by the filters."""
    out = (df.assign(year=df[DATE_COL].dt.year)
           .groupby([DATE_COL, "store_type", "region", "season", "is_holiday", "year"],
                    as_index=False)[TARGET].sum()
           .rename(columns={TARGET: "revenue"}))
    out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d")
    return out


def weekly_type_trend(df: pd.DataFrame, n_weeks: int = 60) -> pd.DataFrame:
    first = sorted(df[DATE_COL].unique())[:n_weeks]
    return (df[df[DATE_COL].isin(first)]
            .pivot_table(index=DATE_COL, columns="store_type", values=TARGET, aggfunc="sum")
            .fillna(0))


def seasonal_pattern(df: pd.DataFrame) -> pd.Series:
    return df.groupby("season")[TARGET].mean().reindex(SEASON_ORDER)


def correlations(df: pd.DataFrame) -> pd.Series:
    data = df.assign(week_of_year=df[DATE_COL].dt.isocalendar().week.astype(int),
                     month=df[DATE_COL].dt.month)
    return data[CORR_COLUMNS].corrwith(data[TARGET]).sort_values(ascending=False)


def markdown_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Markdown vs. sales correlation, on all weeks and on non-holiday weeks only.

    Markdowns run about twice as high on holiday weeks, so the all-weeks number
    mostly measures the holiday effect. The non-holiday column removes it.
    """
    non_hol = df[df["is_holiday"] == 0]
    return pd.DataFrame({
        "all_weeks": df[MARKDOWNS].corrwith(df[TARGET]),
        "non_holiday": non_hol[MARKDOWNS].corrwith(non_hol[TARGET]),
    })


def top_departments(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return df.groupby("department")[TARGET].sum().nlargest(n)


def holiday_names(df: pd.DataFrame) -> list[str]:
    return [h for h in df["holiday_name"].unique() if h != NO_HOLIDAY]
