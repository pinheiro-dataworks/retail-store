"""Turn a pipeline run into the JSON files served by the static dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics
from .config import DATE_COL, TARGET, TEST_START, VALID_START
from .pipeline import RunResult

SCHEMA_VERSION = 2


def _r(value, digits: int = 2):
    return round(float(value), digits)


def _date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _histogram(values: np.ndarray, bins: int = 40, clip_q: float = 0.005) -> dict:
    lo, hi = np.quantile(values, [clip_q, 1 - clip_q])
    counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    return {"bin_start": [_r(e) for e in edges[:-1]], "bin_end": [_r(e) for e in edges[1:]],
            "count": counts.tolist()}


def evaluation_block(res: RunResult) -> dict:
    test = res.test
    weekly = test.groupby(DATE_COL).agg(
        actual=(TARGET, "sum"), lightgbm=("pred_lightgbm", "sum"),
        seasonal_naive=("pred_seasonal_naive", "sum"), is_holiday=("is_holiday", "max"),
        holiday=("holiday_name", "first"))
    abs_err = test.assign(err_lgbm=(test[TARGET] - test["pred_lightgbm"]).abs(),
                          err_naive=(test[TARGET] - test["pred_seasonal_naive"]).abs())
    weekly_err = abs_err.groupby(DATE_COL)[["err_lgbm", "err_naive"]].mean()

    return {
        "holdout": {"train_end": VALID_START, "valid": [VALID_START, TEST_START],
                    "test_start": TEST_START, "test_end": _date(test[DATE_COL].max()),
                    "test_rows": len(test), "horizon_weeks": 1},
        "models": [
            {"model": r.model, "kind": r.kind, "wmae": _r(r.wmae), "mae": _r(r.mae),
             "rmse": _r(r.rmse), "wape": _r(r.wape, 4), "bias": _r(r.bias, 4),
             "r2": _r(r.r2, 4), "skill_vs_naive": _r(r.skill_vs_naive, 4)}
            for r in res.model_table.itertuples()
        ],
        "ablation": {k: _r(v) for k, v in res.ablation.items()},
        "interval": {k: _r(v, 4) for k, v in res.interval.items()},
        "best_params": res.best_params,
        "tuning": [
            {k: (_r(v) if isinstance(v, float) else v) for k, v in row.items() if k != "folds"}
            for row in res.tuning_table.to_dict(orient="records")
        ],
        "weekly": {
            "dates": [_date(d) for d in weekly.index],
            "actual": [_r(v) for v in weekly["actual"]],
            "lightgbm": [_r(v) for v in weekly["lightgbm"]],
            "seasonal_naive": [_r(v) for v in weekly["seasonal_naive"]],
            "is_holiday": weekly["is_holiday"].astype(int).tolist(),
            "holiday": [h if h != "None" else None for h in weekly["holiday"]],
            "mae_lightgbm": [_r(v) for v in weekly_err["err_lgbm"]],
            "mae_seasonal_naive": [_r(v) for v in weekly_err["err_naive"]],
        },
        "residuals": _histogram((test[TARGET] - test["pred_lightgbm"]).to_numpy()),
        "horizons": [
            {"horizon": int(r.horizon), "wmae": _r(r.wmae), "wape": _r(r.wape, 4),
             "bias": _r(r.bias, 4), "naive_wmae": _r(r.naive_wmae),
             "skill_vs_naive": _r(r.skill_vs_naive, 4)}
            for r in res.horizon_table.itertuples()
        ],
    }


def forecast_block(res: RunResult) -> dict:
    total = res.future_total.sort_values("horizon")
    by_type = res.future.pivot_table(index=DATE_COL, columns="store_type", values="forecast",
                                     aggfunc="sum", observed=True).sort_index()
    history = res.panel.groupby(DATE_COL)[TARGET].sum().iloc[-26:]
    return {
        "history": {"dates": [_date(d) for d in history.index],
                    "actual": [_r(v) for v in history]},
        "dates": [_date(d) for d in total[DATE_COL]],
        "horizon": total["horizon"].astype(int).tolist(),
        "forecast": [_r(v) for v in total["forecast"]],
        "lower": [_r(v) for v in total["lower"]],
        "upper": [_r(v) for v in total["upper"]],
        "by_store_type": {t: [_r(v) for v in by_type[t]] for t in by_type.columns},
        "holidays": {_date(d): h for d, h in
                     res.future.drop_duplicates(DATE_COL)[[DATE_COL, "holiday_name"]]
                     .itertuples(index=False) if h != "None"},
    }


def dashboard_payload(res: RunResult) -> dict:
    df = res.panel
    lift = analytics.holiday_lift(df)
    k = analytics.kpis(df, lift)
    types = analytics.store_type_summary(df)
    regions = analytics.region_summary(df)
    monthly = analytics.monthly_revenue(df)
    trend = analytics.weekly_type_trend(df)
    corr = analytics.correlations(df)
    events = lift["by_event"]

    return {
        "schema_version": SCHEMA_VERSION,
        "kpis": {key: _r(v) for key, v in k.items()},
        "monthly_revenue": {"labels": monthly["month"].tolist(),
                            "revenue": [_r(v) for v in monthly["revenue"]]},
        "store_weekly": analytics.store_weekly(df).round({"revenue": 2}).to_dict(orient="records"),
        "by_type": [{"type": r.store_type, "label": r.label, "revenue": _r(r.revenue),
                     "avg_week": _r(r.avg_week), "num_stores": int(r.num_stores),
                     "share": _r(r.share, 4), "avg_size": _r(r.avg_size, 0)}
                    for r in types.itertuples()],
        "by_region": [{"region": r.region, "revenue": _r(r.revenue), "avg_week": _r(r.avg_week),
                       "stores": int(r.stores)} for r in regions.itertuples()],
        "weekly_type_trend": {"labels": [_date(d) for d in trend.index],
                              **{f"type_{t}": [_r(v) for v in trend[t]] for t in trend.columns}},
        "holiday_impact": [{"holiday": r.holiday_name, "avg_sales": _r(r.avg_sales),
                            "lift_pct": _r(r.raw_lift * 100),
                            "controlled_lift_pct": _r(r.controlled_lift * 100),
                            "weeks": int(r.weeks)} for r in events.itertuples()],
        "non_holiday_baseline": _r(lift["non_holiday_avg"]),
        "seasonal_pattern": [{"season": s, "avg_sales": _r(v)}
                             for s, v in analytics.seasonal_pattern(df).items()],
        "feature_importance": [{"feature": r.feature, "importance": _r(r.share, 6),
                                "wmae_increase": _r(r.wmae_increase)}
                               for r in res.importance.itertuples()],
        "pearson_correlation": [{"feature": f, "correlation": _r(v, 4)}
                                for f, v in corr.items() if not f.startswith("markdown")],
        "markdown_correlation": [{"markdown": f.replace("_", " ").title(),
                                  "correlation": _r(corr[f], 4)} for f in analytics.MARKDOWNS],
        "top_departments": [{"dept": f"Dept {int(d)}", "revenue": _r(v)}
                            for d, v in analytics.top_departments(df).items()],
        "evaluation": evaluation_block(res),
        "forecast": forecast_block(res),
        "meta": {
            "total_stores": int(df["store_id"].nunique()),
            "total_departments": int(df["department"].nunique()),
            "date_range": f"{_date(df[DATE_COL].min())} to {_date(df[DATE_COL].max())}",
            "years": sorted(int(y) for y in df[DATE_COL].dt.year.unique()),
        },
    }


def assistant_context(payload: dict) -> dict:
    """Compact, precomputed facts injected into the assistant's system prompt."""
    k, meta, ev = payload["kpis"], payload["meta"], payload["evaluation"]
    by_name = {m["model"]: m for m in ev["models"]}
    best = by_name["LightGBM + lags"]
    naive = by_name["Seasonal naive (t-52)"]
    return {
        "dataset_summary": (
            "Retail Store weekly sales dashboard. Data: an adapted version of the Kaggle "
            "'Walmart Recruiting - Store Sales Forecasting' dataset (published by Noopur Bhatt), "
            f"not the original Walmart data. {meta['total_stores']} stores, "
            f"{meta['total_departments']} departments, 4 US regions, store types A/B/C. "
            f"Period: {meta['date_range']}. Total revenue: ${k['total_sales'] / 1e9:.2f}B."
        ),
        "kpis": {
            "total_sales": f"${k['total_sales'] / 1e9:.2f}B",
            "avg_weekly_sales": f"${k['avg_weekly_sales'] / 1e3:.1f}K per store-department-week",
            "holiday_sales_lift": (f"+{k['holiday_lift_pct']:.1f}% raw, "
                                   f"+{k['holiday_lift_controlled_pct']:.1f}% controlled for "
                                   "store x department"),
            "max_store_revenue": f"${k['max_store_revenue'] / 1e6:.1f}M (best single store)",
        },
        "store_types": {t["type"]: {"description": f"{t['label']} stores",
                                    "count": t["num_stores"],
                                    "share_revenue": f"{t['share'] * 100:.1f}%"}
                        for t in payload["by_type"]},
        "regions": {r["region"]: f"${r['revenue'] / 1e9:.2f}B" for r in payload["by_region"]},
        "holiday_lift_by_event": {h["holiday"]: f"+{h['controlled_lift_pct']:.1f}%"
                                  for h in payload["holiday_impact"]},
        "seasons": [s["season"] for s in payload["seasonal_pattern"]],
        "top_features": [f["feature"] for f in payload["feature_importance"][:5]],
        "model_evaluation": {
            "protocol": (f"Time-based holdout: trained before {ev['holdout']['test_start']}, "
                         f"tested {ev['holdout']['test_start']} to {ev['holdout']['test_end']}, "
                         "one-week-ahead."),
            "best_model": "LightGBM with lag/rolling features",
            "wmae": round(best["wmae"]),
            "wape_pct": round(best["wape"] * 100, 1),
            "bias_pct": round(best["bias"] * 100, 1),
            "seasonal_naive_wmae": round(naive["wmae"]),
            "skill_vs_seasonal_naive_pct": round(best["skill_vs_naive"] * 100, 1),
            "interval_coverage_80_pct": round(ev["interval"]["coverage"] * 100, 1),
        },
        "forecast_next_weeks": [
            {"week": d, "total_forecast": f"${v / 1e6:.1f}M"}
            for d, v in zip(payload["forecast"]["dates"], payload["forecast"]["forecast"],
                            strict=True)
        ],
        "top_departments": [d["dept"] for d in payload["top_departments"]],
    }


def write_json(obj: dict, path: Path, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent, allow_nan=False, default=_json_default)
        fh.write("\n")


def _json_default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")
