"""
Retil Store — Data Generation Pipeline
Processes raw Kaggle CSVs → dashboard_data.json + rag_context.json
Run once before deployment: python generate_data.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import json
import warnings
import os

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_kaggle")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data …")
sales    = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
stores   = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))

sales["date"]    = pd.to_datetime(sales["date"])
features["date"] = pd.to_datetime(features["date"])

# ── Master merge ──────────────────────────────────────────────────────────────
df = (sales
      .merge(stores,   on="store_id")
      .merge(features, on=["store_id", "date"], suffixes=("", "_feat")))

df["year"]   = df["date"].dt.year
df["month"]  = df["date"].dt.to_period("M").astype(str)
df["week"]   = df["date"].dt.isocalendar().week.astype(int)
df["week_num"] = (df["date"] - df["date"].min()).dt.days // 7

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_sales    = float(df["weekly_sales"].sum())
avg_weekly     = float(df["weekly_sales"].mean())
hol_avg        = float(df[df["is_holiday"] == 1]["weekly_sales"].mean())
non_hol_avg    = float(df[df["is_holiday"] == 0]["weekly_sales"].mean())
holiday_lift   = float((hol_avg - non_hol_avg) / non_hol_avg * 100)
max_store_rev  = float(df.groupby("store_id")["weekly_sales"].sum().max())

# ── Monthly Revenue Time Series ───────────────────────────────────────────────
monthly = (df.groupby("month")["weekly_sales"]
             .sum()
             .reset_index()
             .rename(columns={"weekly_sales": "revenue"}))
monthly = monthly.sort_values("month")

# ── Store Weekly (for dynamic filtering in JS) ─────────────────────────────────
store_weekly = (df.groupby(["date", "store_type", "region", "season", "is_holiday", "year"])
                  ["weekly_sales"].sum()
                  .reset_index())
store_weekly["date"] = store_weekly["date"].dt.strftime("%Y-%m-%d")
store_weekly = store_weekly.rename(columns={"weekly_sales": "revenue"})

# ── Revenue by Store Type ─────────────────────────────────────────────────────
by_type = (df.groupby("store_type")
             .agg(revenue=("weekly_sales", "sum"),
                  avg_week=("weekly_sales", "mean"),
                  count=("store_id", "count"))
             .reset_index())
# count unique stores per type
store_counts = stores.groupby("store_type")["store_id"].nunique().reset_index()
store_counts.columns = ["store_type", "num_stores"]
by_type = by_type.merge(store_counts, on="store_type")
by_type = by_type.sort_values("revenue", ascending=False)

# ── Revenue by Region ─────────────────────────────────────────────────────────
by_region = (df.groupby("region")["weekly_sales"]
               .sum()
               .reset_index()
               .rename(columns={"weekly_sales": "revenue"})
               .sort_values("revenue", ascending=False))

# ── Weekly Sales Trend by Type (first 60 weeks) ───────────────────────────────
weekly_type = (df.groupby(["week_num", "store_type"])["weekly_sales"]
                 .sum()
                 .reset_index())
weekly_type = weekly_type[weekly_type["week_num"] < 60].sort_values("week_num")
weekly_pivot = weekly_type.pivot(index="week_num", columns="store_type", values="weekly_sales").fillna(0)
weekly_pivot.columns = [f"type_{c}" for c in weekly_pivot.columns]

# Week labels — find actual dates
week_dates = (df.groupby("week_num")["date"].min().reset_index())
week_dates = week_dates[week_dates["week_num"] < 60].sort_values("week_num")
weekly_labels = week_dates["date"].dt.strftime("%Y-%m-%d").tolist()
weekly_A = weekly_pivot["type_A"].tolist() if "type_A" in weekly_pivot else []
weekly_B = weekly_pivot["type_B"].tolist() if "type_B" in weekly_pivot else []
weekly_C = weekly_pivot["type_C"].tolist() if "type_C" in weekly_pivot else []

# ── Holiday Impact on Sales ───────────────────────────────────────────────────
hol_df = df[df["holiday_name"].notna()].copy()
holiday_impact = (hol_df.groupby("holiday_name")["weekly_sales"]
                         .mean()
                         .reset_index()
                         .rename(columns={"weekly_sales": "avg_sales"})
                         .sort_values("avg_sales", ascending=False))
non_hol_baseline = non_hol_avg
holiday_impact["lift_pct"] = (holiday_impact["avg_sales"] - non_hol_baseline) / non_hol_baseline * 100

# ── Seasonal Pattern (radar) ──────────────────────────────────────────────────
season_order = ["Spring", "Summer", "Fall", "Winter"]
seasonal = (df.groupby("season")["weekly_sales"]
              .mean()
              .reindex(season_order)
              .reset_index()
              .rename(columns={"weekly_sales": "avg_sales"}))

# ── Feature Engineering & ML ─────────────────────────────────────────────────
print("Training ML models …")
ml_df = df.copy()
le_type   = LabelEncoder()
le_region = LabelEncoder()
le_season = LabelEncoder()

ml_df["type_enc"]   = le_type.fit_transform(ml_df["store_type"])
ml_df["region_enc"] = le_region.fit_transform(ml_df["region"])
ml_df["season_enc"] = le_season.fit_transform(ml_df["season"])
ml_df["week_of_year"] = ml_df["date"].dt.isocalendar().week.astype(int)
ml_df["month_num"] = ml_df["date"].dt.month

mk_cols = ["markdown_1", "markdown_2", "markdown_3", "markdown_4", "markdown_5"]
for c in mk_cols:
    ml_df[c] = ml_df[c].fillna(0)

FEATURES = ["store_size", "type_enc", "region_enc", "season_enc",
            "is_holiday", "temperature", "fuel_price", "cpi",
            "unemployment", "week_of_year", "month_num",
            "markdown_1", "markdown_2", "markdown_3", "markdown_4", "markdown_5"]

X = ml_df[FEATURES].fillna(0)
y = ml_df["weekly_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = float(mean_absolute_error(y_test, rf_pred))
rf_r2   = float(r2_score(y_test, rf_pred))
rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

gb_mae  = float(mean_absolute_error(y_test, gb_pred))
gb_r2   = float(r2_score(y_test, gb_pred))
gb_rmse = float(np.sqrt(mean_squared_error(y_test, gb_pred)))

# Feature Importance
feat_imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

# Pearson Correlation with Weekly Sales
numeric_cols = ["store_size", "is_holiday", "temperature", "fuel_price",
                "cpi", "unemployment", "week_of_year", "month_num",
                "markdown_1", "markdown_2", "markdown_3", "markdown_4", "markdown_5"]
pearson = ml_df[numeric_cols + ["weekly_sales"]].corr()["weekly_sales"].drop("weekly_sales")
pearson = pearson.sort_values(ascending=False).reset_index()
pearson.columns = ["feature", "correlation"]

# ── Markdown Channel Correlation ──────────────────────────────────────────────
markdown_corr = ml_df[mk_cols + ["weekly_sales"]].corr()["weekly_sales"].drop("weekly_sales")
markdown_corr = markdown_corr.reset_index()
markdown_corr.columns = ["markdown", "correlation"]
markdown_corr["markdown"] = markdown_corr["markdown"].str.replace("_", " ").str.title()

# ── Top 10 Departments by Revenue ────────────────────────────────────────────
top_depts = (df.groupby("department")["weekly_sales"]
               .sum()
               .sort_values(ascending=False)
               .head(10)
               .reset_index()
               .rename(columns={"weekly_sales": "revenue"}))

# ── Actual vs Predicted (sampled for scatter) ─────────────────────────────────
sample_idx = np.random.RandomState(42).choice(len(y_test), size=200, replace=False)
actual_sample   = y_test.values[sample_idx].tolist()
predicted_rf    = rf_pred[sample_idx].tolist()
predicted_gb    = gb_pred[sample_idx].tolist()

# ── Revenue by Region (detailed, with min/max bands for chart labels) ─────────
region_detail = (df.groupby("region")
                   .agg(revenue=("weekly_sales","sum"),
                        avg_week=("weekly_sales","mean"),
                        stores=("store_id","nunique"))
                   .reset_index()
                   .sort_values("revenue", ascending=False))

# ── Assemble JSON ─────────────────────────────────────────────────────────────
def fmt(val, divisor=1):
    return round(float(val) / divisor, 2)

dashboard_data = {
    "kpis": {
        "total_sales":       fmt(total_sales),
        "avg_weekly_sales":  fmt(avg_weekly),
        "holiday_lift_pct":  fmt(holiday_lift),
        "max_store_revenue": fmt(max_store_rev),
    },
    "monthly_revenue": {
        "labels":  monthly["month"].tolist(),
        "revenue": [fmt(v) for v in monthly["revenue"]],
    },
    "store_weekly": store_weekly.to_dict(orient="records"),
    "by_type": [
        {
            "type":       row["store_type"],
            "revenue":    fmt(row["revenue"]),
            "avg_week":   fmt(row["avg_week"]),
            "num_stores": int(row["num_stores"]),
        }
        for _, row in by_type.iterrows()
    ],
    "by_region": [
        {"region": row["region"], "revenue": fmt(row["revenue"]),
         "avg_week": fmt(row["avg_week"]), "stores": int(row["stores"])}
        for _, row in region_detail.iterrows()
    ],
    "weekly_type_trend": {
        "labels":  weekly_labels,
        "type_A":  [fmt(v) for v in weekly_A],
        "type_B":  [fmt(v) for v in weekly_B],
        "type_C":  [fmt(v) for v in weekly_C],
    },
    "holiday_impact": [
        {"holiday": row["holiday_name"],
         "avg_sales": fmt(row["avg_sales"]),
         "lift_pct": fmt(row["lift_pct"])}
        for _, row in holiday_impact.iterrows()
    ],
    "non_holiday_baseline": fmt(non_hol_avg),
    "seasonal_pattern": [
        {"season": row["season"], "avg_sales": fmt(row["avg_sales"])}
        for _, row in seasonal.iterrows()
    ],
    "feature_importance": [
        {"feature": row["feature"], "importance": round(float(row["importance"]), 6)}
        for _, row in feat_imp.iterrows()
    ],
    "pearson_correlation": [
        {"feature": row["feature"], "correlation": round(float(row["correlation"]), 4)}
        for _, row in pearson.iterrows()
    ],
    "markdown_correlation": [
        {"markdown": row["markdown"], "correlation": round(float(row["correlation"]), 4)}
        for _, row in markdown_corr.iterrows()
    ],
    "top_departments": [
        {"dept": f"Dept {int(row['department'])}", "revenue": fmt(row["revenue"])}
        for _, row in top_depts.iterrows()
    ],
    "ml_models": {
        "random_forest": {
            "r2":   round(rf_r2,   4),
            "mae":  round(rf_mae,  2),
            "rmse": round(rf_rmse, 2),
        },
        "gradient_boosting": {
            "r2":   round(gb_r2,   4),
            "mae":  round(gb_mae,  2),
            "rmse": round(gb_rmse, 2),
        },
    },
    "actual_vs_predicted": {
        "actual":        [round(v, 2) for v in actual_sample],
        "predicted_rf":  [round(v, 2) for v in predicted_rf],
        "predicted_gb":  [round(v, 2) for v in predicted_gb],
    },
    "meta": {
        "total_stores":      int(stores["store_id"].nunique()),
        "total_departments": int(sales["department"].nunique()),
        "date_range":        f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "years":             sorted([int(y) for y in df["year"].unique()]),
    },
}

out_path = os.path.join(OUT_DIR, "dashboard_data.json")
with open(out_path, "w") as f:
    json.dump(dashboard_data, f, allow_nan=False)
print(f"Saved: {out_path}")

# ── RAG Context ───────────────────────────────────────────────────────────────
rag = {
    "dataset_summary": (
        "Retil Store Sales Forecasting Dashboard — based on the Walmart Recruiting Store Sales Forecasting "
        "dataset adapted by Noopur Bhatt. The dataset covers 50 Walmart retail stores across 4 US regions "
        "(East, North, South, West) with store types A, B, and C. "
        f"Period: {dashboard_data['meta']['date_range']}. "
        f"Total revenue: ${total_sales/1e9:.2f}B across {int(sales['weekly_sales'].count()):,} weekly sales records "
        f"spanning {int(sales['department'].nunique())} departments."
    ),
    "kpis": {
        "total_sales":         f"${total_sales/1e9:.2f}B",
        "avg_weekly_sales":    f"${avg_weekly/1e3:.1f}K per store-department",
        "holiday_sales_lift":  f"+{holiday_lift:.1f}% above non-holiday baseline",
        "max_store_revenue":   f"${max_store_rev/1e6:.1f}M (best-performing single store)",
    },
    "store_types": {
        "A": {"description": "Large-format stores", "count": 26, "share_revenue": "73.7%"},
        "B": {"description": "Medium-format stores", "count": 16, "share_revenue": "21.7%"},
        "C": {"description": "Small-format stores",  "count":  8, "share_revenue": "4.6%"},
    },
    "regions": {
        "East":  f"${float(by_region[by_region['region']=='East']['revenue'].values[0])/1e9:.2f}B",
        "North": f"${float(by_region[by_region['region']=='North']['revenue'].values[0])/1e9:.2f}B",
        "South": f"${float(by_region[by_region['region']=='South']['revenue'].values[0])/1e9:.2f}B",
        "West":  f"${float(by_region[by_region['region']=='West']['revenue'].values[0])/1e9:.2f}B",
    },
    "holidays": [r["holiday_name"] for _, r in holiday_impact.iterrows()],
    "holiday_lift_by_event": {
        row["holiday_name"]: f"+{row['lift_pct']:.1f}%" for _, row in holiday_impact.iterrows()
    },
    "seasons": season_order,
    "top_features": feat_imp["feature"].head(5).tolist(),
    "ml_performance": {
        "random_forest":     {"R2": rf_r2,  "MAE": rf_mae,  "RMSE": rf_rmse},
        "gradient_boosting": {"R2": gb_r2,  "MAE": gb_mae,  "RMSE": gb_rmse},
    },
    "top_departments": [f"Dept {int(r['department'])}" for _, r in top_depts.iterrows()],
}

rag_path = os.path.join(OUT_DIR, "rag_context.json")
with open(rag_path, "w") as f:
    json.dump(rag, f, indent=2, allow_nan=False)
print(f"Saved: {rag_path}")
print("Done.")
