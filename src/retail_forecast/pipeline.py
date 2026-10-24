"""End-to-end run: tune, evaluate on the holdout, forecast, and collect results."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines, metrics
from .config import DATE_COL, TARGET, TEST_START
from .explain import permutation_importance
from .features import build_features, feature_columns
from .forecast import aggregate_band, evaluate_horizons, forecast_future
from .models import LGBMForecaster, QuantileInterval, RandomForestForecaster
from .tuning import SEARCH_SPACE, tune
from .validation import temporal_split

log = logging.getLogger(__name__)

HORIZONS = range(1, 9)
QUICK_SPACE = {"objective": ["l2"], "num_leaves": [31], "min_child_samples": [200]}

# Feature set of the original v1 model, kept to show what the new features add.
V1_FEATURES = ["store_size", "store_type", "region", "season", "is_holiday", "temperature",
               "fuel_price", "cpi", "unemployment", "week_of_year", "month",
               "markdown_1", "markdown_2", "markdown_3", "markdown_4", "markdown_5"]


@dataclass
class RunResult:
    panel: pd.DataFrame
    test: pd.DataFrame  # holdout rows with every model's prediction as a column
    model_table: pd.DataFrame
    ablation: dict
    interval: dict
    importance: pd.DataFrame
    tuning_table: pd.DataFrame
    best_params: dict
    horizon_table: pd.DataFrame
    future: pd.DataFrame
    future_total: pd.DataFrame


def _score(name: str, kind: str, test: pd.DataFrame, pred: np.ndarray) -> dict:
    return {"model": name, "kind": kind,
            **metrics.evaluate(test[TARGET], pred, test["is_holiday"])}


def run(panel: pd.DataFrame, quick: bool = False) -> RunResult:
    feats = build_features(panel, horizon=1)
    feats["season"] = feats["season"].astype("category")
    cols = feature_columns(1)
    split = temporal_split(feats)
    pre_test = feats[feats[DATE_COL] < pd.Timestamp(TEST_START)]
    test = split.test.copy()

    log.info("Tuning LightGBM on %d backtest candidates", 1 if quick else 12)
    tuning = tune(pre_test, cols, space=QUICK_SPACE if quick else SEARCH_SPACE)
    params = tuning.best_params
    log.info("Best params: %s", params)

    log.info("Fitting models for the H2 2024 holdout")
    lgbm_es = LGBMForecaster(cols, params=params).fit(split.train, split.valid)
    lgbm = LGBMForecaster(cols, params=params, n_estimators=lgbm_es.n_estimators).fit(pre_test)
    test["pred_lightgbm"] = lgbm.predict(test)
    test["pred_random_forest"] = RandomForestForecaster(cols).fit(pre_test).predict(test)
    test["pred_rf_v1"] = RandomForestForecaster(V1_FEATURES).fit(pre_test).predict(test)
    test["pred_seasonal_naive"] = baselines.seasonal_naive(panel, test)
    test["pred_last_value"] = baselines.last_value(panel, test, horizon=1)
    test["pred_historical_mean"] = baselines.historical_mean(panel, test, TEST_START)

    rows = [
        _score("Seasonal naive (t-52)", "baseline", test, test["pred_seasonal_naive"]),
        _score("Last value (t-1)", "baseline", test, test["pred_last_value"]),
        _score("Store x dept mean", "baseline", test, test["pred_historical_mean"]),
        _score("Random Forest (v1 features)", "model", test, test["pred_rf_v1"]),
        _score("Random Forest + lags", "model", test, test["pred_random_forest"]),
        _score("LightGBM + lags", "model", test, test["pred_lightgbm"]),
    ]
    table = pd.DataFrame(rows)
    naive = table.loc[table["model"].str.startswith("Seasonal naive"), "wmae"].item()
    table["skill_vs_naive"] = 1 - table["wmae"] / naive

    log.info("Ablation: adding the exogenous columns back")
    exo_cols = feature_columns(1, include_exogenous=True)
    exo_es = LGBMForecaster(exo_cols, params=params).fit(split.train, split.valid)
    exo = LGBMForecaster(exo_cols, params=params, n_estimators=exo_es.n_estimators).fit(pre_test)
    exo_wmae = metrics.wmae(test[TARGET], exo.predict(test), test["is_holiday"])
    base_wmae = table.loc[table["model"] == "LightGBM + lags", "wmae"].item()
    ablation = {"without_exogenous_wmae": base_wmae, "with_exogenous_wmae": exo_wmae}

    log.info("Fitting quantile models for the 80% interval")
    qi = QuantileInterval(cols, params=params).fit(pre_test)
    test["lower"], test["upper"] = qi.predict(test)
    hol = test["is_holiday"] == 1
    interval = {
        "nominal": qi.nominal_coverage,
        "coverage": metrics.interval_coverage(test[TARGET], test["lower"], test["upper"]),
        "coverage_holiday": metrics.interval_coverage(
            test.loc[hol, TARGET], test.loc[hol, "lower"], test.loc[hol, "upper"]),
    }

    log.info("Permutation importance on the holdout")
    importance = permutation_importance(lgbm.predict, test, cols)

    log.info("Direct multi-horizon evaluation and forecast (h = 1..%d)", max(HORIZONS))
    horizon_table, rounds, agg_errors = evaluate_horizons(panel, params, HORIZONS)
    future = forecast_future(panel, params, rounds)
    future_total = aggregate_band(future, agg_errors)

    return RunResult(panel=panel, test=test, model_table=table, ablation=ablation,
                     interval=interval, importance=importance,
                     tuning_table=tuning.table, best_params=params,
                     horizon_table=horizon_table, future=future, future_total=future_total)
