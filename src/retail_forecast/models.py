"""Model wrappers.

LightGBM is the main model. Random Forest stays as a reference so the
dashboard can show how much of the gain comes from the features and how much
from the learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import HOLIDAY_WEIGHT, SEED, TARGET

DEFAULT_LGBM_PARAMS = {
    "objective": "l2",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100


def holiday_weights(df: pd.DataFrame) -> np.ndarray:
    """Same weighting as WMAE, so the training loss matches the metric."""
    return np.where(df["is_holiday"].to_numpy() > 0, HOLIDAY_WEIGHT, 1.0)


@dataclass
class LGBMForecaster:
    features: list[str]
    params: dict = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    n_estimators: int | None = None  # None -> pick with early stopping
    model: lgb.LGBMRegressor | None = None

    def _make(self, n_estimators: int, **overrides) -> lgb.LGBMRegressor:
        params = {**self.params, **overrides}
        return lgb.LGBMRegressor(n_estimators=n_estimators, random_state=SEED,
                                 verbose=-1, n_jobs=-1, **params)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> LGBMForecaster:
        """Fit on `train`; if `valid` is given, early-stop on its WMAE-weighted L1."""
        train = train.dropna(subset=[TARGET])
        if valid is not None:
            self.model = self._make(MAX_ROUNDS)
            self.model.fit(
                train[self.features], train[TARGET], sample_weight=holiday_weights(train),
                eval_X=(valid[self.features],), eval_y=(valid[TARGET],),
                eval_sample_weight=[holiday_weights(valid)], eval_metric="l1",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True,
                                              verbose=False)],
            )
            self.n_estimators = int(self.model.best_iteration_)
        else:
            if self.n_estimators is None:
                raise ValueError("pass a validation frame or set n_estimators")
            self.model = self._make(self.n_estimators)
            self.model.fit(train[self.features], train[TARGET],
                           sample_weight=holiday_weights(train))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        kwargs = {"num_iteration": self.n_estimators} if self.n_estimators else {}
        return self.model.predict(df[self.features], **kwargs)


def _to_numeric(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Random Forest in scikit-learn needs numbers: categoricals -> codes, NaN -> -1."""
    out = df[features].copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes
    return out.fillna(-1)


@dataclass
class RandomForestForecaster:
    features: list[str]
    model: RandomForestRegressor | None = None

    def fit(self, train: pd.DataFrame) -> RandomForestForecaster:
        train = train.dropna(subset=[TARGET])
        self.model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                           max_features=0.5, n_jobs=-1, random_state=SEED)
        self.model.fit(_to_numeric(train, self.features), train[TARGET],
                       sample_weight=holiday_weights(train))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model.predict(_to_numeric(df, self.features))
