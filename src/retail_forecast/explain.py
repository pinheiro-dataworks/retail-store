"""Model-agnostic feature importance.

v1 plotted Random Forest MDI (impurity) importances, which are computed on the
training data and favour high-cardinality numeric columns. Permutation
importance measures how much the holdout WMAE gets worse when one column is
shuffled, so it reflects what the model actually relies on out of sample.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from . import metrics
from .config import SEED, TARGET


def permutation_importance(predict: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                           features: list[str], n_repeats: int = 3,
                           seed: int = SEED) -> pd.DataFrame:
    """Increase in WMAE when each feature is shuffled (mean and std over repeats)."""
    rng = np.random.default_rng(seed)
    y, hol = df[TARGET].to_numpy(), df["is_holiday"].to_numpy()
    baseline = metrics.wmae(y, predict(df), hol)

    rows = []
    for col in features:
        deltas = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[col] = df[col].to_numpy()[rng.permutation(len(df))]
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                shuffled[col] = pd.Categorical(shuffled[col], categories=df[col].cat.categories)
            deltas.append(metrics.wmae(y, predict(shuffled), hol) - baseline)
        rows.append({"feature": col, "wmae_increase": float(np.mean(deltas)),
                     "std": float(np.std(deltas))})

    out = pd.DataFrame(rows).sort_values("wmae_increase", ascending=False)
    out["share"] = out["wmae_increase"].clip(lower=0) / out["wmae_increase"].clip(lower=0).sum()
    out.attrs["baseline_wmae"] = baseline
    return out.reset_index(drop=True)
