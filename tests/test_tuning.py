from retail_forecast.features import build_features, feature_columns
from retail_forecast.tuning import candidates, tune


def test_candidates_cover_the_grid():
    grid = candidates({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(grid) == 6
    assert {"a": 2, "b": "z"} in grid


def test_tune_ranks_candidates(panel):
    feats = build_features(panel, 1)
    result = tune(feats, feature_columns(1),
                  space={"num_leaves": [3, 7], "min_child_samples": [5]},
                  n_folds=2, test_weeks=6, end="2024-06-01", inner_valid_weeks=6)
    assert len(result.table) == 2
    assert result.table["wmae_mean"].is_monotonic_increasing
    assert result.best_params["num_leaves"] == result.table.loc[0, "num_leaves"]
