import numpy as np

from retail_forecast.explain import permutation_importance


def test_only_the_used_feature_matters(panel):
    df = panel.copy()
    df["noise"] = np.random.default_rng(1).normal(size=len(df))

    def predict(frame):  # "model" that only looks at store_size
        return frame["store_size"].to_numpy() / 4

    imp = permutation_importance(predict, df, ["store_size", "noise"], n_repeats=2)
    assert imp.loc[0, "feature"] == "store_size"
    assert imp.loc[0, "wmae_increase"] > 0
    assert imp.set_index("feature").loc["noise", "wmae_increase"] == 0
    assert imp["share"].sum() == 1
