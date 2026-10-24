import numpy as np
import pytest

from retail_forecast.features import build_features, feature_columns
from retail_forecast.models import LGBMForecaster, RandomForestForecaster, holiday_weights
from retail_forecast.validation import temporal_split


@pytest.fixture
def split(panel):
    return temporal_split(build_features(panel, 1), "2023-10-01", "2024-01-01")


def test_holiday_weights(panel):
    w = holiday_weights(panel)
    assert set(np.unique(w)) == {1.0, 5.0}


def test_lgbm_fits_with_early_stopping_and_predicts(split):
    model = LGBMForecaster(feature_columns(1), params={"objective": "l2", "num_leaves": 7,
                                                       "min_child_samples": 5}).fit(
        split.train, split.valid)
    assert model.n_estimators >= 1
    pred = model.predict(split.test)
    assert pred.shape == (len(split.test),)
    assert np.isfinite(pred).all()


def test_lgbm_refit_needs_rounds(split):
    with pytest.raises(ValueError):
        LGBMForecaster(feature_columns(1)).fit(split.train)


def test_random_forest_handles_categoricals_and_nans(split):
    pred = RandomForestForecaster(feature_columns(1)).fit(split.train).predict(split.test)
    assert np.isfinite(pred).all()
