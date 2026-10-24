import pandas as pd
import pytest

from retail_forecast.validation import expanding_window_folds, iter_fold_frames, temporal_split


def test_temporal_split_has_no_overlap(panel):
    split = temporal_split(panel, "2023-06-01", "2024-01-01")
    assert split.train["date"].max() < split.valid["date"].min()
    assert split.valid["date"].max() < split.test["date"].min()
    assert len(split.train) + len(split.valid) + len(split.test) == len(panel)


def test_temporal_split_rejects_bad_order(panel):
    with pytest.raises(ValueError):
        temporal_split(panel, "2024-01-01", "2023-01-01")


def test_expanding_folds_are_contiguous_and_in_the_past(panel):
    folds = expanding_window_folds(panel["date"], n_folds=3, test_weeks=8)
    assert len(folds) == 3
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert prev.test_end == nxt.test_start
        assert prev.train_end < nxt.train_end
    for _, train, test in iter_fold_frames(panel, folds):
        assert train["date"].max() < test["date"].min()
        assert test["date"].nunique() == 8


def test_expanding_folds_respect_end(panel):
    end = pd.Timestamp("2023-12-01")
    folds = expanding_window_folds(panel["date"], n_folds=2, test_weeks=4, end=end)
    for _, _, test in iter_fold_frames(panel, folds):
        assert test["date"].max() < end


def test_too_many_folds_raises(panel):
    with pytest.raises(ValueError):
        expanding_window_folds(panel["date"], n_folds=50, test_weeks=10)
