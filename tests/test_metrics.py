import numpy as np
import pytest

from retail_forecast import metrics


def test_wmae_weights_holiday_weeks_five_times():
    y = [100, 100]
    p = [90, 50]  # errors of 10 on a normal week and 50 on a holiday week
    assert metrics.wmae(y, p, [0, 1]) == pytest.approx((10 + 5 * 50) / 6)


def test_wmae_without_holidays_equals_mae():
    y, p = [10, 20, 30], [12, 18, 33]
    assert metrics.wmae(y, p, [0, 0, 0]) == pytest.approx(metrics.mae(y, p))


def test_wape_and_bias():
    y, p = [100, 300], [110, 270]
    assert metrics.wape(y, p) == pytest.approx(40 / 400)
    assert metrics.bias(y, p) == pytest.approx(-20 / 400)


def test_bias_sign_means_over_forecast():
    assert metrics.bias([100], [120]) > 0
    assert metrics.bias([100], [80]) < 0


def test_skill_score():
    assert metrics.skill_score(75, 100) == pytest.approx(0.25)
    assert metrics.skill_score(100, 100) == 0
    with pytest.raises(ValueError):
        metrics.skill_score(1, 0)


def test_perfect_forecast():
    y = np.array([5.0, 7.0, 9.0])
    m = metrics.evaluate(y, y, [0, 1, 0])
    assert m["mae"] == m["wmae"] == m["wape"] == m["bias"] == 0
    assert m["r2"] == 1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.mae([1, 2], [1])


def test_interval_coverage():
    assert metrics.interval_coverage([1, 5, 10], [0, 6, 9], [2, 7, 11]) == pytest.approx(2 / 3)
