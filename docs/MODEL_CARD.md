# Model card: Retail Store weekly sales forecaster

## Overview

| | |
|---|---|
| Task | Forecast weekly sales for each store × department, 1 to 8 weeks ahead |
| Model | LightGBM regressor, one model per horizon (direct strategy) |
| Version | 2.0.0 |
| Code | `src/retail_forecast/` (`python -m retail_forecast`) |
| Owner | Renan Pinheiro |

## Intended use

Supports weekly planning decisions such as inventory, replenishment and
staffing, at the store × department level or aggregated. It is a portfolio
project built on an adapted public dataset. It is **not** validated for any
real retailer.

Not intended for: individual SKUs, daily granularity, stores or departments
that are not in the training data, or horizons beyond 8 weeks.

## Data

Adapted version of the Kaggle *Walmart Recruiting — Store Sales Forecasting*
data (Noopur Bhatt): 50 stores × 20 departments × 156 weeks
(2022-01-01 → 2024-12-21), 156,000 rows. Store attributes: type (A/B/C), size and
region. See [`data/README.md`](../data/README.md) for provenance and checksums.

The macro and promotion columns (temperature, fuel price, CPI, unemployment,
markdowns 1–5) behave like independent random draws: their lag-1
autocorrelation is about 0, and they do not correlate with the part of sales
that the series level and the calendar leave unexplained. Markdowns *look*
correlated with sales (r ≈ 0.10), but only because they run about 2.2× higher
in holiday weeks. Restricted to non-holiday weeks, r ≈ 0.00.

## Features

All history features for a row at week *t* come from sales up to *t − h*,
where *h* is the forecast horizon. A unit test enforces this.

| Group | Features |
|---|---|
| Identity | `store_id`, `department`, `store_type`, `region`, `store_size` |
| Calendar | `is_holiday`, `holiday_name`, `week_of_year`, `month` |
| History | `lag_h`, `lag_h+1`, `lag_h+3`, `lag_52`, `roll_mean_4`, `roll_mean_13`, `roll_std_4`, `expanding_mean` |

The exogenous columns are excluded. Adding them back gives a holdout WMAE of
31,852 against 31,850 without them.

## Training and validation

- **Split by date.** Training runs through 2023-12-31, validation (early
  stopping) covers 2024-01-01 → 2024-06-30, and the test set is
  2024-07-01 → 2024-12-21 (25 weeks, 25,000 rows). The test set was scored
  once per configuration reported here.
- **Hyperparameter search.** 12 candidates (objective, `num_leaves`,
  `min_child_samples`), scored on three 13-week expanding-window folds that
  end before the test period. Inside each fold, the number of boosting rounds
  is chosen on a separate inner validation window. All candidates landed
  within 1% of each other.
  Selected: `l2`, 31 leaves, `min_child_samples=200`, learning rate 0.03.
- **Loss weighting.** Holiday weeks get a weight of 5 in training, matching
  the evaluation metric.
- **Final model.** Refit on everything before the test period, using the
  number of rounds chosen by early stopping.

## Metrics (test set, one week ahead)

| Model | WMAE | WAPE | Bias | Skill vs. naive |
|---|---|---|---|---|
| Seasonal naive, y(t−52) | 42,545 | 58.1% | −3.8% | 0% |
| Last value, y(t−1) | 42,877 | 59.7% | −2.9% | −0.8% |
| Store × dept historical mean | 36,023 | 45.6% | −9.7% | 15.3% |
| Random Forest, v1 features | 33,639 | 45.6% | −0.1% | 20.9% |
| Random Forest + lags | 31,965 | 43.2% | +1.1% | 24.9% |
| **LightGBM + lags** | **31,850** | **43.0%** | **+0.5%** | **25.1%** |

- **WMAE** is MAE with holiday weeks weighted 5×; it was the official
  metric of the source competition. **WAPE** = Σ|y − ŷ| / Σy. **Bias** =
  Σ(ŷ − y) / Σy, so a positive value means over-forecasting.
- **Skill** = 1 − WMAE / WMAE(seasonal naive).
- **By week type:** MAE is 23.4K on normal weeks (naive: 31.9K) and 38.6K
  on holiday weeks (naive: 51.1K).
- **By store type:** WAPE is 42.8% for A, 43.5% for B and 44.0% for C.
- **By horizon:** WMAE stays between 31.8K and 31.9K for every horizon from
  1 to 8 weeks, with about 25% skill at each one.
- **80% interval** (quantile LightGBM, α = 0.1/0.9): empirical coverage is
  78.6% overall and 78.8% in holiday weeks.

For comparison, v1 reported MAE 25.6K and R² 0.43 using a shuffled split.
Under the time-based split, the same v1 models score MAE 28.0K.

## Feature importance

Permutation importance on the test set, measured as the WMAE increase when a
feature is shuffled:

| Feature | WMAE increase |
|---|---|
| `expanding_mean` | +7,062 |
| `holiday_name` | +3,424 |
| `store_id` | +2,573 |
| `department` | +756 |
| `store_size` | +601 |
| `roll_mean_13` | +516 |

## Limitations

- **High noise floor.** About 50% of the variance in weekly sales is
  week-to-week noise around each series' level, which caps WAPE at roughly
  43% for any model in this setup. On this data, the 25% skill over the
  seasonal naive forecast is close to what can be achieved.
- **Little short-term signal.** Recent lags add little. Most of the gain
  comes from estimating each series' level and its holiday response. Do not
  expect the model to react to sudden shifts in demand.
- **Holiday calendar.** Holiday flags for future weeks are hard-coded
  (2024-12-28 Christmas, 2025-01-04 New Year) and follow the data's own
  irregular tagging.
- **Aggregate band.** The forecast band for total sales is the empirical
  10th–90th percentile of the aggregate errors on the test set (25 weeks per
  horizon). It is indicative only, and less reliable in holiday weeks.
- **Synthetic exogenous data.** Because the macro and promotion columns are
  noise here, this project cannot say anything about promotion or price
  effects.
- **Single test period.** Backtesting covered H1 2024 only, and the test set
  covers H2 2024 only.

## Reproducing

```bash
make setup && make data && make train   # about 2.5 minutes on a laptop CPU
make test
```
