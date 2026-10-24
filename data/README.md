# Data

Raw files live in `data/raw/` and are **not** committed. Get them with:

```bash
python scripts/download_data.py --dataset <owner>/<dataset-slug>   # Kaggle API
python scripts/download_data.py --verify-only                     # files copied by hand
```

| File | Rows | Grain |
|---|---|---|
| `sales.csv` | 156,000 | store × department × week |
| `stores.csv` | 50 | store |
| `features.csv` | 7,800 | store × week |

**Provenance.** This is an *adapted* version of the Kaggle competition
[Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting),
published by [Noopur Bhatt](https://www.kaggle.com/noopurbhatt). It is not the
original Walmart data: dates were shifted to 2022-01-01 → 2024-12-21, there
are 50 stores instead of 45 and 20 departments instead of 81+.

**Known quirks** (see `notebooks/01_eda.ipynb`):

- Temperature, fuel price, CPI, unemployment and markdowns behave like
  independent draws week to week (lag-1 autocorrelation ≈ 0). They are kept
  in the EDA but carry no forecasting signal.
- Markdowns have no missing values, unlike the original competition data.
- Holiday weeks are not perfectly regular (e.g. Black Friday is tagged on two
  weeks in 2024).

The script checks SHA-256 hashes so you know you are training on the same
files used for the published metrics.
