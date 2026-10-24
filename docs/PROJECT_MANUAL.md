# Retail Store: project manual

This manual covers how the project fits together: the pipeline modules, what
each dashboard section shows, how the assistant works, and how to deploy.
For the model itself (metrics, limitations), see [MODEL_CARD.md](MODEL_CARD.md).
For the evidence behind the modelling choices, see
[`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).

## Contents

1. [Architecture](#1-architecture)
2. [Pipeline modules](#2-pipeline-modules)
3. [Validation protocol](#3-validation-protocol)
4. [Metrics](#4-metrics)
5. [Dashboard sections](#5-dashboard-sections)
6. [Assistant](#6-assistant)
7. [Deployment](#7-deployment)
8. [Running locally](#8-running-locally)
9. [Changing things safely](#9-changing-things-safely)

---

## 1. Architecture

```
data/raw/*.csv ──► python -m retail_forecast ──► web/data/dashboard_data.json
 (Kaggle, verified        │                       web/data/assistant_context.json
  by SHA-256)             │                       artifacts/ (tuning table, holdout
                          │                                   predictions, metrics)
                          ▼
                 web/ (static site on Vercel)
                   index.html + js/dashboard.js   reads dashboard_data.json
                   api/chat.js (Node function)    reads assistant_context.json
                                                  and calls Groq
```

The Python side runs offline, and its output is committed. Vercel only
serves static files plus one serverless function, so the deployment never
runs Python.

## 2. Pipeline modules

| Module | Responsibility |
|---|---|
| `config.py` | Paths, series keys, holiday weight, holdout dates, seed |
| `data.py` | Load the CSVs and build the store × department × week panel. Fails on duplicates, unmatched joins or gaps in the calendar |
| `features.py` | Horizon-aware lags, rolling statistics and expanding mean; categoricals with fixed categories; `feature_columns(h)` |
| `baselines.py` | Seasonal naive, last value, historical mean |
| `metrics.py` | WMAE, WAPE, bias, MAE, RMSE, R², skill score, interval coverage |
| `validation.py` | Date-based holdout and expanding-window folds |
| `models.py` | `LGBMForecaster` (holiday-weighted, early stopping), `RandomForestForecaster`, `QuantileInterval` |
| `tuning.py` | Grid search scored on backtest folds |
| `explain.py` | Permutation importance on the holdout |
| `forecast.py` | Direct multi-horizon evaluation and future forecast, plus the aggregate band |
| `analytics.py` | Descriptive aggregations for the dashboard (holiday lift raw and controlled, store-type stats, correlations) |
| `pipeline.py` | Orchestrates one full run and returns a `RunResult` |
| `export.py` | Turns a `RunResult` into the two JSON files |
| `__main__.py` | CLI (`--quick` skips the grid search) |

A full run takes about 2.5 minutes on a laptop CPU. Most of that time is the
grid search (≈ 95 s) and the eight horizon models.

## 3. Validation protocol

v1 used `train_test_split(shuffle=True)`. For a time series this trains on
weeks that come after the weeks being scored, so the error estimate is too
optimistic. Under a proper split, the v1 models' MAE goes from 25.6K to 28.0K.

The current protocol:

```
2022-01-01 ───────── 2023-12-31 │ 2024-01-01 ── 2024-06-30 │ 2024-07-01 ── 2024-12-21
          train                 │   validation             │   test (scored once)
                                │   (early stopping)       │
```

- **Tuning** uses three 13-week expanding-window folds that all end before
  2024-07-01. Each fold sets aside its last 8 training weeks to choose the
  number of boosting rounds, so a fold's evaluation weeks never influence
  that choice.
- **No leakage at the feature level.** Every history feature for week *t*
  uses sales up to *t − h* only. `tests/test_features.py` multiplies all
  later sales by 1,000 and checks that the features stay the same.
- **Horizons.** One model per horizon *h* = 1…8 (direct strategy). Each one
  is evaluated on the same test weeks and then refit on all history to
  produce the forecast.

## 4. Metrics

| Metric | Definition | Why |
|---|---|---|
| WMAE | Σ wᵢ\|yᵢ − ŷᵢ\| / Σ wᵢ, with w = 5 on holiday weeks | Official metric of the source competition; holiday misses are costlier |
| WAPE | Σ\|y − ŷ\| / Σy | Standard retail headline, robust to small weeks |
| Bias | Σ(ŷ − y) / Σy | Positive = over-forecast (excess stock), negative = stock-outs |
| Skill | 1 − WMAE / WMAE(seasonal naive) | Gain over doing nothing clever |
| MAE / RMSE / R² | Usual definitions | Kept for reference, always shown next to the baseline |
| Coverage | Share of actuals inside [q10, q90] | Checks whether the 80% interval is honest |

## 5. Dashboard sections

| Section | What it shows | Source key |
|---|---|---|
| KPIs | Total sales, average weekly sales per store-department, holiday lift (raw, with the controlled figure underneath), best store | `kpis` |
| Revenue time series | Monthly totals, re-aggregated by the filters | `monthly_revenue`, `store_weekly` |
| Store segmentation | Revenue by store type (donut and table) and by region | `by_type`, `by_region` |
| Weekly trend by type | First 60 weeks per store type | `weekly_type_trend` |
| Seasonality & holidays | Average sales per holiday week vs. non-holiday weeks; seasonal radar starting at zero | `holiday_impact`, `seasonal_pattern` |
| Feature engineering | Permutation importance (LightGBM, holdout); Pearson correlations | `feature_importance`, `pearson_correlation` |
| Markdowns & departments | Markdown correlation on all weeks vs. non-holiday weeks; top 10 departments | `markdown_correlation`, `top_departments` |
| Model evaluation | Headline metrics, comparison table vs. baselines, actual vs. forecast over the holdout, error by week, residuals, error by horizon | `evaluation` |
| Forecast | Last 26 weeks plus the next 8 weeks with an 80% band; per-store-type table | `forecast` |
| Assistant | Chat grounded on the precomputed facts | `/api/chat` |

The Year / Region / Type / Season filters re-aggregate `store_weekly` in the
browser. They update the KPIs, the monthly series, and the type and region
charts. The model sections always show the full holdout.

## 6. Assistant

**Grounded prompting, not RAG.** There is no retrieval step. The pipeline
writes a small JSON of facts (`assistant_context.json`), and
`api/_lib/prompt.js` puts all of it into the system prompt of every request.
That works while the facts fit comfortably in a prompt (about 2K characters
here). Retrieval would only be worth adding with a much larger corpus.

```
Browser ── POST /api/chat {message} ──► api/chat.js
                                          ├─ origin check (same-origin or ALLOWED_ORIGINS)
                                          ├─ validate message (≤ 1,000 chars)
                                          ├─ rate limit per IP (Upstash or in-memory)
                                          └─ Groq openai/gpt-oss-120b, temperature 0.2
```

- **API key.** `GROQ_API_KEY` is read from the Vercel environment. If it is
  not set, the page asks visitors for their own key, keeps it in memory for
  that tab only and forwards it to Groq over HTTPS. The key is never stored.
- **Tests.** `web/tests/chat.test.js` uses `node:test` and needs no
  dependencies.
- **Evaluation.** `eval/assistant_questions.json` has 13 factual questions
  and 2 that the assistant should decline.
  `python scripts/eval_assistant.py --url <deployment>/api/chat` scores a
  running deployment.

## 7. Deployment

The Vercel project points at the `web/` directory (Settings → General → Root
Directory).

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Recommended | Shared key, so visitors do not need their own |
| `GROQ_MODEL` | No (default `openai/gpt-oss-120b`) | Groq model id, so a retired model can be swapped without a code change |
| `RATE_LIMIT_PER_MINUTE` | No (default 10) | Requests per IP per minute |
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | No | Rate-limit counters shared across instances |
| `ALLOWED_ORIGINS` | No | Extra origins allowed to call the API, comma separated |

To deploy new numbers:

1. Run `make train`.
2. Commit `web/data/`.
3. Push. Vercel redeploys on its own.

## 8. Running locally

```bash
make setup                          # pinned Python deps + dev tools
make data                           # verify data/raw/ (or RETAIL_DATASET=<owner>/<slug> make data)
make train                          # pipeline -> web/data/
make test                           # pytest + node tests
make serve                          # static preview on http://localhost:8080 (no chat)
cd web && npx vercel dev            # full app including /api/chat
```

## 9. Changing things safely

- **New feature.** Add it in `features.py` and build it from the target
  shifted by `horizon`. The leakage test will catch any feature that reads
  the future.
- **New model.** Give it a `fit` / `predict` interface, add a row in
  `pipeline.run()` and it will appear in the comparison table.
- **Changed JSON shape.** Update `tests/test_web_data.py` and
  `js/dashboard.js` in the same commit.
- **Retrained model.** If the numbers move, `tests/test_assistant_eval.py`
  will point at the expected answers in `eval/` that need updating.
