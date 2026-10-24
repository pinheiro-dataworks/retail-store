# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **Assistant model.** Groq retired `llama-3.3-70b-versatile` on 2026-08-16,
  so every chat request failed. The endpoint now uses `openai/gpt-oss-120b`
  with low reasoning effort, and the model id can be changed with the
  `GROQ_MODEL` variable.

## [2.0.0] - 2026-10-24

This release rebuilds the forecasting methodology, moves the code into a
tested package and makes the dashboard show how the model was evaluated and
what it forecasts.

### Fixed
- **Temporal leakage.** The model was evaluated with a shuffled
  `train_test_split` on time-series data. Models are now trained and scored
  on date-based splits; under that split, the v1 models' MAE goes from 25.6K
  to 28.0K.
- **Typo.** "Retil" is now "Retail" in the UI, logo, prompts, package name
  and docs.
- **Demo link.** The README demo link pointed to a 404; it now matches the
  live deployment.
- **Misleading charts.**
  - Markdown correlation was a holiday confounder: r ≈ 0.10 overall, ≈ 0.00
    in non-holiday weeks.
  - The seasonal radar axis did not start at zero.
  - Holiday lift is now labelled raw vs. controlled.

### Added
- **`retail_forecast` package**
  - Panel loader with integrity checks.
  - Horizon-aware lag, rolling and expanding-mean features.
  - Baselines: seasonal naive, last value and historical mean.
  - Metrics: WMAE, WAPE, bias, skill score and interval coverage.
  - Backtest folds and hyperparameter tuning.
  - LightGBM with a holiday-weighted loss.
  - Quantile intervals.
  - Permutation importance.
  - Direct multi-horizon forecasting (1–8 weeks).
- **CLI.** `python -m retail_forecast` replaces `generate_data.py`.
- **Dashboard: model evaluation section.**
  - Metric cards and a comparison table against the baselines.
  - Actual vs. forecast over the holdout.
  - Error by week.
  - Residual histogram.
  - Error by horizon.
- **Dashboard: forecast section.** The next 8 weeks with an 80% band.
- **Assistant.**
  - Server-side Groq key, with bring-your-own-key as a fallback.
  - Per-IP rate limit, input cap and CORS allowlist.
  - Node tests.
  - An evaluation question set and runner.
- **Tests and CI.**
  - 61 pytest tests, including a leakage test and a JSON contract test.
  - 12 node:test tests.
  - GitHub Actions running ruff, pytest and the node tests.
- **Reproducibility.**
  - Pinned requirements and `pyproject.toml`.
  - Makefile.
  - Kaggle download script with SHA-256 checks.
  - `.gitattributes` to normalise line endings.
- **Documentation.**
  - Executed EDA notebook.
  - Model card.
  - Rewritten project manual and README.
  - Screenshots and a GIF.
  - MIT license.

### Changed
- **Folder layout.** The deployable site moved from `retil_store/` to
  `web/`, so the Vercel root directory must be `web`.
- **Assistant naming.** It is described as grounded prompting, not RAG, and
  its context file is now `assistant_context.json`.
- **Feature importance.** Shown as permutation importance on the holdout
  instead of Random Forest MDI.
- **Exogenous columns.** Macro and markdown columns are excluded from the
  model after an ablation showed no gain.

### Removed
- Unsupported claims: "built on real retail data", "GPT-4o quality", and the
  R² cards with hand-written insights.
- The random 200-point actual-vs-predicted scatter plot.

## [1.0.0] - 2026-05-31

- First version: static Chart.js dashboard, Random Forest and Gradient
  Boosting trained on a random split, and a Groq chat with an injected JSON
  context.
