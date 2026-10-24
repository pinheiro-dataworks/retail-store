# Retil Store — Sales Forecasting Dashboard
## Project Manual v1.0.0

**Author**: Renan Pinheiro  
**GitHub**: [pinheiro-dataworks](https://github.com/pinheiro-dataworks)  
**LinkedIn**: [pinheirodata](https://www.linkedin.com/in/pinheirodata/)  
**Year**: 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Architecture and Project Flow](#3-architecture-and-project-flow)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [Data Pipeline — generate_data.py](#6-data-pipeline--generate_datapy)
7. [Dashboard Sections and Charts](#7-dashboard-sections-and-charts)
8. [Statistical and ML Concepts](#8-statistical-and-ml-concepts)
9. [AI Chat — RAG with Groq](#9-ai-chat--rag-with-groq)
10. [Deployment on Vercel](#10-deployment-on-vercel)
11. [How to Run Locally](#11-how-to-run-locally)

---

## 1. Project Overview

Retil Store is a fully interactive, production-ready **Sales Forecasting Dashboard** built on real retail data. It combines:

- **Exploratory data analysis (EDA)** — KPIs, time series, segmentation.
- **Seasonality and holiday intelligence** — quantifying how events boost sales.
- **Feature engineering and correlation analysis** — understanding what drives sales.
- **Machine learning** — Random Forest and Gradient Boosting forecasting models.
- **Conversational AI** — a Groq-powered assistant with Retrieval-Augmented Generation (RAG) for natural-language queries about the data.

The dashboard is fully static (HTML/CSS/JS), with a single serverless function for the AI chat, making it deployable on Vercel at zero cost.

---

## 2. Dataset

**Source**: Walmart Recruiting — Store Sales Forecasting (Kaggle), adapted by **Noopur Bhatt**.

The dataset consists of three CSV files:

| File | Rows | Key Columns |
|---|---|---|
| `sales.csv` | 156,000 | `store_id`, `department`, `date`, `weekly_sales`, `is_holiday` |
| `stores.csv` | 50 | `store_id`, `store_type`, `store_size`, `region` |
| `features.csv` | 7,800 | `store_id`, `date`, `temperature`, `fuel_price`, `markdown_1–5`, `cpi`, `unemployment`, `is_holiday`, `holiday_name`, `season` |

**Key characteristics**:
- **50 stores** across 4 US regions: East, North, South, West.
- **3 store types**: A (large), B (medium), C (small).
- **20 departments** per store, covering all product categories.
- **Date range**: 2022-01-01 to 2024-12-25 (36 months, 156,000 weekly observations).
- **5 holiday events**: New Year, Independence Day, Labor Day, Black Friday, Christmas.
- **4 seasons**: Spring, Summer, Fall, Winter.

---

## 3. Architecture and Project Flow

```
raw CSV data (data_kaggle/)
        │
        ▼
  generate_data.py  ──  Python 3 / Pandas / scikit-learn
        │
        ├──▶  data/dashboard_data.json   (charts, KPIs, ML results)
        └──▶  data/rag_context.json      (RAG context for AI chat)
                      │
                      ▼
              retil_store/ (static site)
                ├── index.html        ← dashboard structure
                ├── css/styles.css    ← dark theme
                ├── js/dashboard.js   ← Chart.js rendering + filters
                └── api/chat.js       ← Vercel serverless (Groq RAG)
                          │
                          ▼
                     Vercel (production)
```

**Flow summary**:

1. The Python script (`generate_data.py`) is run **once** before deployment.
2. It processes the raw CSVs, trains two ML models, computes all chart data, and writes two JSON files.
3. The static site loads `dashboard_data.json` at runtime via `fetch()`.
4. All charts are rendered client-side using Chart.js 4.x.
5. The filter panel (Year, Region, Type, Season) dynamically re-aggregates data from the `store_weekly` subset embedded in the JSON.
6. When the user activates the AI chat, their question is sent to `/api/chat.js`, which loads `rag_context.json` and calls the Groq API with the context as a system prompt.

---

## 4. Technology Stack

### 4.1 Python (Data Analysis and Machine Learning)

**Why Python?** Python is the industry standard for data science. Its ecosystem — Pandas, NumPy, scikit-learn — covers the full pipeline from raw data ingestion to model training without any infrastructure overhead.

| Library | Version | Role |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading, merging, grouping, time-series aggregation |
| `numpy` | ≥ 1.24 | Numerical operations, array sampling |
| `scikit-learn` | ≥ 1.3 | `RandomForestRegressor`, `GradientBoostingRegressor`, metrics, encoders |
| `json` | stdlib | Output serialization |

The Python step runs **offline** (not on the server), which eliminates any compute cost in production.

### 4.2 JavaScript (Dashboard Interactivity)

**Why vanilla JS + Chart.js?** For a data dashboard, a heavy SPA framework (React, Vue) adds unnecessary build complexity. Vanilla JS with Chart.js gives full control, fast initial load, and zero build step — critical for a Vercel static deployment.

| Library | Version | Role |
|---|---|---|
| `Chart.js` | 4.4.3 (CDN) | All 11 chart types: line, bar, doughnut, scatter, radar |
| Vanilla JS | ES2020 | Filter logic, DOM manipulation, API calls |

### 4.3 HTML / CSS (Structure and Visual Design)

**Why custom CSS instead of Tailwind/Bootstrap?** A custom stylesheet gives pixel-perfect control over the dark theme, ensuring the dashboard matches the design specification. It avoids loading large utility frameworks for a single-page app.

- Google Fonts (Inter, JetBrains Mono) for a professional typographic hierarchy.
- CSS Custom Properties (`--variables`) for a consistent palette throughout.
- CSS Grid for the two-column and four-column KPI layouts.
- Responsive breakpoints at 1100px and 640px.

### 4.4 Groq API (AI Chat)

**Why Groq instead of OpenAI?** Groq offers a **free tier** (30 req/min) with `llama-3.3-70b-versatile`, a model comparable in quality to GPT-4o. This makes the AI assistant completely free for the user, with no credit card required.

### 4.5 Vercel (Deployment)

**Why Vercel?** Vercel provides:
- Zero-configuration static site hosting (index.html, CSS, JS, JSON served from CDN).
- Serverless Functions in Node.js for the `/api/chat` endpoint — no server to manage.
- Free Hobby tier sufficient for this dashboard.
- Automatic HTTPS, global CDN, Git-based deploys.

---

## 5. Project Structure

```
retil_store/
├── index.html              ← Dashboard entry point
├── vercel.json             ← Vercel routing and function config
├── package.json            ← Node.js metadata
├── generate_data.py        ← Python data pipeline (run before deploy)
├── PROJECT_MANUAL.md       ← This document
│
├── css/
│   └── styles.css          ← Complete dark-theme stylesheet
│
├── js/
│   └── dashboard.js        ← Chart.js charts + filter logic + chat UI
│
├── api/
│   └── chat.js             ← Vercel serverless function (Groq RAG)
│
├── data/
│   ├── dashboard_data.json ← Generated by generate_data.py
│   └── rag_context.json    ← Generated by generate_data.py
│
└── img/
    └── logo.png            ← Retil Store branding
```

---

## 6. Data Pipeline — generate_data.py

The script performs these steps in sequence:

### Step 1 — Load and Merge
All three CSVs are loaded with Pandas and merged on `store_id` and `date`, producing a master DataFrame of 156,000 rows with all features available for each sale observation.

### Step 2 — Feature Engineering
New columns derived before ML training:
- `year`, `month`, `week_num` — temporal features.
- `type_enc`, `region_enc`, `season_enc` — label-encoded categorical features.
- `week_of_year`, `month_num` — cyclical time features.
- `markdown_1–5` — NaN values filled with 0 (no markdown = 0 discount).

### Step 3 — Aggregation for Charts
Pre-computed aggregations are stored in the output JSON:
- **KPIs**: `sum`, `mean`, and lift calculations.
- **Monthly revenue**: grouped by `YYYY-MM` period.
- **Store weekly** (filterable): grouped by `date × store_type × region × season`.
- **By type / by region**: `groupby` with revenue, average, and store counts.
- **Weekly trend**: first 60 weeks, pivoted by store type.
- **Holiday impact**: mean sales per holiday vs. baseline.
- **Seasonal pattern**: mean sales per season.

### Step 4 — ML Model Training

**Feature set** (17 features):
`store_size`, `type_enc`, `region_enc`, `season_enc`, `is_holiday`, `temperature`, `fuel_price`, `cpi`, `unemployment`, `week_of_year`, `month_num`, `markdown_1–5`

**Train/test split**: 80% / 20%, `random_state=42`.

Two models are trained:
- `RandomForestRegressor(n_estimators=100)`
- `GradientBoostingRegressor(n_estimators=100)`

Metrics computed: **MAE**, **R²**, **RMSE**.
Feature importances (RF) and Pearson correlations are extracted and stored.

A random sample of 200 test-set predictions is stored for the scatter plot.

### Step 5 — RAG Context
A structured JSON summary of all key statistics is written to `rag_context.json`. This file is loaded by the serverless function at runtime and injected as the LLM's system prompt.

---

## 7. Dashboard Sections and Charts

### 7.1 Key Performance Indicators (KPIs)

Four cards, each with a colored accent bar at the top:

| KPI | Value | Explanation |
|---|---|---|
| Total Sales | $8.81B | Sum of all `weekly_sales` across 156,000 records |
| Avg Weekly Sales | $56.5K | Mean `weekly_sales` per store-department pair |
| Holiday Sales Lift | +59.9% | `(avg_holiday_sales − avg_non_holiday) / avg_non_holiday × 100` |
| Best Store Revenue | $303.2M | Maximum cumulative revenue for a single store |

KPIs update dynamically when filters are applied.

### 7.2 Revenue Time Series

**Chart type**: Line chart with gradient fill.  
**Data**: Monthly aggregated revenue across all store-department pairs.  
**Insight**: Reveals seasonal patterns, year-over-year growth, and the effect of major holidays.  
The X-axis shows YYYY-MM labels; the Y-axis shows revenue in abbreviated format ($M).

### 7.3 Store Segmentation

**Left — Revenue by Store Type (Doughnut chart + table)**  
A doughnut chart shows the revenue share of each store type:
- Type A (gold): 26 stores, 73.7% of revenue.
- Type B (cyan): 16 stores, 21.7% of revenue.
- Type C (purple): 8 stores, 4.6% of revenue.

The size disparity reflects both the greater number of Type A stores and their much larger footprint (average store size ~213,000 sq ft vs. ~31,600 sq ft for Type C).

**Right — Revenue by Region (Horizontal bar chart)**  
Regions ranked by total revenue:
1. East — $2.9B
2. North — $2.5B
3. South — $2.0B
4. West — $1.5B

### 7.4 Weekly Sales Trend by Store Type

**Chart type**: Multi-line (3 series).  
**Data**: First 60 business weeks, one line per store type.  
**Insight**: Demonstrates the dramatic scale difference between types. Type A lines show strong weekly variability driven by holidays; Type C lines are nearly flat by comparison.

### 7.5 Holiday Impact on Sales

**Chart type**: Vertical bar chart.  
**Data**: Average weekly sales during each named holiday vs. the non-holiday baseline.  

This chart visualises the **holiday uplift** — how much each event boosts sales above the baseline. Black Friday and Christmas consistently show the largest uplifts, reflecting Walmart's role in US retail holiday spending.

### 7.6 Seasonal Pattern (Radar Chart)

**Chart type**: Radar (spider) chart with 4 axes (Spring, Summer, Fall, Winter).  
**Insight**: The area of the polygon represents total seasonal sales. The asymmetry of the polygon reveals which seasons have above-average sales. Winter (Q4) typically shows the largest spike due to the holiday shopping season.

### 7.7 Feature Importance — Random Forest

**Chart type**: Horizontal bar chart.  
**Data**: `feature_importances_` from the trained `RandomForestRegressor`.  

Feature importance (mean decrease in impurity, MDI) indicates how much each feature reduces variance across all trees in the forest. The top features are almost always `store_size` and store type-related encodings, confirming that structural store characteristics dominate the variance in weekly sales.

### 7.8 Pearson Correlation with Weekly Sales

**Chart type**: Diverging horizontal bar chart (teal = positive, red = negative).  
**Data**: `df.corr()['weekly_sales']` for all numeric features.  

**Pearson r** measures the linear association between each numeric feature and weekly sales. A value close to +1 means the feature increases with sales; close to −1 means it decreases. `store_size` consistently shows the highest positive correlation.

### 7.9 Markdown Channel Correlation

**Chart type**: Vertical bar chart.  
**Data**: Pearson correlation of `markdown_1` through `markdown_5` with `weekly_sales`.  

Markdowns are promotional discounts applied across 5 channels. Their correlation with weekly sales reveals which channel has the strongest discount-to-revenue relationship. Near-zero or negative correlations can indicate that discounts are often applied during slower periods (reverse causality).

### 7.10 Top 10 Departments by Revenue

**Chart type**: Vertical bar chart (purple gradient).  
**Data**: Sum of `weekly_sales` grouped by `department`, top 10.  

High-revenue departments (typically groceries, electronics, and home goods) dominate. This chart guides inventory allocation and promotional planning.

### 7.11 Machine Learning — Predictive Models

Two models are compared side by side.

**Random Forest**
- An ensemble of decision trees trained with bootstrap aggregation (bagging).
- Each tree is trained on a random data subset with random feature selection, reducing overfitting.
- The final prediction is the **average of all tree predictions**.

**Gradient Boosting**
- Trees are trained **sequentially**: each new tree corrects the residual errors of the ensemble so far.
- Typically outperforms Random Forest on tabular data with enough iterations.
- More sensitive to hyperparameter tuning (learning rate, depth, n_estimators).

**Metrics explained**:

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | mean(|y − ŷ|) | Average absolute error in dollars. Easy to interpret. |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more than MAE. |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained. 1 = perfect, 0 = no better than mean. |

**Note on R² values (~0.43)**: An R² of ~0.43 is expected for this problem. Weekly sales at the store-department level have high inherent variance driven by localized demand shocks (weather events, local promotions, regional economic changes) that are not captured in the available features. The models capture the structural effects (store type, size, seasonality) well.

### 7.12 Actual vs Predicted Scatter Plot

**Chart type**: Scatter plot + diagonal reference line.  
**Data**: 200 random test-set samples. Each point shows (actual, predicted RF) for a single store-department-week.

**Interpretation**: Points on or near the diagonal dashed line ("perfect fit") indicate accurate predictions. The spread perpendicular to the line reflects prediction error. Systematic deviations from the diagonal indicate model bias.

---

## 8. Statistical and ML Concepts

### 8.1 Holiday Lift
A multiplicative or additive measure of how much a holiday increases sales above the baseline:

```
Lift (%) = (avg_holiday_sales − avg_non_holiday_sales) / avg_non_holiday_sales × 100
```

A lift of +59.9% means holiday weeks generate 59.9% more revenue per observation than average non-holiday weeks.

### 8.2 Pearson Correlation Coefficient
```
r = Σ[(xi − x̄)(yi − ȳ)] / [n · σx · σy]
```
Ranges from −1 to +1. Only measures **linear** relationships. Features with high non-linear relationships (such as cyclical time variables) may show near-zero Pearson r yet still be important predictors in tree models.

### 8.3 Feature Importance (Random Forest — MDI)
Mean Decrease in Impurity measures how much a feature reduces the weighted variance of the target variable across all splits in all trees. Higher = more important.

**Caution**: MDI can overestimate the importance of high-cardinality numeric features. For more robust importance estimates, permutation importance should be used in production.

### 8.4 Train/Test Split
The dataset is split 80/20 with `random_state=42` to ensure reproducibility. All model metrics (MAE, RMSE, R²) are computed exclusively on the **held-out test set** to reflect out-of-sample performance.

### 8.5 Mean Absolute Error (MAE)
The average absolute difference between actual and predicted sales. Measured in dollars, it is directly interpretable: an MAE of $25,000 means the model is off by $25K on average per store-department-week.

### 8.6 Root Mean Squared Error (RMSE)
Similar to MAE but squares the errors before averaging, then takes the square root. This gives extra weight to large prediction errors, making RMSE a stricter quality measure when large errors are costly.

---

## 9. AI Chat — RAG with Groq

### 9.1 What is RAG?
**Retrieval-Augmented Generation (RAG)** is a technique where relevant external knowledge is retrieved and injected into the LLM's context before it generates a response. This grounding prevents the model from hallucinating facts about the specific dataset.

In this dashboard, the "retrieval" step is simplified: instead of a vector database, a structured JSON summary of the dataset (`rag_context.json`) is loaded at runtime and inserted directly into the system prompt. This is sometimes called **document-stuffing RAG** — appropriate when the context is small and deterministic.

### 9.2 Architecture

```
Browser  ──POST {message, apiKey}──▶  /api/chat.js (Vercel Serverless)
                                           │
                                    reads  │  rag_context.json
                                           │
                                    ──POST──▶  api.groq.com/v1/chat/completions
                                           │      Model: llama-3.3-70b-versatile
                                    ◀──reply──
                                           │
Browser  ◀──JSON {reply}────────────────────
```

### 9.3 Security
- The Groq API key is provided by the user and sent in the POST body to the serverless function over HTTPS.
- The key is **never logged or stored** server-side — it exists only in the request body and is forwarded once to the Groq API.
- Users should use Groq's free-tier keys, which have rate limits that prevent abuse.

### 9.4 Model — Llama 3.3 70B
`llama-3.3-70b-versatile` is Meta's open-weight large language model hosted on Groq's inference infrastructure. At 70 billion parameters, it achieves GPT-4o-comparable quality on analytical and question-answering tasks, making it ideal for data analysis conversations.

---

## 10. Deployment on Vercel

### Prerequisites
- A [Vercel](https://vercel.com) account (free Hobby plan is sufficient).
- The `vercel` CLI: `npm install -g vercel`.

### Steps

1. **Generate the data** (once, before deploying):
   ```bash
   cd retil_store
   python generate_data.py
   ```
   This creates `data/dashboard_data.json` and `data/rag_context.json`.

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   cd retil_store
   vercel --prod
   ```
   Vercel detects the `api/` folder and deploys `chat.js` as a serverless function automatically. All other files (HTML, CSS, JS, data) are served as static assets from Vercel's CDN.

4. **Visit the URL** provided by Vercel. The dashboard is live.

### Environment Variables
No environment variables are required on Vercel. The Groq API key is supplied at runtime by the user in the browser.

### Re-deploying with New Data
If the source CSVs are updated:
```bash
python generate_data.py  # regenerate JSON
vercel --prod            # redeploy
```

---

## 11. How to Run Locally

### Option A — Vercel Dev Server (recommended)

```bash
npm install -g vercel
cd retil_store
python generate_data.py   # only needed once
vercel dev
```

Open `http://localhost:3000`.

### Option B — Simple HTTP Server

The serverless function won't work without Vercel, but you can preview the static dashboard:

```bash
cd retil_store
python -m http.server 8080
```

Open `http://localhost:8080`. The AI chat will not work in this mode (no serverless runtime), but all charts and filters will be fully functional.

### Python Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
```

Install with:
```bash
pip install pandas numpy scikit-learn
```

---

*Retil Store Dashboard — Built with Python, JavaScript, HTML/CSS, and deployed on Vercel. Powered by Groq Llama 3.3 70B.*
