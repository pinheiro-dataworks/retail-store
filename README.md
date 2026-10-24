<p align="center">
  <img src="web/img/logo.png" alt="Retail Store" width="320"/>
</p>

# Retail Store — Sales Forecasting Dashboard

An interactive sales forecasting dashboard built on real retail data, combining exploratory analysis, machine learning models, and a conversational AI assistant powered by Groq.

**Live demo:** [retail-store.vercel.app](https://retail-store.vercel.app)

---

## Objective

Turn raw weekly sales data from 50 retail stores into a fully interactive analytics dashboard that enables business users to:

- Monitor key revenue KPIs across stores, regions, and store types
- Explore seasonal patterns and the impact of holiday events on sales
- Understand which features drive weekly sales through correlation and feature importance analysis
- Compare the predictive performance of Random Forest and Gradient Boosting models
- Ask natural-language questions about the data via an AI assistant

---

## Dataset

**Source:** [Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) (Kaggle)  
**Adapted by:** Noopur Bhatt

| File | Description |
|---|---|
| `sales.csv` | 156,000 weekly sales records across 50 stores and 20 departments (2022–2024) |
| `stores.csv` | Store metadata: type (A/B/C), size, and region (East/North/South/West) |
| `features.csv` | External variables: temperature, fuel price, CPI, unemployment, markdowns, holidays, and seasons |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data analysis & ML | Python · Pandas · NumPy · scikit-learn | Industry standard for data pipelines and model training |
| Charts & interactivity | JavaScript · Chart.js 4.x | Lightweight, flexible charting with no build step |
| Structure & styling | HTML5 · CSS3 (custom dark theme) | Full design control, no framework overhead |
| AI chat | Groq API · Llama 3.3 70B | Free tier, GPT-4o quality, no credit card required |
| Serverless function | Node.js (Vercel Functions) | Proxies Groq API calls with RAG context injection |
| Deployment | Vercel | Zero-config static hosting + serverless functions on free tier |

---

## Features

- **4 KPI cards** — Total Sales ($8.81B), Avg Weekly Sales, Holiday Lift (+59.9%), Best Store Revenue
- **11 interactive charts** — line, bar, doughnut, radar, and scatter plot
- **Dynamic filters** — Year, Region, Store Type, and Season update charts in real time
- **ML models** — Random Forest and Gradient Boosting with MAE, RMSE, and R² metrics
- **Actual vs. Predicted scatter plot** — visual model evaluation on 200 test samples
- **AI Chat with RAG** — Groq Llama 3.3 70B grounded in dataset context via a serverless function

---

## Project Structure

```
web/
├── index.html              ← Dashboard entry point
├── vercel.json             ← Vercel routing config
├── package.json
├── generate_data.py        ← Python pipeline (run once before deploy)
├── PROJECT_MANUAL.md       ← Full technical documentation
├── css/styles.css
├── js/dashboard.js
├── api/chat.js             ← Groq RAG serverless function
├── data/
│   ├── dashboard_data.json ← Pre-computed chart data
│   └── rag_context.json    ← AI chat context
└── img/logo.png
```

---

## Getting Started

### 1. Generate the data (Python, run once)

```bash
pip install pandas numpy scikit-learn
cd web
python generate_data.py
```

### 2. Preview locally

```bash
python -m http.server 8080
# open http://localhost:8080
```

### 3. Deploy to Vercel

```bash
npm install -g vercel
vercel --prod
```

> The AI chat requires a free Groq API key from [console.groq.com](https://console.groq.com). No credit card needed.

---

## Author

**Renan Pinheiro**  
Data Scientist

[![GitHub](https://img.shields.io/badge/GitHub-pinheiro--dataworks-181717?logo=github)](https://github.com/pinheiro-dataworks)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-pinheirodata-0A66C2?logo=linkedin)](https://www.linkedin.com/in/pinheirodata/)

---

*Retail Store Dashboard — Version 1.0.0 · © 2026*
