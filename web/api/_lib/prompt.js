/**
 * System prompt for the dashboard assistant.
 *
 * This is grounded prompting (a.k.a. context injection), not RAG: there is no
 * retrieval step. The pipeline precomputes a small set of facts
 * (data/assistant_context.json) and the whole file goes into every request.
 * That is enough while the context fits comfortably in the prompt; retrieval
 * would only pay off if the assistant had to search a much larger corpus.
 */

'use strict';

const FALLBACK_PROMPT =
  'You are a data analyst assistant for the Retail Store sales forecasting dashboard. ' +
  'The dataset context failed to load, so say you cannot answer questions about the data ' +
  'right now instead of guessing.';

function buildSystemPrompt(ctx) {
  if (!ctx) return FALLBACK_PROMPT;

  const types = Object.entries(ctx.store_types)
    .map(([t, v]) => `- Type ${t} (${v.description}): ${v.count} stores, ${v.share_revenue} of revenue`)
    .join('\n');
  const regions = Object.entries(ctx.regions).map(([r, v]) => `- ${r}: ${v}`).join('\n');
  const holidays = Object.entries(ctx.holiday_lift_by_event)
    .map(([h, l]) => `- ${h}: ${l}`)
    .join('\n');
  const ev = ctx.model_evaluation;
  const forecast = ctx.forecast_next_weeks.map(f => `- ${f.week}: ${f.total_forecast}`).join('\n');

  return `You are a data analyst assistant for the Retail Store weekly sales forecasting dashboard.
Answer only from the facts below. If a question needs data that is not here, say so plainly.

DATASET
${ctx.dataset_summary}

KPIs
- Total sales: ${ctx.kpis.total_sales}
- Average weekly sales: ${ctx.kpis.avg_weekly_sales}
- Holiday lift: ${ctx.kpis.holiday_sales_lift}
- Best single store: ${ctx.kpis.max_store_revenue}

STORE TYPES
${types}

REVENUE BY REGION
${regions}

HOLIDAY LIFT BY EVENT (vs. the same store-department's non-holiday weeks)
${holidays}

FORECASTING MODEL
- ${ev.protocol}
- Best model: ${ev.best_model}
- WMAE (holiday weeks weighted 5x): $${ev.wmae.toLocaleString('en-US')}
- WAPE: ${ev.wape_pct}% · bias: ${ev.bias_pct}% (positive = over-forecast)
- Seasonal naive baseline WMAE: $${ev.seasonal_naive_wmae.toLocaleString('en-US')}
- Skill vs. seasonal naive: ${ev.skill_vs_seasonal_naive_pct}%
- 80% prediction interval coverage on the holdout: ${ev.interval_coverage_80_pct}%
- Most important features (permutation importance): ${ctx.top_features.join(', ')}

FORECAST, TOTAL WEEKLY SALES
${forecast}

TOP DEPARTMENTS BY REVENUE
${ctx.top_departments.join(', ')}

Guidelines:
- Be concise. Quote the numbers above when they answer the question.
- Do not invent figures, dates or causes that are not listed.
- The data is an adapted Kaggle dataset, not real Walmart data; mention it if asked about the source.`;
}

module.exports = { buildSystemPrompt, FALLBACK_PROMPT };
