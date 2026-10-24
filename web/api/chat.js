/**
 * Retil Store — Groq AI Chat Serverless Function
 * Vercel Edge Function: /api/chat
 *
 * RAG approach: loads rag_context.json at runtime and injects it as a
 * structured system prompt, giving the LLM accurate, dataset-specific
 * knowledge without any vector database.
 */

const fs   = require('fs');
const path = require('path');

const RAG_PATH = path.join(process.cwd(), 'data', 'rag_context.json');

let ragContext = null;

function loadRagContext() {
  if (ragContext) return ragContext;
  try {
    const raw = fs.readFileSync(RAG_PATH, 'utf-8');
    ragContext = JSON.parse(raw);
  } catch {
    ragContext = null;
  }
  return ragContext;
}

function buildSystemPrompt(ctx) {
  if (!ctx) {
    return `You are a helpful data analyst assistant for the Retil Store Sales Forecasting Dashboard.
Answer questions about retail store sales data clearly and concisely.`;
  }

  return `You are an expert data analyst assistant for the **Retil Store Sales Forecasting Dashboard**.
You have access to the following accurate dataset context. Use it to give precise, data-driven answers.

---
**DATASET OVERVIEW**
${ctx.dataset_summary}

**KEY PERFORMANCE INDICATORS**
- Total Sales: ${ctx.kpis.total_sales}
- Average Weekly Sales: ${ctx.kpis.avg_weekly_sales}
- Holiday Sales Lift: ${ctx.kpis.holiday_sales_lift}
- Best Single Store Revenue: ${ctx.kpis.max_store_revenue}

**STORE TYPES**
- Type A (Large): ${ctx.store_types.A.count} stores · ${ctx.store_types.A.share_revenue} of total revenue
- Type B (Medium): ${ctx.store_types.B.count} stores · ${ctx.store_types.B.share_revenue} of total revenue
- Type C (Small): ${ctx.store_types.C.count} stores · ${ctx.store_types.C.share_revenue} of total revenue

**REVENUE BY REGION**
- East:  ${ctx.regions.East}
- North: ${ctx.regions.North}
- South: ${ctx.regions.South}
- West:  ${ctx.regions.West}

**HOLIDAY EVENTS & SALES LIFT**
${Object.entries(ctx.holiday_lift_by_event).map(([h, l]) => `- ${h}: ${l} vs baseline`).join('\n')}

**TOP PREDICTIVE FEATURES** (Random Forest importance)
${ctx.top_features.join(', ')}

**ML MODEL PERFORMANCE**
- Random Forest:     R²=${ctx.ml_performance.random_forest.R2.toFixed(3)}, MAE=$${Math.round(ctx.ml_performance.random_forest.MAE).toLocaleString()}, RMSE=$${Math.round(ctx.ml_performance.random_forest.RMSE).toLocaleString()}
- Gradient Boosting: R²=${ctx.ml_performance.gradient_boosting.R2.toFixed(3)}, MAE=$${Math.round(ctx.ml_performance.gradient_boosting.MAE).toLocaleString()}, RMSE=$${Math.round(ctx.ml_performance.gradient_boosting.RMSE).toLocaleString()}

**TOP DEPARTMENTS BY REVENUE**
${ctx.top_departments.join(', ')}

**SEASONS IN DATASET**: ${ctx.seasons.join(', ')}
---

Guidelines:
- Answer only what is asked. Be concise but informative.
- Use the data above when relevant — cite specific numbers when helpful.
- If asked something outside the dataset scope, say so honestly.
- Format numbers clearly (e.g., $8.81B, +59.9%).`;
}

module.exports = async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin',  '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message, apiKey } = req.body || {};

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid message' });
  }
  if (!apiKey || !apiKey.startsWith('gsk_')) {
    return res.status(400).json({ error: 'Missing or invalid Groq API key' });
  }

  const ctx          = loadRagContext();
  const systemPrompt = buildSystemPrompt(ctx);

  try {
    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        messages: [
          { role: 'system',  content: systemPrompt },
          { role: 'user',    content: message       },
        ],
        temperature: 0.4,
        max_tokens:  600,
      }),
    });

    if (!groqRes.ok) {
      const errBody = await groqRes.json().catch(() => ({}));
      const errMsg  = errBody?.error?.message || `Groq API error ${groqRes.status}`;
      return res.status(groqRes.status).json({ error: errMsg });
    }

    const data  = await groqRes.json();
    const reply = data?.choices?.[0]?.message?.content || 'No response from model.';
    return res.status(200).json({ reply });

  } catch (err) {
    return res.status(500).json({ error: `Server error: ${err.message}` });
  }
};
