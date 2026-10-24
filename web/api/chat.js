/**
 * Retail Store — assistant endpoint (Vercel Node.js serverless function).
 * GET  /api/chat                       ->  { serverKey: boolean }
 * POST /api/chat  { message, apiKey? }  ->  { reply }
 *
 * The Groq key normally lives in the GROQ_API_KEY environment variable, so
 * visitors can use the chat without one. If it is not set, the endpoint falls
 * back to a key supplied by the visitor (bring-your-own-key).
 *
 * The assistant is grounded on precomputed facts (see _lib/prompt.js); there is
 * no retrieval step.
 *
 * Environment:
 *   GROQ_API_KEY              shared key (optional)
 *   ALLOWED_ORIGINS           extra origins allowed to call the API, comma separated
 *   RATE_LIMIT_PER_MINUTE     requests per IP per minute (default 10)
 *   UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN   shared rate-limit store (optional)
 */

'use strict';

const { buildSystemPrompt } = require('./_lib/prompt');
const { checkOrigin, clientIp, validateMessage } = require('./_lib/http');
const { createLimiter } = require('./_lib/ratelimit');

let defaultContext = null;
try {
  // A static require lets Vercel bundle the JSON with the function.
  defaultContext = require('../data/assistant_context.json');
} catch {
  defaultContext = null;
}

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const MODEL = 'llama-3.3-70b-versatile';

function createHandler({ env = process.env, fetchImpl = fetch, limiter = createLimiter(env),
                         context = defaultContext } = {}) {
  return async function handler(req, res) {
    const origin = checkOrigin(req, env);
    if (origin.corsOrigin) {
      res.setHeader('Access-Control-Allow-Origin', origin.corsOrigin);
      res.setHeader('Vary', 'Origin');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    }
    if (!origin.allowed) return res.status(403).json({ error: 'Origin not allowed' });

    const serverKey = env.GROQ_API_KEY || '';

    if (req.method === 'OPTIONS') return res.status(204).end();
    if (req.method === 'GET') return res.status(200).json({ serverKey: Boolean(serverKey) });
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

    const body = req.body || {};
    const msg = validateMessage(body.message);
    if (!msg.ok) return res.status(400).json({ error: msg.error });

    const apiKey = serverKey || body.apiKey;
    if (!apiKey || typeof apiKey !== 'string' || !apiKey.startsWith('gsk_')) {
      return res.status(400).json({ error: 'Missing or invalid Groq API key' });
    }

    let limit;
    try {
      limit = await limiter(clientIp(req));
    } catch {
      limit = { allowed: true };  // a limiter outage should not take the chat down
    }
    if (!limit.allowed) {
      res.setHeader('Retry-After', String(limit.retryAfter || 60));
      return res.status(429).json({ error: 'Too many requests, please wait a minute and try again' });
    }

    try {
      const groqRes = await fetchImpl(GROQ_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: MODEL,
          messages: [
            { role: 'system', content: buildSystemPrompt(context) },
            { role: 'user',   content: msg.text },
          ],
          temperature: 0.2,
          max_tokens:  600,
        }),
      });

      if (!groqRes.ok) {
        const errBody = await groqRes.json().catch(() => ({}));
        const status = groqRes.status === 401 ? 401 : groqRes.status === 429 ? 429 : 502;
        return res.status(status).json({
          error: errBody?.error?.message || `Groq API error ${groqRes.status}`,
        });
      }

      const data = await groqRes.json();
      const reply = data?.choices?.[0]?.message?.content || 'No response from model.';
      return res.status(200).json({ reply });
    } catch {
      return res.status(502).json({ error: 'Could not reach the language model' });
    }
  };
}

module.exports = createHandler();
module.exports.createHandler = createHandler;
