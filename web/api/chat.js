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
 */

'use strict';

const { buildSystemPrompt } = require('./_lib/prompt');

let context = null;
try {
  // A static require lets Vercel bundle the JSON with the function.
  context = require('../data/assistant_context.json');
} catch {
  context = null;
}

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const MODEL = 'llama-3.3-70b-versatile';

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin',  '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  const serverKey = process.env.GROQ_API_KEY || '';

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method === 'GET') return res.status(200).json({ serverKey: Boolean(serverKey) });
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { message, apiKey: userKey } = req.body || {};

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid message' });
  }
  const apiKey = serverKey || userKey;
  if (!apiKey || typeof apiKey !== 'string' || !apiKey.startsWith('gsk_')) {
    return res.status(400).json({ error: 'Missing or invalid Groq API key' });
  }

  try {
    const groqRes = await fetch(GROQ_URL, {
      method: 'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: 'system', content: buildSystemPrompt(context) },
          { role: 'user',   content: message },
        ],
        temperature: 0.2,
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
