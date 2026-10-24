/**
 * Request helpers for the chat endpoint: CORS policy, client IP and input checks.
 */

'use strict';

const MAX_MESSAGE_CHARS = 1000;

function allowedOrigins(env = process.env) {
  return (env.ALLOWED_ORIGINS || '')
    .split(',')
    .map(o => o.trim().replace(/\/$/, ''))
    .filter(Boolean);
}

/**
 * Same-origin requests are always fine. Cross-origin requests are only
 * accepted from ALLOWED_ORIGINS. Requests without an Origin header (curl,
 * server-to-server) are left to the rate limiter.
 */
function checkOrigin(req, env = process.env) {
  const origin = req.headers?.origin;
  if (!origin) return { allowed: true, corsOrigin: null };
  let host;
  try {
    host = new URL(origin).host;
  } catch {
    return { allowed: false, corsOrigin: null };
  }
  if (host === req.headers.host) return { allowed: true, corsOrigin: null };
  if (allowedOrigins(env).includes(origin)) return { allowed: true, corsOrigin: origin };
  return { allowed: false, corsOrigin: null };
}

function clientIp(req) {
  const fwd = req.headers?.['x-forwarded-for'];
  if (fwd) return String(fwd).split(',')[0].trim();
  return req.headers?.['x-real-ip'] || req.socket?.remoteAddress || 'unknown';
}

function validateMessage(message) {
  if (typeof message !== 'string') return { ok: false, error: 'Missing or invalid message' };
  const text = message.trim();
  if (!text) return { ok: false, error: 'Message is empty' };
  if (text.length > MAX_MESSAGE_CHARS) {
    return { ok: false, error: `Message is too long (max ${MAX_MESSAGE_CHARS} characters)` };
  }
  return { ok: true, text };
}

module.exports = { MAX_MESSAGE_CHARS, allowedOrigins, checkOrigin, clientIp, validateMessage };
