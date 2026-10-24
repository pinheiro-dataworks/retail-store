/**
 * Fixed-window rate limiting per client IP.
 *
 * With UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN set, counters live in
 * Upstash Redis and are shared by every function instance. Without them the
 * limiter falls back to an in-memory map, which only protects a single warm
 * instance but needs no setup. Both are called through the same interface.
 */

'use strict';

const WINDOW_SECONDS = 60;

function createMemoryLimiter({ limit, windowSeconds = WINDOW_SECONDS, now = Date.now } = {}) {
  const hits = new Map();
  return async function check(key) {
    const t = now();
    const entry = hits.get(key);
    if (!entry || t >= entry.resetAt) {
      hits.set(key, { count: 1, resetAt: t + windowSeconds * 1000 });
      return { allowed: true, remaining: limit - 1 };
    }
    entry.count += 1;
    if (hits.size > 10_000) {
      for (const [k, v] of hits) if (t >= v.resetAt) hits.delete(k);
    }
    return {
      allowed: entry.count <= limit,
      remaining: Math.max(0, limit - entry.count),
      retryAfter: Math.ceil((entry.resetAt - t) / 1000),
    };
  };
}

function createUpstashLimiter({ url, token, limit, windowSeconds = WINDOW_SECONDS,
                                fetchImpl = fetch }) {
  return async function check(key) {
    const redisKey = `ratelimit:chat:${key}`;
    const res = await fetchImpl(`${url}/pipeline`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify([
        ['INCR', redisKey],
        ['EXPIRE', redisKey, String(windowSeconds), 'NX'],
        ['TTL', redisKey],
      ]),
    });
    if (!res.ok) throw new Error(`Upstash error ${res.status}`);
    const [incr, , ttl] = await res.json();
    const count = Number(incr.result);
    return {
      allowed: count <= limit,
      remaining: Math.max(0, limit - count),
      retryAfter: Math.max(1, Number(ttl.result)),
    };
  };
}

function createLimiter(env = process.env) {
  const limit = Number(env.RATE_LIMIT_PER_MINUTE) || 10;
  if (env.UPSTASH_REDIS_REST_URL && env.UPSTASH_REDIS_REST_TOKEN) {
    return createUpstashLimiter({
      url: env.UPSTASH_REDIS_REST_URL, token: env.UPSTASH_REDIS_REST_TOKEN, limit,
    });
  }
  return createMemoryLimiter({ limit });
}

module.exports = { createLimiter, createMemoryLimiter, createUpstashLimiter };
