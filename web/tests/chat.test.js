'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { createHandler } = require('../api/chat');
const { buildSystemPrompt, FALLBACK_PROMPT } = require('../api/_lib/prompt');
const { checkOrigin, clientIp, validateMessage } = require('../api/_lib/http');
const { createMemoryLimiter, createUpstashLimiter } = require('../api/_lib/ratelimit');
const context = require('../data/assistant_context.json');

/* Minimal stand-ins for the Vercel req/res objects. */
function mockReq({ method = 'POST', body = {}, headers = {} } = {}) {
  return { method, body, headers: { host: 'retail.example', ...headers } };
}
function mockRes() {
  return {
    statusCode: 200, headers: {}, body: undefined,
    setHeader(k, v) { this.headers[k.toLowerCase()] = v; },
    status(code) { this.statusCode = code; return this; },
    json(obj) { this.body = obj; return this; },
    end() { return this; },
  };
}
function groqOk(reply = 'ok') {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init: { ...init, body: JSON.parse(init.body) } });
    return { ok: true, json: async () => ({ choices: [{ message: { content: reply } }] }) };
  };
  return { fetchImpl, calls };
}
const allowAll = async () => ({ allowed: true });

test('GET reports whether a server key is configured', async () => {
  for (const [env, expected] of [[{ GROQ_API_KEY: 'gsk_x' }, true], [{}, false]]) {
    const res = mockRes();
    await createHandler({ env, limiter: allowAll })(mockReq({ method: 'GET' }), res);
    assert.equal(res.statusCode, 200);
    assert.deepEqual(res.body, { serverKey: expected });
  }
});

test('server key is used and the visitor key is ignored', async () => {
  const { fetchImpl, calls } = groqOk('hello');
  const res = mockRes();
  await createHandler({ env: { GROQ_API_KEY: 'gsk_server' }, fetchImpl, limiter: allowAll })(
    mockReq({ body: { message: 'Hi', apiKey: 'gsk_visitor' } }), res);
  assert.equal(res.statusCode, 200);
  assert.equal(res.body.reply, 'hello');
  assert.equal(calls[0].init.headers.Authorization, 'Bearer gsk_server');
  assert.equal(calls[0].init.body.messages[0].role, 'system');
  assert.equal(calls[0].init.body.messages[1].content, 'Hi');
});

test('falls back to the visitor key when no server key is set', async () => {
  const { fetchImpl, calls } = groqOk();
  const res = mockRes();
  await createHandler({ env: {}, fetchImpl, limiter: allowAll })(
    mockReq({ body: { message: 'Hi', apiKey: 'gsk_visitor' } }), res);
  assert.equal(res.statusCode, 200);
  assert.equal(calls[0].init.headers.Authorization, 'Bearer gsk_visitor');
});

test('rejects missing keys, empty and oversized messages', async () => {
  const handler = createHandler({ env: {}, fetchImpl: groqOk().fetchImpl, limiter: allowAll });
  const cases = [
    { body: { message: 'Hi' }, error: /API key/ },
    { body: { message: '   ', apiKey: 'gsk_a' }, error: /empty/ },
    { body: { message: 'x'.repeat(1001), apiKey: 'gsk_a' }, error: /too long/ },
    { body: { message: 42, apiKey: 'gsk_a' }, error: /invalid message/ },
  ];
  for (const c of cases) {
    const res = mockRes();
    await handler(mockReq({ body: c.body }), res);
    assert.equal(res.statusCode, 400);
    assert.match(res.body.error, c.error);
  }
});

test('blocks unknown cross-origin callers and allows listed ones', async () => {
  const env = { GROQ_API_KEY: 'gsk_s', ALLOWED_ORIGINS: 'https://ok.example' };
  const handler = createHandler({ env, fetchImpl: groqOk().fetchImpl, limiter: allowAll });

  const blocked = mockRes();
  await handler(mockReq({ body: { message: 'Hi' }, headers: { origin: 'https://evil.example' } }), blocked);
  assert.equal(blocked.statusCode, 403);
  assert.equal(blocked.headers['access-control-allow-origin'], undefined);

  const allowed = mockRes();
  await handler(mockReq({ body: { message: 'Hi' }, headers: { origin: 'https://ok.example' } }), allowed);
  assert.equal(allowed.statusCode, 200);
  assert.equal(allowed.headers['access-control-allow-origin'], 'https://ok.example');

  const same = mockRes();
  await handler(mockReq({ body: { message: 'Hi' }, headers: { origin: 'https://retail.example' } }), same);
  assert.equal(same.statusCode, 200);
  assert.equal(same.headers['access-control-allow-origin'], undefined);
});

test('returns 429 with Retry-After once the limit is hit', async () => {
  const limiter = createMemoryLimiter({ limit: 2 });
  const handler = createHandler({ env: { GROQ_API_KEY: 'gsk_s' }, fetchImpl: groqOk().fetchImpl, limiter });
  const statuses = [];
  let last;
  for (let i = 0; i < 3; i += 1) {
    last = mockRes();
    await handler(mockReq({ body: { message: 'Hi' }, headers: { 'x-forwarded-for': '1.2.3.4' } }), last);
    statuses.push(last.statusCode);
  }
  assert.deepEqual(statuses, [200, 200, 429]);
  assert.ok(Number(last.headers['retry-after']) > 0);
});

test('a limiter outage does not take the chat down', async () => {
  const limiter = async () => { throw new Error('redis down'); };
  const res = mockRes();
  await createHandler({ env: { GROQ_API_KEY: 'gsk_s' }, fetchImpl: groqOk().fetchImpl, limiter })(
    mockReq({ body: { message: 'Hi' } }), res);
  assert.equal(res.statusCode, 200);
});

test('upstream errors are mapped and network errors are not echoed', async () => {
  const failing = async () => ({ ok: false, status: 500, json: async () => ({}) });
  const res = mockRes();
  await createHandler({ env: { GROQ_API_KEY: 'gsk_s' }, fetchImpl: failing, limiter: allowAll })(
    mockReq({ body: { message: 'Hi' } }), res);
  assert.equal(res.statusCode, 502);

  const throwing = async () => { throw new Error('ECONNRESET internal detail'); };
  const res2 = mockRes();
  await createHandler({ env: { GROQ_API_KEY: 'gsk_s' }, fetchImpl: throwing, limiter: allowAll })(
    mockReq({ body: { message: 'Hi' } }), res2);
  assert.equal(res2.statusCode, 502);
  assert.doesNotMatch(res2.body.error, /ECONNRESET/);
});

test('memory limiter resets after the window', async () => {
  let t = 0;
  const check = createMemoryLimiter({ limit: 1, windowSeconds: 60, now: () => t });
  assert.equal((await check('ip')).allowed, true);
  assert.equal((await check('ip')).allowed, false);
  assert.equal((await check('other')).allowed, true);
  t = 61_000;
  assert.equal((await check('ip')).allowed, true);
});

test('upstash limiter sends INCR/EXPIRE/TTL and reads the count', async () => {
  let sent;
  const fetchImpl = async (url, init) => {
    sent = { url, body: JSON.parse(init.body) };
    return { ok: true, json: async () => [{ result: 3 }, { result: 0 }, { result: 42 }] };
  };
  const check = createUpstashLimiter({ url: 'https://u.example', token: 't', limit: 2, fetchImpl });
  const r = await check('1.2.3.4');
  assert.equal(sent.url, 'https://u.example/pipeline');
  assert.equal(sent.body[0][0], 'INCR');
  assert.deepEqual(r, { allowed: false, remaining: 0, retryAfter: 42 });
});

test('helpers', () => {
  assert.equal(clientIp({ headers: { 'x-forwarded-for': '9.9.9.9, 10.0.0.1' } }), '9.9.9.9');
  assert.equal(checkOrigin({ headers: { origin: 'not a url' } }, {}).allowed, false);
  assert.equal(checkOrigin({ headers: {} }, {}).allowed, true);
  assert.equal(validateMessage('  hi  ').text, 'hi');
});

test('system prompt is grounded on the exported context', () => {
  const prompt = buildSystemPrompt(context);
  assert.match(prompt, /grounded|Answer only from the facts/);
  assert.ok(prompt.includes(String(context.model_evaluation.wmae.toLocaleString('en-US'))));
  for (const t of Object.keys(context.store_types)) assert.ok(prompt.includes(`Type ${t}`));
  assert.ok(prompt.includes(context.forecast_next_weeks[0].week));
  assert.doesNotMatch(prompt, /undefined|NaN/);
  assert.equal(buildSystemPrompt(null), FALLBACK_PROMPT);
});
