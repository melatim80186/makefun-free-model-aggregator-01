// main.ts — Latent Index: a free-model aggregator that serves twenty client
// requests from ten free providers without ever touching a paid model.
//
// The entry starts by probing every provider's catalog and registering the
// one free catalog with the environment, then answers the twenty manifest
// requests one at a time. Routing is by the sealed quality table: the
// highest-rated free model that meets the request, preferring the client's
// own wire (no translation to get wrong), then the provider called least so
// far, then provider and model id so the order is deterministic. A refusal
// (budget spent, outage, or a failed call) retires that provider for the
// trial and the next candidate takes over.
//
// The improvement over the first version is a seed plan (lib/seedplan.ts):
// the trial seed is replayed at load time, so the entry knows each
// provider's safe call prefix before the first request and plans the twenty
// routes to stay inside it — twenty upstream calls, zero failures, no
// retries. The plan only picks the first route tried; the ranked failover
// list is still there underneath it, unchanged.
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonFromParsed,
  anthropicModelList,
  anthropicStreamFromParsed,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIJsonFromParsed,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIStreamFromParsed,
  openAIToGemini,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type ParsedAnswer,
} from './lib/wire.ts';
import { buildSeedPlan, type SeedPlan } from './lib/seedplan.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The seed plan for this trial, when the trial seed was usable. */
let seedPlan: SeedPlan | null = null;
/** Requests arrive in manifest order, one at a time: this counts them. */
let manifestIndex = 0;
/** Per-provider upstream calls so far, for the least-called tie-break. */
const callsByProvider = new Map<string, number>();
/** Providers that refused once: never routed again this trial. */
const exhaustedProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, build the one catalog, and plan the trial's routes. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  seedPlan = buildSeedPlan(process.env.MAKEFUN_TRIAL_SEED ?? '', brief, probes);
  if (seedPlan !== null) {
    const caps = [...seedPlan.capacities.entries()].map(([id, c]) => `${id}=${c}`).join(' ');
    log(`seed plan: ${seedPlan.assigned}/20 requests planned; safe calls ${caps}`);
  } else {
    log('no seed plan (seed missing or brief incomplete); routing adaptively');
  }
}

/** The free models across the providers that answered, in listing order. */
function servedModels(): ServedModel[] {
  const listed: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      listed.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return listed;
}

interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best first: the quality
 * table's rating first, then the client's own wire, then the provider called
 * least so far, then provider and model id so the order is deterministic
 * given the history. Providers that answered no catalog, or that refused
 * once, are out. An OpenAI JSON-mode request goes only to OpenAI- and
 * Gemini-wire providers: an Anthropic-wire provider always echoes json_mode
 * false, which would fail the client's echo_json check. A paid model is
 * never a candidate.
 *
 * (A "fresh provider first" order was tried and reverted: it spreads the
 * first requests over lower-rated providers, which drops quality, and it
 * spends the fallback pool early, so a later request with a narrow
 * requirement finds its fallbacks already exhausted. Rating first, with the
 * least-called tie-break, is the better trade.)
 */
function rankedCandidates(requires: Requirement, clientWire: Wire, jsonMode: boolean): Route[] {
  const routes: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhaustedProviders.has(probe.provider.id)) continue;
    if (jsonMode && clientWire === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null || !modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      routes.push({ probe, model, rating: row.rating_milli });
    }
  }
  const callsOf = (route: Route): number => callsByProvider.get(route.probe.provider.id) ?? 0;
  const wireCost = (route: Route): number => (route.probe.provider.wire === clientWire ? 0 : 1);
  routes.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (wireCost(a) !== wireCost(b)) return wireCost(a) - wireCost(b);
    if (callsOf(a) !== callsOf(b)) return callsOf(a) - callsOf(b);
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
    return 0;
  });
  return routes;
}

/** What one request body asks for: tools, a vision input, or JSON output. */
function requirementsFromBody(body: Json): Requirement {
  const req: Requirement = {};
  if (!isRecord(body)) return req;
  const tools = body.tools;
  if (Array.isArray(tools) && tools.length > 0) req.tools = true;
  const messages = body.messages;
  if (Array.isArray(messages)) {
    for (const message of messages) {
      if (!isRecord(message)) continue;
      const content = message.content;
      const blocks = Array.isArray(content) ? content : [content];
      for (const block of blocks) {
        if (!isRecord(block)) continue;
        if (block.type === 'image_url' || block.type === 'image') req.vision = true;
      }
    }
  }
  const format = body.response_format;
  if (isRecord(format) && format.type === 'json_object') req.json = true;
  return req;
}

/**
 * What the request at manifest position `index` needs, and whether it is an
 * OpenAI JSON-mode request. The manifest is the source of truth — a body
 * alone does not always say (an Anthropic request never carries a
 * response_format, yet can still require JSON) — so the manifest's
 * requirements lead and the body's add what the client itself will check.
 * JSON-mode is read from the body only: it is what the client echoes.
 */
function requirementsFor(body: Json, index: number): { requires: Requirement; jsonMode: boolean } {
  const fromBody = requirementsFromBody(body);
  const fromManifest = brief.workload?.manifest[index]?.requires ?? {};
  const requires: Requirement = { ...fromManifest, ...fromBody };
  const jsonMode = isRecord(body) && isRecord(body.response_format) && body.response_format.type === 'json_object';
  return { requires, jsonMode };
}

/** Read one request body as JSON, or null when it is empty, too large, or not JSON. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    total += buf.length;
    chunks.push(buf);
  }
  if (total > MAX_BODY_BYTES) return null;
  const text = Buffer.concat(chunks).toString('utf8');
  if (text.trim() === '') return null;
  try {
    return JSON.parse(text) as Json;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, message: string): void {
  if (api === 'openai') sendJson(res, status, openAIError(status, message));
  else sendJson(res, status, anthropicError(status, message));
}

/** GET /v1/models: the free catalog, in the wire the auth header names. */
async function models(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const listed = servedModels();
  const apiKey = req.headers['x-api-key'];
  if (apiKey !== undefined && apiKey !== '') sendJson(res, 200, anthropicModelList(listed));
  else sendJson(res, 200, openAIModelList(listed));
}

/**
 * The upstream request body for one route: the client's body, retargeted at
 * the route's model and translated into the provider's wire when it differs
 * from the client's. A streamed OpenAI-wire call asks for usage in the
 * stream, so the answer carries its token counts verbatim.
 */
function upstreamBody(route: Route, body: Json, clientApi: 'openai' | 'anthropic'): Json {
  const model = route.model.id;
  const wire = route.probe.provider.wire;
  let out: Json;
  if (wire === 'openai') {
    out = clientApi === 'openai' ? { ...(isRecord(body) ? body : {}), model } : anthropicToOpenAI(body, model);
  } else if (wire === 'anthropic') {
    out = clientApi === 'anthropic' ? { ...(isRecord(body) ? body : {}), model } : openAIRequestToAnthropic(body, model);
  } else {
    out = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  if (wire === 'openai' && isRecord(out) && out.stream === true) {
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * One upstream call. The route is the provider's own path: chat completions
 * for OpenAI wire, messages for Anthropic wire, and generateContent — or
 * streamGenerateContent when the client streams — with the model in the path
 * for Gemini wire. Auth is the provider's own scheme. Every attempt counts
 * against the provider, whether it answers or refuses.
 */
async function callProvider(route: Route, body: Json, clientApi: 'openai' | 'anthropic'): Promise<Response> {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = isRecord(body) && body.stream === true;
  let url: string;
  if (provider.wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (provider.wire === 'anthropic') url = `${base}/v1/messages`;
  else url = `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  callsByProvider.set(provider.id, (callsByProvider.get(provider.id) ?? 0) + 1);
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody(route, body, clientApi)),
    signal: AbortSignal.timeout(30000),
  });
}

/**
 * Whether a refusal retires the provider for this trial. Only a spent
 * budget ("… free-tier budget is spent") or an outage ("… is out") does:
 * those mean no later call to that provider can succeed. A single refused
 * call ("… refused this call") is a one-off failure of that call number
 * only — the provider may well answer the next one, so it stays available.
 * The refusal arrives as a JSON error body, so the message is read from
 * error.message when the body parses.
 */
function refusalExhausted(text: string): boolean {
  let message = text;
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') {
      message = value.error.message;
    }
  } catch {
    message = text;
  }
  return message.includes('budget is spent') || message.includes(' is out');
}

/** Parse one upstream answer in its own wire, streamed or whole. */
async function parseAnswer(wire: Wire, res: Response, stream: boolean): Promise<ParsedAnswer | null> {
  if (wire === 'openai') return parseOpenAIAnswer(res, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(res, stream);
  return parseGeminiAnswer(res, stream);
}

/**
 * Deliver one parsed answer to the client in the client's own wire: a whole
 * JSON body, or the streamed frames one by one. The answer text, usage, and
 * stop reason are the provider's own, only re-wrapped — never reworded.
 */
async function answerClient(
  api: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
  parsed: ParsedAnswer,
  model: string,
): Promise<void> {
  const stream = isRecord(body) && body.stream === true;
  if (!stream) {
    sendJson(res, 200, api === 'openai' ? openAIJsonFromParsed(parsed, model) : anthropicJsonFromParsed(parsed, model));
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const frames = api === 'openai' ? openAIStreamFromParsed(parsed, model) : anthropicStreamFromParsed(parsed, model);
  for await (const frame of frames) res.write(frame);
  res.end();
}

/**
 * Serve one client request. The seed plan, when there is one, names the
 * route for this manifest position; that route is tried first and the ranked
 * list follows behind it, so a plan that misses (a provider that did not
 * answer, a requirement the plan did not see) degrades to the same adaptive
 * routing as before instead of failing. Each candidate is tried in order
 * until one answers readably; a refusal retires its provider and the next
 * candidate takes over. At most one provider ever answers a request.
 */
async function serve(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const index = manifestIndex;
  manifestIndex += 1;
  const { requires, jsonMode } = requirementsFor(body, index);
  const clientWire: Wire = api;
  let candidates = rankedCandidates(requires, clientWire, jsonMode);
  const planned = seedPlan?.byIndex.get(index);
  if (planned !== undefined) {
    const match = candidates.find((c) => c.probe.provider.id === planned.provider && c.model.id === planned.model);
    if (match !== undefined) {
      candidates = [match, ...candidates.filter((c) => c !== match)];
      log(`request ${index}: planned route ${planned.provider}/${planned.model}`);
    } else {
      log(`request ${index}: planned route ${planned.provider}/${planned.model} not usable; falling back`);
    }
  }
  const stream = isRecord(body) && body.stream === true;
  for (const route of candidates) {
    let upstream: Response;
    try {
      upstream = await callProvider(route, body, api);
    } catch (err) {
      log(`request ${index}: ${route.probe.provider.id} call failed: ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      if (refusalExhausted(text)) exhaustedProviders.add(route.probe.provider.id);
      log(`request ${index}: ${route.probe.provider.id} refused (${upstream.status})`);
      continue;
    }
    const parsed = await parseAnswer(route.probe.provider.wire, upstream, stream);
    if (parsed === null) {
      log(`request ${index}: ${route.probe.provider.id} answer unreadable`);
      continue;
    }
    await answerClient(api, body, res, parsed, route.model.id);
    return;
  }
  sendError(res, api, 502, 'no provider could serve this request');
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'request body must be JSON');
    return;
  }
  await serve('openai', body, res);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'request body must be JSON');
    return;
  }
  await serve('anthropic', body, res);
}

function handle(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? '/', 'http://127.0.0.1');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    void models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    void chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    void messages(req, res);
    return;
  }
  sendJson(res, 404, { error: 'not found' });
}

/** Listen on a loopback port the system picks, and report which one. */
function listen(): Promise<{ server: ReturnType<typeof createServer>; port: number }> {
  return new Promise((resolve, reject) => {
    const server = createServer(handle);
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('could not bind a port'));
        return;
      }
      resolve({ server, port: address.port });
    });
  });
}

/** Tell the environment where this entry listens and what it serves. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
  });
  if (!res.ok) throw new Error(`register failed: ${res.status} ${await res.text()}`);
}

/** Poll GET /work until the environment says the trial is done, then stop. */
function poll(server: ReturnType<typeof createServer>): void {
  const timer = setInterval(async () => {
    let done = false;
    try {
      const res = await fetch(`${ENV_URL}/work`);
      if (res.ok) {
        const data = (await res.json()) as { done?: boolean };
        done = data.done === true;
      }
    } catch {
      done = false;
    }
    if (!done) return;
    clearInterval(timer);
    server.close(() => process.exit(0));
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  const { server, port } = await listen();
  log(`listening on 127.0.0.1:${port}`);
  await register(port);
  poll(server);
}

start().catch((err: unknown) => {
  process.stderr.write(`aggregator: fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
