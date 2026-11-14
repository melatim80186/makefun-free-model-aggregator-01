# The best free AI model aggregator

One server (`main.ts`, Node 22, standard library only) that serves the OpenAI
Chat Completions API (`POST /v1/chat/completions`) and the Anthropic Messages
API (`POST /v1/messages`), in JSON and in server-sent events, over the ten
simulated free-tier providers the trial's brief names. At trial start it
probes every provider's catalog route, registers `{protocol_version: 1, port,
catalog}` at `POST $MAKEFUN_ENV_URL/register`, serves the twenty requests the
env's client sends (one at a time), polls `GET /work`, and exits 0 when done.
`node main.ts --list` prints the registered catalog byte for byte.

## Routing (the part that wins the points)

Candidates are the free models meeting the request's requirements (from the
brief's quality table and the probed listings; a paid model is never picked),
ranked by `rating_milli` descending. Ties break toward the client's own wire
(no translation to get wrong), then toward the provider called least so far
(spreads the per-trial budgets), then by provider and model id, so the order
is deterministic given the history.

- **JSON-mode OpenAI requests go only to OpenAI- or Gemini-wire providers.**
  An Anthropic-wire provider always echoes `json_mode: false`, which would
  fail the client's `echo_json` check. (Anthropic-client JSON requests carry
  no JSON-mode flag, so any wire serves them — only the model's `json: true`
  capability matters.)
- **Failover:** a 402, 429 or 5xx is never the answer. The next candidate
  serves the request; a refusal whose message says the budget is spent (or
  the provider "is out") marks that provider exhausted for the trial, so the
  exhaustion costs exactly one call. No retries on the same provider for one
  request: fail schedules often fire on consecutive call numbers, so a retry
  would usually burn a second call for nothing.
- **Wire fidelity:** every answer is rebuilt from the parsed provider
  answer. The upstream request carries the client's system prompt, tools,
  output cap, stop sequences, images and JSON mode faithfully on all three
  wires, so the provider echoes them back whole; the client answer re-emits
  the provider's text whole, its usage counts verbatim, and its stop reason
  in the same family. Streaming always carries a usage chunk (OpenAI
  upstream calls set `stream_options.include_usage: true`; Anthropic and
  Gemini usage is translated into the client's usage fields). Fixed `created`
  timestamps everywhere.
- **Catalog:** ten providers, in listing order, free flags from the quality
  table. `GET /v1/models` lists exactly the registered free ids (deduped).

## Files

- `main.ts` — the server: brief parsing, catalog probe, register, `/work`
  poll, both client routes, rating-ranked routing with failover.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the
  catalog probe, the catalog builder.
- `lib/wire.ts` — the client wires and the upstream wires: Anthropic to and
  from Chat Completions, OpenAI and Anthropic to Gemini, provider-answer
  parsers (JSON and SSE, all three wires), and client-answer builders
  (OpenAI and Anthropic, JSON and SSE).

## Practice scores (kit's `grade-local`, process driver, 3 instances)

| trial | score | quality | served | wire | failover | catalog | calls | replay |
|-------|-------|---------|--------|------|----------|---------|-------|--------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 25 | matched |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 | matched |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 | matched |

The starter baseline scores 38 on trial 0. Quality 99 is the practical
ceiling: the best model (OpenRouter `z-ai/glm-5.3-flash:free`, 1475) has a
roughly 10–12-call per-trial budget but is the best pick for about 17 of the
20 requests, so a few weight-1 requests must fall back to 1470 models no
matter how the budget is spent.

## License

MIT. See `LICENSE`.
