// Thin client for the Precedent API.
//
// The base URL is configurable (VITE_API_BASE); by default it points at the
// Vite dev proxy at /api, which forwards to the FastAPI server. The one
// interesting function here is streamQuery: it opens an EventSource against the
// live SSE endpoint and calls back on every trace step, which is what drives the
// pipeline animation. Everything else is a plain fetch.

const BASE = import.meta.env.VITE_API_BASE || "/api";

export async function getHealth() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

export async function getParams() {
  const res = await fetch(`${BASE}/params`);
  return res.json();
}

export async function getModels() {
  const res = await fetch(`${BASE}/models`);
  return res.json();
}

export async function getBills() {
  const res = await fetch(`${BASE}/bills`);
  return res.json();
}

// How a single bill's text splits into chunks under the given params — drives
// the chunk-highlighting view. Re-fetch when the chunk-size slider moves.
export async function getBillChunks(billId, chunkSize, chunkOverlap, chunkStrategy) {
  const p = new URLSearchParams();
  if (chunkSize != null) p.set("chunk_size", chunkSize);
  if (chunkOverlap != null) p.set("chunk_overlap", chunkOverlap);
  if (chunkStrategy != null) p.set("chunk_strategy", chunkStrategy);
  const res = await fetch(`${BASE}/bill/${encodeURIComponent(billId)}/chunks?${p}`);
  if (!res.ok) throw new Error(`chunk lookup failed (${res.status})`);
  return res.json();
}

export async function getGraphMeta() {
  const res = await fetch(`${BASE}/graph/meta`);
  return res.json();
}

export async function getGraphFull({ subject, congress, minShared, limit }) {
  const p = new URLSearchParams();
  if (subject) p.set("subject", subject);
  if (congress) p.set("congress", congress);
  if (minShared != null) p.set("min_shared", minShared);
  if (limit != null) p.set("limit", limit);
  const res = await fetch(`${BASE}/graph/full?${p}`);
  if (!res.ok) throw new Error(`graph fetch failed (${res.status})`);
  return res.json();
}

export async function getSource(symbol) {
  const res = await fetch(`${BASE}/source?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error(`source lookup failed (${res.status})`);
  return res.json();
}

// Map the flat control params into the query string the stream endpoint expects.
function buildStreamUrl(query, graphParams, vectorParams, model) {
  const p = new URLSearchParams({ q: query });
  const g = graphParams || {};
  const v = vectorParams || {};
  if (model) p.set("model", model);
  if (g.top_k != null) p.set("graph_top_k", g.top_k);
  if (g.seed_limit != null) p.set("seed_limit", g.seed_limit);
  if (g.hops != null) p.set("hops", g.hops);
  if (g.legislator_weight != null) p.set("legislator_weight", g.legislator_weight);
  if (g.committee_weight != null) p.set("committee_weight", g.committee_weight);
  if (g.subject_weight != null) p.set("subject_weight", g.subject_weight);
  if (v.top_k != null) p.set("vector_top_k", v.top_k);
  if (v.chunk_strategy != null) p.set("chunk_strategy", v.chunk_strategy);
  if (v.chunk_size != null) p.set("chunk_size", v.chunk_size);
  if (v.chunk_overlap != null) p.set("chunk_overlap", v.chunk_overlap);
  if (v.embedder_dim != null) p.set("embedder_dim", v.embedder_dim);
  if (v.similarity_threshold != null) p.set("similarity_threshold", v.similarity_threshold);
  return `${BASE}/query/stream?${p.toString()}`;
}

// Open the live trace stream. onStep(step) fires per trace event; onDone() when
// both pipelines finish; onError() on transport failure. Returns a close fn.
export function streamQuery(query, graphParams, vectorParams, model, { onStep, onDone, onError }) {
  const url = buildStreamUrl(query, graphParams, vectorParams, model);
  const es = new EventSource(url);

  es.addEventListener("step", (e) => {
    try {
      onStep(JSON.parse(e.data));
    } catch (err) {
      onError?.(err);
    }
  });
  es.addEventListener("done", () => {
    es.close();
    onDone?.();
  });
  es.onerror = (err) => {
    es.close();
    onError?.(err);
  };

  return () => es.close();
}
