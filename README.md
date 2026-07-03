# Precedent

**A side-by-side visualizer for GraphRAG vs. vector RAG on legislative precedent.**

Ask a question about a bill — *"How likely is this to pass?"*, *"What happened to
similar bills?"* — and Precedent runs it through **two retrieval engines at once**
and shows you *how each one works*, step by step, live:

- **GraphRAG** reasons over a knowledge graph of bills, sponsors, committees, and
  outcomes. It finds precedent bills that are *structurally* connected to your
  question — shared sponsors, shared committees — and grounds its answer in what
  actually happened to them.
- **Vector RAG** ranks bill passages by *semantic (lexical) similarity* to your
  question. No structure, just "which text is most alike".

The frontend is built for a developer: it animates each engine's pipeline as it
executes, shows every intermediate artifact (keywords, the precedent subgraph,
the ranked chunks, the exact prompt), lets you **tune the RAG parameters** and
re-run to watch retrieval change, and lets you **peer into the source code**
behind any step with one click.

---

## Why this is interesting

GraphRAG and vector RAG answer the same question by completely different means,
and Precedent makes the difference *visible* rather than asserting it. On the
built-in demo data the two engines overlap only ~19% in what they retrieve — a
graph query about drug pricing surfaces a rural-hospital bill through a *shared
sponsor* that pure text similarity never connects. That divergence is the whole
point: structural relevance vs. lexical relevance, shown honestly, tradeoffs and
all.

---

## Quickstart

Everything runs in-process by default: an **in-memory graph** (no Neo4j) and an
**embedded Chroma** index (no server). With no Anthropic key, each engine returns
a deterministic **extractive** answer built from its retrieved context, so the
app is fully functional out of the box.

**One command** (from the repo root, after creating the venv once):

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
./dev.sh
```

`./dev.sh` starts the API (:8080) and the frontend (:5173) together and shuts
both down on Ctrl-C. Open **http://localhost:5173** — that's the app; the API on
:8080 is just the backend.

Prefer two terminals? Backend: `uvicorn precedent.api.main:app --reload --port 8080`.
Frontend: `cd frontend && npm run dev`. (zsh note: don't paste trailing `#`
comments — interactive zsh passes `#` to the command as an argument.)

Type a question, hit **Run both engines**, and watch the two pipelines execute.
Open the **RAG parameters** panel, change the chunking strategy or a graph
weight, and re-run to see retrieval shift. Switch to the **Knowledge graph** tab
to explore the whole bill-relationship space, and after a run hit **Peer into the
pipeline** for a slow-motion, code-level walkthrough.

### Enable model-generated answers + the model toggle (optional)

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY=... in .env
```

With a key set, the final "generate answer" step writes the grounded analysis
with an LLM instead of the extractive fallback. The **model is chosen in the UI**
from a dropdown (Opus 4.8 / Sonnet 5 / Haiku 4.5) and applies to both engines —
handy for comparing how model choice affects the same grounded answer.

Routing goes through **LiteLLM**, so the calling code (`engine/base.generate_answer`)
is one provider-agnostic `completion(model=..., messages=...)` call. Adding
Gemini or GPT to the toggle later is just more entries in
`assembly/model_config.AVAILABLE_MODELS` plus that provider's API key — no code
change at the call site. The UI reads the list from `GET /models`.

---

## Full stack in Docker (Neo4j + Chroma + API + frontend)

```bash
docker compose -f infra/docker-compose.yml up --build
# API at :8080, visualizer at :8081, Neo4j browser at :7474
```

Compose sets `GRAPH_BACKEND=neo4j` and `CHROMA_MODE=server`, so the exact same
code runs against real backends. `ANTHROPIC_API_KEY` is passed through from your
shell if set.

---

## How it works

```
                       ┌───────────────────────────────────────────────┐
   question ──────────▶│                 Switchboard                    │
                       │  (builds stores + engines from Settings, once) │
                       └───────┬───────────────────────────────┬───────┘
                               │                               │
                   ┌───────────▼──────────┐         ┌──────────▼───────────┐
                   │     GraphEngine       │         │     VectorEngine      │
                   │ parse → seed → expand │         │ embed → search → rank │
                   │  → score → subgraph   │         │   → build context     │
                   │  → context → answer   │         │      → answer         │
                   └───────────┬──────────┘         └──────────┬───────────┘
                               │  GraphStore                    │  VectorStore
                   ┌───────────▼──────────┐         ┌──────────▼───────────┐
                   │ InMemory | Neo4j      │         │ Chroma (embedded|srv) │
                   └──────────────────────┘         └──────────────────────┘

Each engine runs as a generator of TraceStep events → streamed over SSE →
animated live in the frontend, with every step linked to its source code.
```

### The retrieval, in one line each

- **GraphRAG** (`stores/graph/queries.py`): match the query to seed bills by
  subject → walk to their sponsors/committees → find other bills sharing those →
  rank by connection strength (tunable weights) → assemble a subgraph.
- **Vector RAG** (`engine/vector_engine.py`): embed the query → cosine-rank bill
  chunks → return the top passages. Change the chunk size/overlap/embedder
  dimension and the corpus is re-chunked in-process so you can *see* the effect.

### Tunable RAG parameters (the educational core)

Exposed in the UI and via the API (`GET /params`), applied per query:

| GraphRAG | Vector RAG |
|---|---|
| top-k precedents | top-k chunks |
| number of seed bills | **chunking strategy** (recursive / fixed / sentence / sliding / semantic / whole) |
| **traversal depth (1–3 hops)** | chunk size + overlap (chars) |
| shared-sponsor weight | embedder dimensions |
| shared-committee weight | **similarity threshold** |
| subject-overlap weight | |

Every knob has an ⓘ tooltip explaining what it is and its trade-offs, and the
chunking-strategy dropdown describes each strategy inline — the app is meant to
be self-teaching. A live **progress bar** tracks retrieval as the pipeline
streams.

Changing a vector *indexing* knob (strategy, size, overlap, dimension) re-chunks
and re-embeds the corpus in-process for that query, so the chunk map updates to
show the new boundaries. Graph `hops=2+` follows friends-of-friends through the
graph, reaching precedents no direct link (or text match) would surface.

### Three ways to look inside

- **Peer into the code** — every trace step names the function that produced it;
  click **view source** to fetch its real source (`GET /source`, via `inspect`)
  with line numbers.
- **Peer mode** — after a run, a slow-motion walkthrough that steps through every
  stage of both engines, showing each step's intermediate data *and* its source,
  played hands-free or stepped manually (▶ / arrow keys).
- **Knowledge graph tab** — a standalone, interactive explorer of the whole
  bill-relationship space (`GET /graph/full`): pan, zoom, filter by subject or
  congress, and click a bill to light up everything it shares sponsors and
  committees with.

---

## Design choices worth knowing

- **Runs anywhere, scales up by config.** The `GraphStore` interface has two
  backends (in-memory NetworkX-style + Neo4j); Chroma runs embedded or as a
  server. One env var switches each. Engines don't know or care which is live.
- **Transparent, offline embedder by default.** The default embedder is a
  hashing term-frequency model (`preprocessing/embedding.py`) — no model
  download, deterministic, and it literally *embodies* the "lexical relevance"
  that vector RAG represents. Swap in `sentence-transformers` by changing one
  factory function.
- **Claude is optional, never required.** No key → a clear extractive answer;
  a key → Claude generates the analysis. The app is always runnable.
- **Fair comparison by construction.** Both engines share the same corpus, same
  system prompt, and same model call. Only the *retrieved context* differs.

---

## Evaluation

`eval/` scores both engines' retrieval against a labelled golden set:

```bash
python eval/run_eval.py
```

Reports precision@k, recall@k, MRR per engine, plus the Jaccard overlap between
the two engines' retrieved sets (lower = they reason more differently). Add your
own queries to `eval/golden_set.json`.

---

## Real data: ingest from GovInfo

The demo ships with a curated 16-bill seed (`scripts/build_seed.py`) chosen so the
graph has real structure out of the box. To run against **real legislative data**,
ingest from GovInfo:

```bash
# get a free key at https://api.govinfo.gov, put GOVINFO_API_KEY in .env
python scripts/ingest.py --congress 118 --limit 300
# grow it further / add another Congress:
python scripts/ingest.py --congress 119 --since 2025-01-01 --limit 500 --append
```

This discovers BILLSTATUS packages for the Congress, fetches and parses each bill
(sponsors, cosponsors, committees, subjects, summary, derived outcome), and writes
`data/bills.json`. On the next start the app loads that in preference to the seed —
so one ingestion run upgrades the whole thing from 16 demo bills to hundreds of
real ones, and GraphRAG builds far more relationships. The health badge shows
whether you're on `seed` or `ingested` data.

Every bill is one API call and GovInfo rate-limits default keys (~1000/hour), so
ingestion is bounded by `--limit`; run it repeatedly with `--append` to grow the
corpus.

---

## Project layout

```
src/precedent/
  assembly/      app_config (Settings), model_config, switchboard (wiring)
  engine/        base (trace + Claude call), graph_engine, vector_engine
  stores/graph/  schema, loader (InMemory + Neo4j), queries (the GraphRAG algo)
  stores/vector/ index_config (Chroma), loader (chunk→embed→search)
  preprocessing/ chunking, embedding, entity_extraction, ingestion/, parsers/
  response/      formatter, logger
  api/main.py    FastAPI: /query, /query/stream (SSE), /source, /params, /models, /health
  params.py      tunable RAG knobs
  models.py      the shared bill-record shape + helpers
frontend/        Vite + React developer visualizer
eval/            metrics, run_eval, golden_set
scripts/         build_seed.py
infra/           Dockerfiles + docker-compose
```

## Commands

```bash
make api        # run the API locally
make frontend   # run the visualizer
make test       # pytest
make lint       # ruff + mypy
make compose    # full stack in Docker
```
