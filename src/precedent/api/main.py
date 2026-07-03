"""
The Precedent HTTP API.

This is the thin transport layer over the switchboard. It exposes:

* ``GET  /health``          -- readiness + how the app is wired.
* ``GET  /params``          -- the tunable RAG knobs and their defaults/ranges,
                               so the front end can render controls generically.
* ``POST /query``           -- run both engines and return the buffered
                               side-by-side comparison (trace + answers).
* ``GET  /query/stream``    -- the same run as a live Server-Sent Events stream,
                               interleaving the two engines' trace steps so the
                               developer front end can animate the pipeline as it
                               executes. (GET, so the browser's EventSource works.)
* ``GET  /source``          -- return the source code of the function behind any
                               trace step, so a developer can peer into exactly
                               what a stage did.
* ``GET  /bills``           -- the loaded corpus, for the picker / overview.

Endpoints stay about transport: they parse params, call the switchboard's
engines, and shape the result via response.formatter. All the retrieval logic
lives behind the switchboard.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from precedent.assembly.model_config import AVAILABLE_MODELS, DEFAULT_MODEL, resolve_model
from precedent.assembly.switchboard import get_switchboard
from precedent.params import GraphParams, VectorParams
from precedent.response.formatter import format_comparison

app = FastAPI(
    title="Precedent API",
    description="GraphRAG vs Vector RAG for legislative precedent, with a live "
    "developer visualizer.",
    version="0.1.0",
)

# The switchboard reads CORS origins from Settings; the app is created at import
# time so we fetch settings directly here for the middleware.
_settings = get_switchboard().settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Body for POST /query: the question plus optional per-engine knobs."""

    query: str
    model: str | None = None  # LiteLLM route id from GET /models; None = default
    graph_params: dict[str, Any] | None = None
    vector_params: dict[str, Any] | None = None


@app.get("/")
def root() -> dict[str, Any]:
    """
    A friendly landing payload so hitting the API root isn't a bare 404.

    The API is not the app -- the visualizer runs separately (Vite on :5173, or
    the built frontend on :8081). This just points there and lists the routes.
    """
    return {
        "service": "Precedent API",
        "note": "This is the API. Open the visualizer at http://localhost:5173 "
        "(dev) or http://localhost:8081 (docker).",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/models",
            "/params",
            "/bills",
            "/bill/{id}/chunks",
            "/graph/full",
            "/graph/meta",
            "/query",
            "/query/stream",
            "/source",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Readiness snapshot: backends, corpus size, whether the LLM is enabled."""
    return get_switchboard().health()


@app.get("/models")
def models() -> dict[str, Any]:
    """
    The models the UI toggle can pick from, plus the default.

    Routed through LiteLLM, so each id is a ``<provider>/<model>`` string. Adding
    a provider later means adding entries in model_config.AVAILABLE_MODELS and
    that provider's key -- this endpoint and the UI need no other change.
    """
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {"id": m.id, "label": m.label, "provider": m.provider} for m in AVAILABLE_MODELS
        ],
    }


@app.get("/params")
def params_schema() -> dict[str, Any]:
    """
    The tunable RAG parameters with defaults and suggested ranges.

    The front end renders a slider/control per entry from this, so adding a new
    knob is a one-line change here plus honouring it in the params dataclass --
    the UI doesn't need to hard-code the list.
    """
    from precedent.preprocessing.chunking import STRATEGY_HELP

    return {
        "graph": {
            "defaults": GraphParams().to_dict(),
            "controls": [
                {
                    "key": "top_k",
                    "label": "Precedents (top-k)",
                    "type": "range",
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "help": "How many precedent bills the graph engine returns. Higher = more "
                    "context for the answer, but weaker/less-related bills creep in.",
                },
                {
                    "key": "seed_limit",
                    "label": "Seed bills",
                    "type": "range",
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "help": "How many topically-matched bills are used as entry points into the "
                    "graph. More seeds cast a wider net; fewer keep the search focused.",
                },
                {
                    "key": "hops",
                    "label": "Traversal depth (hops)",
                    "type": "range",
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "help": "How far to walk the graph. 1 hop = bills that directly share a "
                    "sponsor or committee with a seed. 2 hops = friends-of-friends "
                    "(added at a decayed weight). More hops reach less-obvious "
                    "precedents that text search can't, but add noise.",
                },
                {
                    "key": "legislator_weight",
                    "label": "Shared-sponsor weight",
                    "type": "range",
                    "min": 0,
                    "max": 5,
                    "step": 0.5,
                    "help": "How much sharing a sponsor/cosponsor counts toward relevance. A "
                    "shared person is specific evidence, so it's weighted highest by "
                    "default. Raise it to prefer precedents by the same legislators.",
                },
                {
                    "key": "committee_weight",
                    "label": "Shared-committee weight",
                    "type": "range",
                    "min": 0,
                    "max": 5,
                    "step": 0.5,
                    "help": "How much sharing a committee counts. Committees are broader than "
                    "individual sponsors, so this is weighted lower by default.",
                },
                {
                    "key": "subject_weight",
                    "label": "Subject-overlap weight",
                    "type": "range",
                    "min": 0,
                    "max": 5,
                    "step": 0.5,
                    "help": "A tie-breaker rewarding precedents that also share a policy subject "
                    "with the query — nudges ranking toward bills that are both "
                    "connected and on-topic.",
                },
            ],
        },
        "vector": {
            "defaults": VectorParams().to_dict(),
            "controls": [
                {
                    "key": "top_k",
                    "label": "Chunks (top-k)",
                    "type": "range",
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "help": "How many text passages to retrieve. Higher gives the answer more "
                    "context but dilutes it with lower-similarity passages.",
                },
                {
                    "key": "chunk_strategy",
                    "label": "Chunking strategy",
                    "type": "select",
                    "help": "Where the bill text is cut into passages before embedding. Each "
                    "strategy retrieves visibly different passages — pick one and open a "
                    "chunk map below to see the boundaries.",
                    "options": [
                        {
                            "value": "recursive",
                            "label": "Recursive (default)",
                            "help": STRATEGY_HELP["recursive"],
                        },
                        {"value": "fixed", "label": "Fixed-size", "help": STRATEGY_HELP["fixed"]},
                        {
                            "value": "sentence",
                            "label": "Sentence",
                            "help": STRATEGY_HELP["sentence"],
                        },
                        {
                            "value": "sliding",
                            "label": "Sliding window",
                            "help": STRATEGY_HELP["sliding"],
                        },
                        {
                            "value": "semantic",
                            "label": "Semantic",
                            "help": STRATEGY_HELP["semantic"],
                        },
                        {"value": "whole", "label": "Whole bill", "help": STRATEGY_HELP["whole"]},
                    ],
                },
                {
                    "key": "chunk_size",
                    "label": "Chunk size (chars)",
                    "type": "range",
                    "min": 80,
                    "max": 1200,
                    "step": 40,
                    "help": "Target passage length. Bigger chunks carry more context per hit but "
                    "blur the match (similarity averages over more text); smaller chunks "
                    "are more precise but can lose surrounding meaning.",
                },
                {
                    "key": "chunk_overlap",
                    "label": "Chunk overlap (chars)",
                    "type": "range",
                    "min": 0,
                    "max": 300,
                    "step": 20,
                    "help": "How much text adjacent chunks share. Overlap keeps an idea that "
                    "straddles a boundary retrievable from either side, at the cost of "
                    "some redundancy.",
                },
                {
                    "key": "embedder_dim",
                    "label": "Embedder dimensions",
                    "type": "range",
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "help": "Size of the embedding vector. More dimensions separate more distinct "
                    "terms (fewer hash collisions) and sharpen similarity; fewer are "
                    "coarser and blur unrelated passages together.",
                },
                {
                    "key": "similarity_threshold",
                    "label": "Min similarity",
                    "type": "range",
                    "min": 0,
                    "max": 0.5,
                    "step": 0.02,
                    "help": "Drop retrieved passages below this cosine similarity. A precision "
                    "filter: raise it to keep only strong matches (and possibly return "
                    "nothing), lower it to keep weak ones.",
                },
            ],
        },
    }


@app.get("/bills")
def bills() -> dict[str, Any]:
    """The loaded corpus, briefly -- for the query picker and the overview."""
    sb = get_switchboard()
    return {
        "count": len(sb.bills),
        "bills": [
            {
                "id": f"{b['congress']}-{b['bill_type'].upper()}-{b['bill_number']}",
                "title": b.get("title"),
                "outcome": b.get("outcome"),
                "subjects": b.get("subjects", []),
            }
            for b in sb.bills
        ],
    }


@app.get("/bill/{bill_id}/chunks")
def bill_chunks(
    bill_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunk_strategy: str | None = None,
) -> dict[str, Any]:
    """
    Return how one bill's text splits into chunks under the given chunking
    params -- the data the UI needs to *highlight* the chunk boundaries.

    Because the split is recomputed here from the same `chunk_bill` the vector
    store uses, changing the chunk-size or strategy control and re-fetching shows
    the exact boundaries the retriever would see. That's the "watch chunking
    change the text" feature made concrete.
    """
    from precedent.models import bill_id as make_id
    from precedent.preprocessing.chunking import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_OVERLAP,
        DEFAULT_STRATEGY,
        chunk_bill,
    )

    sb = get_switchboard()
    bill = next((b for b in sb.bills if make_id(b) == bill_id), None)
    if bill is None:
        raise HTTPException(status_code=404, detail=f"bill not found: {bill_id}")

    chunks = chunk_bill(
        bill,
        chunk_size=chunk_size or DEFAULT_CHUNK_SIZE,
        overlap=chunk_overlap if chunk_overlap is not None else DEFAULT_OVERLAP,
        strategy=chunk_strategy or DEFAULT_STRATEGY,
    )
    return {
        "bill_id": bill_id,
        "title": bill.get("title"),
        "outcome": bill.get("outcome"),
        "chunks": [{"chunk_id": c.chunk_id, "index": c.index, "text": c.text} for c in chunks],
    }


@app.get("/graph/meta")
def graph_meta() -> dict[str, Any]:
    """Distinct subjects and congresses in the graph -- for the explorer filters."""
    from precedent.stores.graph.queries import collect_metadata

    return collect_metadata(get_switchboard().graph_store)


@app.get("/graph/full")
def graph_full(
    subject: str | None = None,
    congress: str | None = None,
    limit: int = 120,
    min_shared: int = 1,
) -> dict[str, Any]:
    """
    The bill-to-bill relationship graph for the standalone knowledge-graph
    explorer: bills are nodes, edges connect bills that share sponsors and/or
    committees, weighted by how much they share. Filters scope the view and
    ``limit`` keeps the most-connected bills so a large corpus stays navigable.
    """
    from precedent.stores.graph.queries import build_bill_graph

    return build_bill_graph(
        get_switchboard().graph_store,
        subject=subject,
        congress=congress,
        limit=limit,
        min_shared=min_shared,
    )


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Run both engines and return the buffered side-by-side comparison."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    sb = get_switchboard()
    model = resolve_model(request.model)
    gp = GraphParams.from_dict({"top_k": sb.settings.top_k, **(request.graph_params or {})})
    vp = VectorParams.from_dict({"top_k": sb.settings.top_k, **(request.vector_params or {})})

    graph_output = sb.graph_engine.execute(request.query, gp, model)
    vector_output = sb.vector_engine.execute(request.query, vp, model)

    sb.query_logger.log(
        request.query,
        graph_output.answer,
        vector_output.answer,
        extra={"model": model, "graph_params": gp.to_dict(), "vector_params": vp.to_dict()},
    )
    return format_comparison(request.query, graph_output, vector_output)


@app.get("/query/stream")
def query_stream(
    q: str = Query(..., description="the question"),
    model: str | None = None,
    graph_top_k: int | None = None,
    seed_limit: int | None = None,
    hops: int | None = None,
    legislator_weight: float | None = None,
    committee_weight: float | None = None,
    subject_weight: float | None = None,
    vector_top_k: int | None = None,
    chunk_strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedder_dim: int | None = None,
    similarity_threshold: float | None = None,
) -> StreamingResponse:
    """
    Stream the two engines' trace steps live as Server-Sent Events.

    The engines are plain generators, so we round-robin between them -- pulling
    one step from each in turn -- and emit each step as it happens. That's what
    lets the front end show both pipelines advancing side by side in real time
    rather than one-then-the-other. A final ``done`` event closes the stream.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    sb = get_switchboard()
    chosen_model = resolve_model(model)
    gp = GraphParams.from_dict(
        {
            "top_k": graph_top_k or sb.settings.top_k,
            "seed_limit": seed_limit,
            "hops": hops,
            "legislator_weight": legislator_weight,
            "committee_weight": committee_weight,
            "subject_weight": subject_weight,
        }
    )
    vp = VectorParams.from_dict(
        {
            "top_k": vector_top_k or sb.settings.top_k,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedder_dim": embedder_dim,
            "similarity_threshold": similarity_threshold,
        }
    )

    def event_stream() -> Iterator[str]:
        graph_gen = sb.graph_engine.run(q, gp, chosen_model)
        vector_gen = sb.vector_engine.run(q, vp, chosen_model)
        active = [graph_gen, vector_gen]
        answers: dict[str, dict[str, Any]] = {}

        while active:
            for gen in list(active):
                try:
                    step = next(gen)
                except StopIteration:
                    active.remove(gen)
                    continue
                if step.step == "answer" and step.phase == "done" and step.payload:
                    answers[step.engine] = step.payload
                yield _sse("step", step.to_dict())

        # Log the completed comparison, then signal completion to the client.
        sb.query_logger.log(
            q,
            answers.get("graph", {}).get("answer", ""),
            answers.get("vector", {}).get("answer", ""),
            extra={
                "model": chosen_model,
                "graph_params": gp.to_dict(),
                "vector_params": vp.to_dict(),
            },
        )
        yield _sse("done", {"query": q})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/source")
def source(
    symbol: str = Query(..., description="dotted path, e.g. module.func or module.Class.method"),
) -> dict[str, Any]:
    """
    Return the source code of the function behind a trace step.

    This is what powers "peer into the code": each trace step carries the dotted
    path of the function that produced it, and the front end fetches it here to
    show the developer the exact implementation. Restricted to this project's
    own package so the endpoint can't be used to read arbitrary modules.
    """
    if not symbol.startswith("precedent."):
        raise HTTPException(status_code=400, detail="symbol must be within the precedent package")

    obj = _resolve_symbol(symbol)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"symbol not found: {symbol}")
    try:
        code = inspect.getsource(obj)
        source_file = inspect.getsourcefile(obj)
        _, start_line = inspect.getsourcelines(obj)
    except (OSError, TypeError) as exc:
        raise HTTPException(status_code=404, detail=f"source unavailable: {exc}") from exc

    # Present the file path relative to the package root for a clean UI label.
    rel = source_file.split("precedent/", 1)[-1] if source_file else symbol
    return {"symbol": symbol, "file": f"precedent/{rel}", "start_line": start_line, "source": code}


def _resolve_symbol(symbol: str) -> Any:
    """
    Resolve a dotted path to a function or method object.

    Walks from the longest importable module prefix down through attribute
    accesses, so both ``module.function`` and ``module.Class.method`` resolve.
    """
    parts = symbol.split(".")
    # Find the longest prefix that imports as a module.
    module = None
    attr_parts: list[str] = []
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            attr_parts = parts[i:]
            break
        except ModuleNotFoundError:
            continue
    if module is None:
        return None
    obj: Any = module
    for attr in attr_parts:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
