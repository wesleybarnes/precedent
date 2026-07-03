"""
Shaping the two engines' outputs into the response the front end consumes.

The API runs both engines on one question and returns a single comparison
object. This module is the one place that decides the exact shape of that
object, so the front end has a stable contract and the endpoint code stays
about transport, not formatting.
"""

from __future__ import annotations

from typing import Any

from precedent.engine.base import EngineOutput


def format_comparison(
    query: str,
    graph_output: EngineOutput,
    vector_output: EngineOutput,
) -> dict[str, Any]:
    """
    Combine both engine outputs into the side-by-side comparison payload.

    Each side carries its full step trace (so the buffered ``/query`` response
    can render the same pipeline the streaming endpoint animates), its answer,
    whether that answer was model-generated or extractive, and its
    engine-specific retrieval payload (subgraph for graph, ranked chunks for
    vector).
    """
    return {
        "query": query,
        "graph": _format_side(graph_output),
        "vector": _format_side(vector_output),
    }


def _format_side(output: EngineOutput) -> dict[str, Any]:
    return {
        "engine": output.engine,
        "answer": output.answer,
        "generated": output.generated,
        "model": output.model,
        "note": output.note,
        "retrieval": output.retrieval,
        "steps": output.steps,
    }
