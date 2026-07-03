"""
Tunable RAG parameters -- the knobs a user can turn per query to *see* how RAG
design choices change the output.

This is the educational core of Precedent: the same question, run with a
different chunk size or a different graph scoring weight, retrieves different
evidence and can produce a different answer. Exposing these as per-request
parameters (rather than hard-coded constants) lets the front end offer sliders
and lets a learner build intuition for what each choice actually does.

Every field has a default that reproduces the app's standard behaviour, so a
request that sends no params behaves exactly like the baseline. ``from_dict``
ignores unknown keys and falls back to defaults for missing ones, so the API
can hand a partial, user-supplied dict straight in without validation ceremony.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from precedent.preprocessing.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_STRATEGY,
)

# The embedder dimension the persistent (baseline) index is built with. Must
# match HashingEmbedder's default in preprocessing/embedding.py -- a request
# asking for this dimension can use the baseline index; any other dimension
# forces the in-process re-index path.
_BASELINE_DIM = 512


@dataclass(frozen=True)
class VectorParams:
    """Knobs for the vector-RAG pipeline."""

    top_k: int = 6
    # Chunking strategy: *where* to cut the text (sentence/fixed/word/whole).
    chunk_strategy: str = DEFAULT_STRATEGY
    # Chunk sizing: bigger chunks give more context per hit but blur the match;
    # smaller chunks are more precise but can lose surrounding meaning.
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_OVERLAP
    # Embedder resolution: more buckets separate more terms (less hash collision)
    # at the cost of sparser vectors. Turning this knob shows how embedding
    # capacity changes which passages look "similar".
    embedder_dim: int = _BASELINE_DIM
    # Drop retrieved passages below this cosine similarity -- a precision filter.
    similarity_threshold: float = 0.0

    @property
    def is_custom(self) -> bool:
        """True if any indexing-time knob differs from the baseline index."""
        return (
            self.chunk_strategy != DEFAULT_STRATEGY
            or self.chunk_size != DEFAULT_CHUNK_SIZE
            or self.chunk_overlap != DEFAULT_OVERLAP
            or self.embedder_dim != _BASELINE_DIM
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VectorParams":
        return _from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphParams:
    """Knobs for the GraphRAG pipeline."""

    top_k: int = 6
    seed_limit: int = 3  # how many topical seed bills to enter the graph from
    hops: int = 1  # graph traversal depth: 1 = direct neighbours, 2 = friends-of-friends
    # Scoring weights: how much a shared sponsor vs. a shared committee vs. a
    # shared subject counts toward "graph relevance". Raising the legislator
    # weight, say, makes the engine prefer precedents by the same people.
    legislator_weight: float = 2.0
    committee_weight: float = 1.0
    subject_weight: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GraphParams":
        return _from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(cls, data: dict[str, Any] | None):
    """Build a params dataclass from a partial dict, ignoring unknown keys."""
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known and v is not None}
    return cls(**filtered)
