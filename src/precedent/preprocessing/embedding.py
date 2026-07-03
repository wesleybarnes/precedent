"""
Turning bill text into vectors.

The vector-RAG side of Precedent ranks bill chunks by semantic similarity, and
that starts here: an ``Embedder`` maps text to a fixed-length vector so chunks
and queries live in the same space and cosine similarity means "how alike".

The default embedder is deliberately a transparent, dependency-free
term-frequency hashing embedder rather than a downloaded neural model. Two
reasons, both in the spirit of this project:

* It runs anywhere, offline, with nothing to install or download -- so the
  whole app is demonstrable on a fresh laptop, which is the entire point of the
  local-first design.
* It literally embodies the thing the vector engine is supposed to represent in
  the comparison: *lexical* relevance -- "this text uses the same words about
  the same topic". That is exactly the contrast Precedent draws against the
  graph engine's structural reasoning, so a lexical embedder makes the tradeoff
  honest and legible instead of hiding it inside a black box.

If you want true neural semantics, install ``sentence-transformers`` and swap
``build_embedder`` to return a wrapper around it -- the rest of the code only
depends on the ``Embedder`` interface below, so nothing else changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


def _tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; the shared unit for every embedder here."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Embedder(ABC):
    """Maps text to a fixed-length, comparable vector."""

    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text."""

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single string (e.g. a query)."""
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """
    A hashing term-frequency embedder: bag-of-words, hashed into ``dim`` slots,
    L2-normalised so cosine similarity is just a dot product.

    Every token is hashed to a bucket and its (sub-linear) frequency added
    there. It has no vocabulary to fit and no state to persist -- the same text
    always yields the same vector on any machine -- which is what makes it a
    good, reproducible default for a teaching project. Its "understanding" is
    purely lexical: two chunks are close when they share words, not when they
    share meaning, and that limitation is exactly what the graph engine exists
    to contrast with.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-tf-{dim}"

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _tokenize(text):
                # Sub-linear term weighting dampens very frequent words so a
                # chunk that says "tax" ten times doesn't drown one that uses
                # several on-topic terms once each.
                vec[self._bucket(token)] += 1.0
            for i, value in enumerate(vec):
                if value:
                    vec[i] = 1.0 + math.log(value)
            norm = math.sqrt(sum(v * v for v in vec))
            if norm:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.

    Exposed as a plain function (not buried in a store) because the visualizer
    shows *why* a chunk ranked where it did, and "the score is this dot product"
    is a claim a developer should be able to read the code for and verify.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def build_embedder() -> Embedder:
    """
    Return the configured embedder.

    Currently always the hashing embedder (offline, deterministic, transparent).
    Kept as a factory so a future neural embedder can be selected here without
    touching any call site.
    """
    return HashingEmbedder()
