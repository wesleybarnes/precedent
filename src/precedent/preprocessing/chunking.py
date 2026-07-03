"""
Splitting a bill into the retrievable chunks the vector engine ranks.

Vector RAG doesn't retrieve whole bills -- it retrieves *passages*. A user's
question usually concerns one provision, and returning the single most relevant
paragraph is both more precise and more honest about *why* it matched than
handing back a 40-page document. So each bill's text is split here into
overlapping, sentence-aware chunks, and those chunks -- not the bills -- are
what get embedded and searched.

The splitter is a small, readable recursive character splitter rather than a
library call, because the visualizer lets a developer peer into exactly how a
bill became the chunks they see ranked, and that story should be one short
function they can read, not an opaque dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from precedent.models import bill_id

# Target chunk size in characters and how much neighbouring chunks overlap.
# Overlap keeps a sentence that straddles a boundary retrievable from either
# side, so a relevant passage isn't lost just because it fell on a seam.
# The default is intentionally small so the demo's short bill summaries split
# into a few visible chunks out of the box -- turn the chunk-size control up to
# see them merge, or down to see them fragment.
DEFAULT_CHUNK_SIZE = 220
DEFAULT_OVERLAP = 40
DEFAULT_STRATEGY = "recursive"

# The chunking strategies a user can toggle between -- the ones actually named in
# RAG practice. Each is a different answer to "where do we cut the text", and
# they retrieve visibly different passages, which is what the strategy control
# lets a learner feel. Human-readable descriptions and trade-offs live in
# STRATEGY_HELP (surfaced as UI tooltips).
STRATEGIES = ("recursive", "fixed", "sentence", "sliding", "semantic", "whole")

STRATEGY_HELP: dict[str, str] = {
    "recursive": "The common default (LangChain's RecursiveCharacterTextSplitter). "
    "Splits on the biggest natural boundary that fits — paragraphs, then "
    "sentences, then words — so chunks stay coherent and near the target size. "
    "Good all-rounder.",
    "fixed": "Fixed-width character windows, ignoring any boundary. Simple and uniform, "
    "but it cuts mid-word and mid-sentence, which can split a relevant idea across "
    "two chunks and hurt retrieval. The naive baseline.",
    "sentence": "One sentence per chunk. Maximally precise — a hit points at an exact "
    "sentence — but loses surrounding context, and very short chunks can match "
    "on stray words. Chunk size is ignored.",
    "sliding": "Fixed windows with heavy (~50%) overlap, sliding across the text. The "
    "overlap means an idea near a boundary is captured whole in at least one "
    "window, at the cost of more, more-redundant chunks.",
    "semantic": "Groups consecutive sentences while they stay on the same topic (measured by "
    "embedding similarity) and starts a new chunk when the topic shifts. Chunks "
    "align to meaning, not length — the 'smart' strategy — but costs an embedding "
    "pass to build.",
    "whole": "No chunking: the whole bill is one passage. Maximum context per hit, but "
    "similarity gets diluted across the whole document, so specific matches are "
    "weaker. The other baseline.",
}


@dataclass
class Chunk:
    """One retrievable passage plus the bill metadata needed to cite it."""

    chunk_id: str  # "{bill_id}#{index}"
    bill_id: str
    index: int
    text: str
    title: str | None
    outcome: str | None


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend a short tail of each chunk onto the next, for continuity."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        out.append((prev[-overlap:] + " " + cur).strip())
    return out


def _split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursive character splitter: greedily pack whole sentences up to the size
    target, and hard-split any single sentence that is itself too long. This is
    the boundary-aware behaviour of the common default splitter.
    """
    chunks: list[str] = []
    current = ""
    for sentence in _sentences(text):
        while len(sentence) > chunk_size:  # a lone over-long sentence: hard split
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sentence[:chunk_size].strip())
            sentence = sentence[chunk_size:]
        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return _apply_overlap([c for c in chunks if c], overlap)


def _split_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-width character windows, stepping by ``chunk_size - overlap``."""
    step = max(1, chunk_size - overlap)
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), step)
        if text[i : i + chunk_size].strip()
    ]


def _split_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """One sentence per chunk -- the finest-grained option."""
    return _sentences(text)


def _split_sliding(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed windows with ~50% overlap sliding across the text."""
    step = max(1, chunk_size // 2)
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), step)
        if text[i : i + chunk_size].strip()
    ]


def _split_semantic(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Group consecutive sentences while they stay on-topic, measured by embedding
    similarity, and start a new chunk when the topic shifts (or the size cap is
    hit). Chunks align to *meaning* rather than length.

    Uses the project's hashing embedder so this stays dependency-free and
    inspectable; the similarity threshold is low because lexical hashing produces
    modest similarities (see preprocessing/embedding.py).
    """
    from precedent.preprocessing.embedding import HashingEmbedder, cosine_similarity

    sentences = _sentences(text)
    if len(sentences) <= 1:
        return [text] if text else []
    embedder = HashingEmbedder(dim=256)
    vectors = embedder.embed(sentences)

    chunks: list[str] = []
    current = [sentences[0]]
    centroid = vectors[0]
    for sentence, vec in zip(sentences[1:], vectors[1:]):
        similar = cosine_similarity(centroid, vec) >= 0.12
        fits = len(" ".join(current)) + len(sentence) + 1 <= chunk_size
        if similar and fits:
            current.append(sentence)
            centroid = [(a + b) / 2 for a, b in zip(centroid, vec)]
        else:
            chunks.append(" ".join(current))
            current = [sentence]
            centroid = vec
    chunks.append(" ".join(current))
    return chunks


_SPLITTERS = {
    "recursive": _split_recursive,
    "fixed": _split_fixed,
    "sentence": _split_sentence,
    "sliding": _split_sliding,
    "semantic": _split_semantic,
}


def _split_text(text: str, chunk_size: int, overlap: int, strategy: str) -> list[str]:
    """Dispatch to the chosen chunking strategy."""
    text = text.strip()
    if not text:
        return []
    if strategy == "whole":
        return [text]  # no chunking -- the whole bill is one passage
    return _SPLITTERS.get(strategy, _split_recursive)(text, chunk_size, overlap)


def chunk_bill(
    bill: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = DEFAULT_STRATEGY,
) -> list[Chunk]:
    """
    Turn one bill into its list of chunks under the chosen strategy.

    The chunked text is the title followed by the plain-language summary -- the
    fields a semantic search should actually match against. Every chunk carries
    the bill id, title, and outcome so a retrieved passage can be cited and its
    real-world result shown without a second lookup.
    """
    bid = bill_id(bill)
    body_parts = [bill.get("title") or "", bill.get("summary") or ""]
    body = ". ".join(part for part in body_parts if part)

    chunks: list[Chunk] = []
    for index, piece in enumerate(_split_text(body, chunk_size, overlap, strategy)):
        chunks.append(
            Chunk(
                chunk_id=f"{bid}#{index}",
                bill_id=bid,
                index=index,
                text=piece,
                title=bill.get("title"),
                outcome=bill.get("outcome"),
            )
        )
    return chunks


def chunk_bills(
    bills: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = DEFAULT_STRATEGY,
) -> list[Chunk]:
    """Chunk a whole batch of bills, flattening the result into one list."""
    out: list[Chunk] = []
    for bill in bills:
        out.extend(chunk_bill(bill, chunk_size, overlap, strategy))
    return out
