"""
The vector store: chunk -> embed -> upsert, and the similarity search the
vector engine runs.

This is the counterpart to the graph store. Where the graph store holds
entities and relationships, this holds bill *passages* as vectors and answers
"which passages are most similar to this query". It wraps a Chroma collection
(embedded or server, per index_config.py) and the project's ``Embedder``, and
returns rich intermediate results so the developer visualizer can show the
query vector, the candidate passages, and each one's similarity score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from precedent.assembly.app_config import Settings
from precedent.preprocessing.chunking import Chunk, chunk_bills
from precedent.preprocessing.embedding import Embedder, build_embedder
from precedent.stores.vector.index_config import build_chroma_client, get_collection

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    """A retrieved passage with the numbers behind its rank."""

    chunk_id: str
    bill_id: str
    text: str
    title: str | None
    outcome: str | None
    similarity: float  # cosine similarity in [-1, 1]; higher is closer
    distance: float  # the raw cosine distance Chroma returned (1 - similarity)


@dataclass
class VectorRetrieval:
    """Everything the vector stage produced -- for the answer and the visualizer."""

    query: str
    query_vector_preview: list[float]  # first few dims, for display only
    vector_dim: int
    embedder_name: str
    chunks: list[ScoredChunk] = field(default_factory=list)


class VectorStore:
    """A Chroma-backed store of embedded bill chunks."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self._settings = settings
        self._embedder = embedder or build_embedder()
        self._client = build_chroma_client(settings)
        self._collection = get_collection(self._client, settings.chroma_collection)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def add_bills(self, bills: list[dict[str, Any]]) -> None:
        """Chunk, embed, and upsert a batch of bills."""
        chunks = chunk_bills(bills)
        self.add_chunks(chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self._embedder.embed([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "bill_id": c.bill_id,
                    "index": c.index,
                    "title": c.title or "",
                    "outcome": c.outcome or "",
                }
                for c in chunks
            ],
        )
        logger.info("VectorStore upserted %d chunks", len(chunks))

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """
        Drop and recreate the collection.

        Used on startup in the local/dev config so the persisted embedded index
        always reflects the *current* chunking defaults -- otherwise a change to
        chunk size would be masked by chunks embedded under the old settings and
        left sitting in data/chroma.
        """
        try:
            self._client.delete_collection(self._settings.chroma_collection)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = get_collection(self._client, self._settings.chroma_collection)

    def search(self, query: str, top_k: int) -> VectorRetrieval:
        """
        Embed the query and return the top_k most similar chunks.

        Chroma reports cosine *distance* (0 = identical, 2 = opposite); we
        convert to the more intuitive similarity (``1 - distance``) for display
        while keeping the raw distance so a curious developer can see both. The
        query vector's first few dimensions are returned too, so the "embed the
        query" stage in the visualizer shows a real number, not a mystery.
        """
        query_vec = self._embedder.embed_one(query)
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        scored: list[ScoredChunk] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, document, meta, distance in zip(ids, documents, metadatas, distances):
            scored.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    bill_id=meta.get("bill_id", ""),
                    text=document,
                    title=meta.get("title") or None,
                    outcome=meta.get("outcome") or None,
                    similarity=round(1.0 - distance, 4),
                    distance=round(distance, 4),
                )
            )

        return VectorRetrieval(
            query=query,
            query_vector_preview=[round(v, 4) for v in query_vec[:8]],
            vector_dim=self._embedder.dim,
            embedder_name=self._embedder.name,
            chunks=scored,
        )
