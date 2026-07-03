"""
The vector-RAG engine.

It answers the same question the graph engine does, but by a completely
different route: it embeds the query, finds the bill passages whose embeddings
are most similar, and hands Claude a context of those top-ranked chunks. There
is no notion of sponsors, committees, or outcomes-as-connections here -- only
"which text is most lexically similar to what you asked". Running it side by
side with the graph engine is what makes the structural-vs-lexical tradeoff
visible, which is the thing Precedent exists to show.

Two things make this engine especially educational:

* Its ``run`` streams ``TraceStep`` events, so the front end watches each stage
  -- embed, similarity search, rank, build context, answer -- execute live with
  the real query vector and per-chunk scores on display.
* It honours the tunable ``VectorParams``. When a user changes the chunk size,
  overlap, or embedder dimension, the engine re-chunks and re-embeds the corpus
  *in-process for that one query* and ranks with a transparent cosine
  similarity, so the effect of the change is immediately visible in the
  retrieved passages. With default params it uses the persistent Chroma index.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any

from precedent.assembly.app_config import Settings
from precedent.assembly.model_config import ModelConfig, SYSTEM_PROMPT, resolve_model
from precedent.engine.base import Engine, TraceStep, generate_answer, run_step
from precedent.models import outcome_label
from precedent.params import VectorParams
from precedent.preprocessing.chunking import chunk_bills
from precedent.preprocessing.embedding import HashingEmbedder, cosine_similarity
from precedent.stores.vector.loader import ScoredChunk, VectorRetrieval, VectorStore


class VectorEngine(Engine):
    """Retrieval-and-answer over embedded bill passages (semantic similarity)."""

    name = "vector"

    def __init__(
        self,
        store: VectorStore,
        settings: Settings,
        model_config: ModelConfig,
        corpus: list[dict[str, Any]] | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._model_config = model_config
        # The raw bills, kept so a query that changes the chunking or embedding
        # can be re-indexed on the fly without touching the persistent store.
        self._corpus = corpus or []

    def run(
        self, query: str, params: VectorParams | None = None, model: str | None = None
    ) -> Iterator[TraceStep]:
        p = params or VectorParams(top_k=self._settings.top_k)
        resolved_model = resolve_model(model)

        retrieval = yield from run_step(
            engine=self.name,
            step="embed_search",
            title="Embed query & search by similarity",
            description="Map the query to a vector with the same embedder used on every "
            "chunk, then find the nearest passages by cosine similarity. If "
            "you changed the chunking or embedder, the corpus is re-indexed "
            "in-process for this query so the effect is visible immediately.",
            source_symbol="precedent.engine.vector_engine.VectorEngine._search",
            index=0,
            work=lambda: self._search(query, p),
        )

        yield from run_step(
            engine=self.name,
            step="retrieve",
            title="Rank retrieved chunks",
            description="Order the retrieved passages by cosine similarity -- the pure "
            "lexical-relevance signal, with no graph structure involved.",
            source_symbol="precedent.engine.vector_engine.VectorEngine._rank",
            index=1,
            work=lambda: self._rank(retrieval),
        )

        prompt = yield from run_step(
            engine=self.name,
            step="build_context",
            title="Build grounded context",
            description="Concatenate the top-ranked passages (with their bill id and "
            "outcome) into the context handed to Claude.",
            source_symbol="precedent.engine.vector_engine.VectorEngine._build_prompt",
            index=2,
            work=lambda: self._build_prompt(query, retrieval),
        )

        yield from run_step(
            engine=self.name,
            step="answer",
            title="Generate grounded answer",
            description="Ask Claude to assess the query using only the retrieved passages "
            "(or fall back to an extractive summary if no API key is set).",
            source_symbol="precedent.engine.base.generate_answer",
            index=3,
            work=lambda: self._answer(prompt, retrieval, resolved_model),
        )

    # --- stage implementations ---

    def _search(self, query: str, p: VectorParams) -> tuple[VectorRetrieval, dict[str, Any]]:
        if p.is_custom and self._corpus:
            retrieval = self._search_ephemeral(query, p)
            index_mode = "ephemeral (re-chunked for this query)"
        else:
            retrieval = self._store.search(query, p.top_k)
            index_mode = "persistent Chroma index"

        # Precision filter: drop passages below the similarity threshold. Applied
        # to both index paths so the knob behaves the same either way.
        kept = [c for c in retrieval.chunks if c.similarity >= p.similarity_threshold]
        dropped = len(retrieval.chunks) - len(kept)
        retrieval.chunks = kept

        payload = {
            "index_mode": index_mode,
            "chunk_strategy": p.chunk_strategy,
            "embedder": retrieval.embedder_name,
            "vector_dim": retrieval.vector_dim,
            "query_vector_preview": retrieval.query_vector_preview,
            "chunk_count": len(retrieval.chunks),
            "dropped_below_threshold": dropped,
            "params": p.to_dict(),
        }
        return retrieval, payload

    def _search_ephemeral(self, query: str, p: VectorParams) -> VectorRetrieval:
        """
        Re-chunk and re-embed the whole corpus for one query, then rank by
        cosine similarity in plain Python.

        This is the "turn the knob and watch it change" path. It is deliberately
        brute-force and readable -- no index, no server -- because its job is to
        make the effect of a chunking-strategy or embedding choice legible, and a
        learner should be able to open this method and see the whole mechanism:
        chunk (by the chosen strategy), embed, score, sort.
        """
        chunks = chunk_bills(self._corpus, p.chunk_size, p.chunk_overlap, p.chunk_strategy)
        embedder = HashingEmbedder(dim=p.embedder_dim)
        query_vec = embedder.embed_one(query)
        chunk_vecs = embedder.embed([c.text for c in chunks])

        scored: list[ScoredChunk] = []
        for chunk, vec in zip(chunks, chunk_vecs):
            sim = cosine_similarity(query_vec, vec)
            scored.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    bill_id=chunk.bill_id,
                    text=chunk.text,
                    title=chunk.title,
                    outcome=chunk.outcome,
                    similarity=round(sim, 4),
                    distance=round(1.0 - sim, 4),
                )
            )
        scored.sort(key=lambda c: c.similarity, reverse=True)
        return VectorRetrieval(
            query=query,
            query_vector_preview=[round(v, 4) for v in query_vec[:8]],
            vector_dim=embedder.dim,
            embedder_name=embedder.name,
            chunks=scored[: p.top_k],
        )

    def _rank(self, retrieval: VectorRetrieval) -> tuple[VectorRetrieval, dict[str, Any]]:
        payload = {"chunks": [self._chunk_payload(c) for c in retrieval.chunks]}
        return retrieval, payload

    def _build_prompt(self, query: str, retrieval: VectorRetrieval) -> tuple[str, dict[str, Any]]:
        lines = []
        for c in retrieval.chunks:
            lines.append(
                f"- {c.bill_id} (similarity {c.similarity:.2f}, outcome "
                f"{outcome_label(c.outcome)}): {c.text}"
            )
        context = "\n".join(lines) if lines else "(no similar passages found)"
        prompt = (
            f"Question: {query}\n\n"
            f"Most similar bill passages (by semantic similarity):\n{context}\n\n"
            f"Using only these passages, assess the question."
        )
        return prompt, {"prompt": prompt, "prompt_chars": len(prompt)}

    def _answer(
        self, prompt: str, retrieval: VectorRetrieval, model: str
    ) -> tuple[dict, dict[str, Any]]:
        fallback = self._extractive_answer(retrieval)
        result = generate_answer(
            SYSTEM_PROMPT, prompt, fallback, model, self._model_config.max_tokens, self._settings
        )
        return result, result

    # --- helpers ---

    def _extractive_answer(self, retrieval: VectorRetrieval) -> str:
        """Deterministic answer built from the ranked chunks, used with no API key."""
        chunks = retrieval.chunks
        if not chunks:
            return (
                "Semantic search found no bill passages similar to this query, so there is "
                "no text to reason from. Assessment: insufficient similar precedent."
            )
        # Vector RAG has no structural outcome model; the best it can do is note
        # the outcomes of the bills whose text matched, which is exactly the
        # limitation the graph engine is meant to expose.
        outcomes = Counter(c.outcome for c in chunks if c.outcome)
        top = chunks[0]
        outcome_summary = (
            ", ".join(f"{outcome_label(o)} ({n})" for o, n in outcomes.most_common()) or "unknown"
        )
        return (
            f"Semantic search returned {len(chunks)} passage(s) most similar to the query. "
            f"The closest (similarity {top.similarity:.2f}) is from {top.bill_id} "
            f'("{top.title}"). Outcomes of the matched bills: {outcome_summary}.\n'
            f"Assessment: based on lexical similarity alone, treat these as topically "
            f"related rather than causally predictive."
        )

    def _chunk_payload(self, c: ScoredChunk) -> dict[str, Any]:
        return {
            "chunk_id": c.chunk_id,
            "bill_id": c.bill_id,
            "title": c.title,
            "outcome": c.outcome,
            "outcome_label": outcome_label(c.outcome),
            "similarity": c.similarity,
            "distance": c.distance,
            "text": c.text,
        }
