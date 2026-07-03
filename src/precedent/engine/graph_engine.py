"""
The GraphRAG engine.

It answers a question by reasoning over the knowledge graph: it finds the bills
the question is topically about (seeds), fans out to the sponsors and committees
attached to them, follows those connections to structurally-related precedent
bills, ranks the precedents by connection strength, and hands Claude a context
grounded in *why those bills are analogous* -- shared people, shared committees,
and what actually happened to them.

Its ``run`` is written as a sequence of ``yield from run_step(...)`` stages, so
the developer front end sees each stage of the pipeline execute live with its
real intermediate data and timing, and can open the exact function behind any
stage. Every stage's ``source_symbol`` points at the code that did the work.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from precedent.assembly.app_config import Settings
from precedent.assembly.model_config import ModelConfig, SYSTEM_PROMPT, resolve_model
from precedent.engine.base import Engine, TraceStep, generate_answer, run_step
from precedent.models import enactment_rate, outcome_label
from precedent.params import GraphParams
from precedent.stores.graph.loader import GraphStore
from precedent.stores.graph.queries import (
    GraphRetrieval,
    ScoredBill,
    _keywords,
    _seed_bills,
    expand_seeds,
    score_candidates,
)
from precedent.stores.graph.queries import _build_subgraph


class GraphEngine(Engine):
    """Retrieval-and-answer over the legislative knowledge graph."""

    name = "graph"

    def __init__(self, store: GraphStore, settings: Settings, model_config: ModelConfig) -> None:
        self._store = store
        self._settings = settings
        self._model_config = model_config

    def run(
        self, query: str, params: GraphParams | None = None, model: str | None = None
    ) -> Iterator[TraceStep]:
        p = params or GraphParams(top_k=self._settings.top_k)
        resolved_model = resolve_model(model)

        keywords = yield from run_step(
            engine=self.name,
            step="parse_query",
            title="Parse query into keywords",
            description="Lower-case the question and drop stopwords to get the topic "
            "terms used to find an entry point into the graph.",
            source_symbol="precedent.stores.graph.queries._keywords",
            index=0,
            work=lambda: self._parse(query),
        )

        seeds = yield from run_step(
            engine=self.name,
            step="seed_match",
            title="Match seed bills by subject",
            description="Find bills whose policy subjects or title overlap the query "
            "keywords. These seeds are the doorway into the graph.",
            source_symbol="precedent.stores.graph.queries._seed_bills",
            index=1,
            work=lambda: self._seed(keywords, p.seed_limit),
        )

        entities = yield from run_step(
            engine=self.name,
            step="expand_graph",
            title="Expand to sponsors & committees",
            description="Walk from each seed bill to the legislators and committees "
            "attached to it -- the hubs that connect bills to each other.",
            source_symbol="precedent.stores.graph.queries.expand_seeds",
            index=2,
            work=lambda: self._expand(seeds),
        )

        precedents = yield from run_step(
            engine=self.name,
            step="score_precedents",
            title="Traverse & score precedents",
            description="Find every bill sharing one of those hubs, and rank them by how "
            "many sponsors/committees they share with the seeds (graph relevance).",
            source_symbol="precedent.stores.graph.queries.score_candidates",
            index=3,
            work=lambda: self._score(seeds, entities, keywords, p),
        )

        retrieval = yield from run_step(
            engine=self.name,
            step="retrieve",
            title="Assemble precedent subgraph",
            description="Build the node/edge subgraph connecting seeds, shared entities, and "
            "the chosen precedents, and compute the enacted-into-law base rate.",
            source_symbol="precedent.stores.graph.queries._build_subgraph",
            index=4,
            work=lambda: self._assemble(query, keywords, seeds, precedents),
        )

        prompt = yield from run_step(
            engine=self.name,
            step="build_context",
            title="Build grounded context",
            description="Format the ranked precedents and their outcomes into the context "
            "handed to Claude, so every claim can cite a real precedent bill.",
            source_symbol="precedent.engine.graph_engine.GraphEngine._build_prompt",
            index=5,
            work=lambda: self._build_prompt(query, retrieval),
        )

        yield from run_step(
            engine=self.name,
            step="answer",
            title="Generate grounded answer",
            description="Ask Claude to assess the query using only the retrieved precedents "
            "(or fall back to an extractive summary if no API key is set).",
            source_symbol="precedent.engine.base.generate_answer",
            index=6,
            work=lambda: self._answer(prompt, retrieval, resolved_model),
        )

    # --- stage implementations: each returns (result_for_next_stage, ui_payload) ---

    def _parse(self, query: str) -> tuple[set[str], dict[str, Any]]:
        keywords = _keywords(query)
        return keywords, {"keywords": sorted(keywords)}

    def _seed(self, keywords: set[str], seed_limit: int) -> tuple[list[str], dict[str, Any]]:
        seeds = _seed_bills(self._store, keywords, limit=seed_limit)
        return seeds, {"seed_bills": [self._bill_brief(bid) for bid in seeds]}

    def _expand(self, seeds: list[str]) -> tuple[tuple[set[str], set[str]], dict[str, Any]]:
        legislators, committees = expand_seeds(self._store, seeds)
        payload = {
            "legislators": [
                (self._store.legislator(b) or {}).get("full_name", b) for b in legislators
            ],
            "committees": sorted(committees),
        }
        return (legislators, committees), payload

    def _score(
        self,
        seeds: list[str],
        entities: tuple[set[str], set[str]],
        keywords: set[str],
        p: GraphParams,
    ) -> tuple[list[ScoredBill], dict[str, Any]]:
        legislators, committees = entities
        precedents = score_candidates(
            self._store,
            seeds,
            legislators,
            committees,
            keywords,
            p.top_k,
            legislator_weight=p.legislator_weight,
            committee_weight=p.committee_weight,
            subject_weight=p.subject_weight,
            hops=p.hops,
        )
        return precedents, {
            "hops": p.hops,
            "precedents": [self._precedent_payload(pc) for pc in precedents],
        }

    def _assemble(
        self,
        query: str,
        keywords: set[str],
        seeds: list[str],
        precedents: list[ScoredBill],
    ) -> tuple[GraphRetrieval, dict[str, Any]]:
        nodes, edges = _build_subgraph(self._store, seeds, precedents)
        retrieval = GraphRetrieval(
            query=query,
            keywords=sorted(keywords),
            seed_bill_ids=seeds,
            precedents=precedents,
            subgraph_nodes=nodes,
            subgraph_edges=edges,
        )
        rate = enactment_rate([p.bill for p in precedents])
        payload = {
            "seed_bill_ids": seeds,
            "precedents": [self._precedent_payload(p) for p in precedents],
            "subgraph": {"nodes": nodes, "edges": edges},
            "enactment_rate": rate,
        }
        return retrieval, payload

    def _build_prompt(self, query: str, retrieval: GraphRetrieval) -> tuple[str, dict[str, Any]]:
        lines = []
        for p in retrieval.precedents:
            shared = []
            if p.shared_legislators:
                shared.append(f"{len(p.shared_legislators)} shared sponsor(s)")
            if p.shared_committees:
                shared.append(f"{len(p.shared_committees)} shared committee(s)")
            lines.append(
                f"- {p.bill_id} \"{p.bill.get('title')}\": outcome "
                f"{outcome_label(p.bill.get('outcome'))}; "
                f"connected via {', '.join(shared) or 'subject overlap'}."
            )
        context = "\n".join(lines) if lines else "(no connected precedents found)"
        rate = enactment_rate([p.bill for p in retrieval.precedents])
        rate_line = (
            f"{rate:.0%} of these precedents became law."
            if rate is not None
            else "No precedent outcomes available."
        )
        prompt = (
            f"Question: {query}\n\n"
            f"Precedent bills found by graph reasoning (shared sponsors/committees):\n"
            f"{context}\n\nBase rate: {rate_line}\n\n"
            f"Using only these precedents, assess the question."
        )
        return prompt, {"prompt": prompt, "prompt_chars": len(prompt)}

    def _answer(
        self, prompt: str, retrieval: GraphRetrieval, model: str
    ) -> tuple[dict, dict[str, Any]]:
        fallback = self._extractive_answer(retrieval)
        result = generate_answer(
            SYSTEM_PROMPT, prompt, fallback, model, self._model_config.max_tokens, self._settings
        )
        return result, result

    # --- helpers ---

    def _extractive_answer(self, retrieval: GraphRetrieval) -> str:
        """Deterministic answer built from the graph result, used with no API key."""
        precedents = retrieval.precedents
        if not precedents:
            return (
                "Graph reasoning found no bills structurally connected (shared sponsors "
                "or committees) to this query's topic, so there is no grounded precedent "
                "to reason from. Assessment: insufficient precedent."
            )
        rate = enactment_rate([p.bill for p in precedents])
        top = precedents[0]
        summary = (
            f"Graph reasoning surfaced {len(precedents)} precedent bill(s) connected to "
            f"this topic through shared sponsors and committees. "
            f"The closest is {top.bill_id} (\"{top.bill.get('title')}\"), which "
            f"{outcome_label(top.bill.get('outcome')).lower()}. "
            f"Across the connected precedents, {rate:.0%} became law."
        )
        verdict = "likely" if (rate or 0) >= 0.5 else "unlikely"
        return f"{summary}\nAssessment: {verdict} to pass based on connected precedent."

    def _bill_brief(self, bid: str) -> dict[str, Any]:
        bill = self._store.get_bill(bid) or {}
        return {"id": bid, "title": bill.get("title"), "outcome": bill.get("outcome")}

    def _precedent_payload(self, p: ScoredBill) -> dict[str, Any]:
        return {
            "id": p.bill_id,
            "title": p.bill.get("title"),
            "outcome": p.bill.get("outcome"),
            "outcome_label": outcome_label(p.bill.get("outcome")),
            "score": round(p.score, 2),
            "shared_legislators": [
                (self._store.legislator(b) or {}).get("full_name", b) for b in p.shared_legislators
            ],
            "shared_committees": p.shared_committees,
            "subject_overlap": p.subject_overlap,
        }
