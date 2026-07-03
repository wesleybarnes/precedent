"""
The GraphRAG retrieval algorithm.

This is the heart of the "graph reasoning" side of Precedent. Given a free-text
question, it finds precedent bills the way a legislative analyst would: not by
matching words, but by walking the graph of who-sponsored-what and
which-committee-saw-what, and preferring precedents that are *structurally*
connected to the bills the question is about.

The algorithm is written against the ``GraphStore`` primitive interface only,
so it runs identically on the in-memory and Neo4j backends. It returns a rich
``GraphRetrieval`` object -- not just the answer bills, but every intermediate
artifact (the seed bills, the shared entities, the per-precedent scores, and
the subgraph) -- because the developer-facing visualizer needs to show each of
those stages, and the answer generator needs the ranked precedents.

Stages
------
1. Seed: lexically match the query against bill subjects/titles. This is the
   only lexical step -- it's the doorway into the graph.
2. Expand: collect the legislators and committees attached to the seed bills.
3. Traverse: find every other bill that shares one of those entities.
4. Score: rank those candidate precedents by how many entities they share with
   the seeds (graph connectivity), lightly boosted by subject overlap.
5. Subgraph: assemble the nodes/edges connecting seeds, shared entities, and
   the top precedents, for the visualizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from precedent.stores.graph.loader import GraphStore

# Words too common to be useful as subject-match keys. Kept tiny on purpose --
# this is a doorway heuristic, not the retrieval itself.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "and",
    "or",
    "in",
    "on",
    "is",
    "are",
    "how",
    "what",
    "will",
    "this",
    "that",
    "bill",
    "act",
    "would",
    "likely",
    "pass",
    "passed",
    "similar",
    "bills",
    "happen",
    "happened",
    "new",
    "about",
}


def _keywords(text: str) -> set[str]:
    """Lower-cased, de-stopworded word set used for lexical seed matching."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


@dataclass
class ScoredBill:
    """A candidate precedent bill with the graph evidence behind its rank."""

    bill_id: str
    score: float
    shared_legislators: list[str] = field(default_factory=list)
    shared_committees: list[str] = field(default_factory=list)
    subject_overlap: list[str] = field(default_factory=list)
    bill: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRetrieval:
    """Everything the graph stage produced -- for the answer and the visualizer."""

    query: str
    keywords: list[str]
    seed_bill_ids: list[str]
    precedents: list[ScoredBill]
    subgraph_nodes: list[dict[str, Any]]
    subgraph_edges: list[dict[str, Any]]


def _seed_bills(store: GraphStore, keywords: set[str], limit: int) -> list[str]:
    """
    Find bills whose subjects or title overlap the query keywords.

    Ranks by raw overlap count so the strongest topical matches become the
    seeds the graph traversal then radiates out from. This is intentionally
    simple: the graph structure, not this match, is what does the real work.
    """
    scored: list[tuple[int, str]] = []
    for bid in store.all_bill_ids():
        bill = store.get_bill(bid) or {}
        subject_words = {s.lower() for s in bill.get("subjects", [])}
        subject_words |= _keywords(bill.get("title") or "")
        overlap = len(keywords & subject_words)
        if overlap:
            scored.append((overlap, bid))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [bid for _, bid in scored[:limit]]


def expand_seeds(store: GraphStore, seeds: list[str]) -> tuple[set[str], set[str]]:
    """
    Collect the legislators and committees attached to the seed bills.

    These entities are the graph "hubs" -- the shared people and committees
    through which precedent bills become reachable from the seeds. Returning
    them as their own step lets the visualizer show the fan-out from bills to
    the entities that connect them, which is the moment the reasoning stops
    being lexical and starts being structural.
    """
    seed_legislators: set[str] = set()
    seed_committees: set[str] = set()
    for bid in seeds:
        seed_legislators |= store.legislators_of(bid)
        seed_committees |= store.committees_of(bid)
    return seed_legislators, seed_committees


def score_candidates(
    store: GraphStore,
    seeds: list[str],
    seed_legislators: set[str],
    seed_committees: set[str],
    keywords: set[str],
    top_k: int,
    legislator_weight: float = 2.0,
    committee_weight: float = 1.0,
    subject_weight: float = 0.5,
    hops: int = 1,
) -> list[ScoredBill]:
    """
    Rank precedent bills by how strongly they connect to the seeds.

    Every bill sharing one of the seed entities is a candidate, and we remember
    *which* entities it shares so the score is explainable. By default a shared
    person is weighted more than a shared committee (it's rarer and more
    specific), and a small subject-overlap boost breaks ties toward precedents
    that are both connected and on-topic. The weights are exposed as parameters
    precisely so a learner can turn them and watch the ranking change.

    ``hops`` controls traversal depth. With ``hops=1`` a candidate must share an
    entity *directly* with a seed. With ``hops=2`` the search also follows the
    entities of the hop-1 candidates to reach "friends of friends" -- bills that
    connect to the topic only indirectly -- scored at a decayed weight so the
    directly-connected precedents still rank first. Raising the depth is the
    clearest demonstration of graph reasoning reaching where text never could.
    """
    candidates: dict[str, ScoredBill] = {}
    seed_set = set(seeds)

    def candidate(bid: str) -> ScoredBill:
        if bid not in candidates:
            candidates[bid] = ScoredBill(bill_id=bid, score=0.0, bill=store.get_bill(bid) or {})
        return candidates[bid]

    def connect(
        legislators: set[str], committees: set[str], decay: float, exclude: set[str]
    ) -> set[str]:
        """Attach every bill sharing one of these entities; return the bills hit."""
        reached: set[str] = set()
        for bioguide in legislators:
            for bid in store.bills_of_legislator(bioguide):
                if bid in exclude:
                    continue
                cand = candidate(bid)
                cand.shared_legislators.append(bioguide)
                cand.score += legislator_weight * decay
                reached.add(bid)
        for committee_name in committees:
            for bid in store.bills_of_committee(committee_name):
                if bid in exclude:
                    continue
                cand = candidate(bid)
                cand.shared_committees.append(committee_name)
                cand.score += committee_weight * decay
                reached.add(bid)
        return reached

    # Hop 1: bills directly sharing a seed entity.
    frontier = connect(seed_legislators, seed_committees, decay=1.0, exclude=seed_set)

    # Further hops: expand from the current frontier's entities, decaying the
    # contribution by half per hop so distance is reflected in the score.
    reached = set(frontier) | seed_set
    decay = 0.5
    for _ in range(2, hops + 1):
        legs, coms = expand_seeds(store, list(frontier))
        frontier = connect(legs, coms, decay=decay, exclude=reached)
        reached |= frontier
        decay *= 0.5
        if not frontier:
            break

    for cand in candidates.values():
        subjects = {s.lower() for s in cand.bill.get("subjects", [])}
        overlap = keywords & subjects
        cand.subject_overlap = sorted(overlap)
        cand.score += subject_weight * len(overlap)

    return sorted(candidates.values(), key=lambda c: c.score, reverse=True)[:top_k]


def graph_retrieve(store: GraphStore, query: str, top_k: int) -> GraphRetrieval:
    """
    Run the full GraphRAG retrieval and return every intermediate artifact.

    Composed from the same stage functions the engine calls one at a time, so
    this composite and the live, per-stage engine trace can never disagree about
    what GraphRAG does. Used directly by tests and the offline eval harness.
    """
    keywords = _keywords(query)
    seeds = _seed_bills(store, keywords, limit=max(3, top_k // 2))
    seed_legislators, seed_committees = expand_seeds(store, seeds)
    precedents = score_candidates(store, seeds, seed_legislators, seed_committees, keywords, top_k)
    subgraph_nodes, subgraph_edges = _build_subgraph(store, seeds, precedents)
    return GraphRetrieval(
        query=query,
        keywords=sorted(keywords),
        seed_bill_ids=seeds,
        precedents=precedents,
        subgraph_nodes=subgraph_nodes,
        subgraph_edges=subgraph_edges,
    )


def _build_subgraph(
    store: GraphStore, seeds: list[str], precedents: list[ScoredBill]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Assemble the visualization payload: nodes and edges connecting the seeds,
    the entities they share, and the chosen precedent bills.

    Only the entities that actually connect a seed to a precedent are included,
    so the picture the developer sees is exactly the evidence the ranking used
    -- no orphan nodes, no edges that didn't matter to the result.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, kind: str, label: str, **extra: Any) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "kind": kind, "label": label, **extra}

    for bid in seeds:
        bill = store.get_bill(bid) or {}
        add_node(bid, "seed_bill", bid, title=bill.get("title"), outcome=bill.get("outcome"))

    for cand in precedents:
        add_node(
            cand.bill_id,
            "precedent_bill",
            cand.bill_id,
            title=cand.bill.get("title"),
            outcome=cand.bill.get("outcome"),
            score=round(cand.score, 2),
        )
        for bioguide in cand.shared_legislators:
            person = store.legislator(bioguide) or {}
            node_id = f"leg:{bioguide}"
            add_node(
                node_id,
                "legislator",
                person.get("full_name") or bioguide,
                party=person.get("party"),
                state=person.get("state"),
            )
            # Edge from the entity to both the seed(s) and this precedent it links.
            edges.append({"source": node_id, "target": cand.bill_id, "rel": "sponsored"})
            for seed in seeds:
                if bioguide in store.legislators_of(seed):
                    edges.append({"source": node_id, "target": seed, "rel": "sponsored"})
        for committee_name in cand.shared_committees:
            node_id = f"com:{committee_name}"
            add_node(node_id, "committee", committee_name)
            edges.append({"source": node_id, "target": cand.bill_id, "rel": "reviewed"})
            for seed in seeds:
                if committee_name in store.committees_of(seed):
                    edges.append({"source": node_id, "target": seed, "rel": "reviewed"})

    # De-duplicate edges (an entity can be reached by more than one path).
    unique_edges = {(e["source"], e["target"], e["rel"]): e for e in edges}
    return list(nodes.values()), list(unique_edges.values())


# --- Full knowledge-graph projection (for the standalone graph explorer) ------

# A hub (a legislator or committee) that touches more than this many bills is
# too generic to imply a meaningful bill-to-bill relationship -- connecting all
# pairs under it would add noise and O(k^2) edges. We skip those hubs so the
# explorer stays legible and bounded.
_MAX_HUB_FANOUT = 40


def collect_metadata(store: GraphStore) -> dict[str, Any]:
    """Distinct subjects and congresses in the graph -- for the explorer filters."""
    subjects: set[str] = set()
    congresses: set[str] = set()
    for bid in store.all_bill_ids():
        bill = store.get_bill(bid) or {}
        subjects.update(bill.get("subjects", []))
        if bill.get("congress"):
            congresses.add(str(bill["congress"]))
    return {"subjects": sorted(subjects), "congresses": sorted(congresses)}


def build_bill_graph(
    store: GraphStore,
    subject: str | None = None,
    congress: str | None = None,
    limit: int = 120,
    min_shared: int = 1,
) -> dict[str, Any]:
    """
    Build a bill-to-bill relationship graph: bills are nodes, and two bills are
    connected when they share sponsors and/or committees.

    This is the "knowledge-graph space" the explorer renders -- it answers "which
    bills are related, and why" directly, independent of any single query. Edges
    are weighted by how many entities two bills share, so the thickest links are
    the strongest relationships. Filters (subject, congress) scope the view, and
    ``limit`` keeps the most-connected bills so a large corpus stays navigable.
    """
    # 1. Select the bills in scope.
    selected: list[str] = []
    for bid in store.all_bill_ids():
        bill = store.get_bill(bid) or {}
        if congress and str(bill.get("congress")) != str(congress):
            continue
        if subject and subject.lower() not in {s.lower() for s in bill.get("subjects", [])}:
            continue
        selected.append(bid)
    scope = set(selected)

    # 2. Aggregate shared-entity counts for every co-occurring pair of bills,
    #    walking each hub (legislator/committee) once.
    from collections import defaultdict

    pair_sponsors: dict[tuple[str, str], int] = defaultdict(int)
    pair_committees: dict[tuple[str, str], int] = defaultdict(int)

    legs: set[str] = set()
    coms: set[str] = set()
    for bid in selected:
        legs |= store.legislators_of(bid)
        coms |= store.committees_of(bid)

    def connect(hub_bills: list[str], bucket: dict[tuple[str, str], int]) -> None:
        bs = sorted(b for b in hub_bills if b in scope)
        if len(bs) < 2 or len(bs) > _MAX_HUB_FANOUT:
            return
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                bucket[(bs[i], bs[j])] += 1

    for leg in legs:
        connect(store.bills_of_legislator(leg), pair_sponsors)
    for com in coms:
        connect(store.bills_of_committee(com), pair_committees)

    # 3. Merge into weighted edges, keeping only pairs sharing >= min_shared.
    edges: list[dict[str, Any]] = []
    degree: dict[str, int] = defaultdict(int)
    all_pairs = set(pair_sponsors) | set(pair_committees)
    for a, b in all_pairs:
        sp = pair_sponsors.get((a, b), 0)
        cm = pair_committees.get((a, b), 0)
        shared = sp + cm
        if shared < min_shared:
            continue
        rel = "both" if sp and cm else ("sponsor" if sp else "committee")
        edges.append({"source": a, "target": b, "weight": shared, "rel": rel})
        degree[a] += shared
        degree[b] += shared

    # 4. Cap to the most-connected bills so the picture stays readable.
    ranked = sorted(scope, key=lambda b: degree.get(b, 0), reverse=True)[:limit]
    keep = set(ranked)
    edges = [e for e in edges if e["source"] in keep and e["target"] in keep]

    from precedent.models import bill_label, bill_type_name

    nodes = []
    for bid in ranked:
        bill = store.get_bill(bid) or {}
        nodes.append(
            {
                "id": bid,
                "label": bill_label(bill),  # e.g. "H.Res. 1102 (118th)"
                "type_name": bill_type_name(bill.get("bill_type")),  # e.g. "House Resolution"
                "title": bill.get("title"),
                "outcome": bill.get("outcome"),
                "congress": bill.get("congress"),
                "subjects": bill.get("subjects", []),
                "degree": degree.get(bid, 0),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "total_bills": len(scope),
        "shown": len(nodes),
    }
