"""
Tests for the standalone knowledge-graph projection (the explorer's data) and
the newer tunable parameters (chunking strategy, similarity threshold, hops).
"""

from precedent.params import GraphParams, VectorParams
from precedent.stores.graph.queries import build_bill_graph, collect_metadata


def test_bill_graph_has_weighted_relationships(switchboard):
    g = build_bill_graph(switchboard.graph_store)
    assert g["nodes"], "explorer should have bill nodes"
    assert g["edges"], "bills in the seed share sponsors/committees, so there are edges"
    # Every edge is a real bill-to-bill relationship with a positive weight.
    node_ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["source"] in node_ids and e["target"] in node_ids
        assert e["weight"] >= 1
        assert e["rel"] in ("sponsor", "committee", "both")


def test_bill_graph_min_shared_filters_edges(switchboard):
    loose = build_bill_graph(switchboard.graph_store, min_shared=1)
    strict = build_bill_graph(switchboard.graph_store, min_shared=3)
    assert len(strict["edges"]) <= len(loose["edges"])


def test_bill_graph_subject_filter(switchboard):
    meta = collect_metadata(switchboard.graph_store)
    assert meta["subjects"]
    subject = "healthcare" if "healthcare" in meta["subjects"] else meta["subjects"][0]
    g = build_bill_graph(switchboard.graph_store, subject=subject)
    # Every shown bill actually carries the filtered subject.
    for n in g["nodes"]:
        assert subject.lower() in {s.lower() for s in n["subjects"]}


def test_chunk_strategy_changes_chunking(switchboard):
    q = "clean energy tax credit"
    whole = switchboard.vector_engine.execute(q, VectorParams(chunk_strategy="whole"))
    fixed = switchboard.vector_engine.execute(
        q, VectorParams(chunk_strategy="fixed", chunk_size=100)
    )
    whole_payload = next(s for s in whole.steps if s["step"] == "embed_search")["payload"]
    fixed_payload = next(s for s in fixed.steps if s["step"] == "embed_search")["payload"]
    assert whole_payload["chunk_strategy"] == "whole"
    assert fixed_payload["chunk_strategy"] == "fixed"
    assert fixed_payload["index_mode"].startswith("ephemeral")


def test_similarity_threshold_drops_weak_matches(switchboard):
    q = "clean energy tax credit"
    hi = switchboard.vector_engine.execute(q, VectorParams(similarity_threshold=0.9))
    payload = next(s for s in hi.steps if s["step"] == "embed_search")["payload"]
    # A very high threshold should drop essentially everything on this small corpus.
    assert payload["chunk_count"] == 0


def test_graph_hops_expand_reach(switchboard):
    # Two hops should reach at least as many precedents as one (never fewer),
    # since the second hop only adds bills.
    q = "immigration visa reform"
    one = switchboard.graph_engine.execute(q, GraphParams(hops=1, top_k=12))
    two = switchboard.graph_engine.execute(q, GraphParams(hops=2, top_k=12))
    assert len(two.retrieval["precedents"]) >= len(one.retrieval["precedents"])
