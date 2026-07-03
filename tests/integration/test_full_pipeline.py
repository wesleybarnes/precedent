"""
End-to-end integration: a question runs through the full switchboard -- both
engines, over the seed corpus, on the local backends -- and produces the
side-by-side comparison the API returns. This is the closest test to what a
user actually experiences.
"""

from precedent.response.formatter import format_comparison


def test_full_comparison_over_seed(switchboard):
    query = "How likely is a prescription drug pricing bill to become law?"
    graph_out = switchboard.graph_engine.execute(query)
    vector_out = switchboard.vector_engine.execute(query)
    result = format_comparison(query, graph_out, vector_out)

    assert result["query"] == query
    for side in ("graph", "vector"):
        assert result[side]["answer"]
        assert result[side]["steps"]
        # Every step reports the function behind it, so the UI can peer into code.
        assert all(s["source_symbol"] for s in result[side]["steps"])

    # The graph side exposes a subgraph; the vector side exposes ranked chunks.
    assert "subgraph" in result["graph"]["retrieval"]
    assert "chunks" in result["vector"]["retrieval"]


def test_engines_can_diverge(switchboard):
    """
    The whole point of Precedent: on a query where structure and text disagree,
    the two engines can surface different precedents. We don't assert they always
    differ (that depends on data), only that both return grounded results so the
    comparison is real.
    """
    query = "immigration visa reform"
    graph_ids = {p["id"] for p in switchboard.graph_engine.execute(query).retrieval["precedents"]}
    vector_ids = {
        c["bill_id"] for c in switchboard.vector_engine.execute(query).retrieval["chunks"]
    }
    assert graph_ids, "graph engine should find connected precedents"
    assert vector_ids, "vector engine should find similar passages"
