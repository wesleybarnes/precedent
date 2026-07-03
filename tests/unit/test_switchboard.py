"""
Tests for the switchboard: that it wires everything together, loads the seed
corpus, and reports a healthy status over the local (no-server) backends.
"""


def test_switchboard_loads_seed(switchboard):
    health = switchboard.health()
    assert health["status"] == "ok"
    assert health["graph_backend"] == "memory"
    assert health["chroma_mode"] == "embedded"
    assert health["bills_in_graph"] > 0
    assert health["chunks_in_vector_store"] > 0


def test_switchboard_llm_disabled_without_key(switchboard):
    # The fixture sets no API key, so the app must run in extractive mode.
    assert switchboard.health()["llm_enabled"] is False


def test_switchboard_exposes_corpus(switchboard):
    assert switchboard.bills, "corpus should be populated for the tunable path"
    # Enrichment ran: every bill carries a subjects list.
    assert all("subjects" in b for b in switchboard.bills)
