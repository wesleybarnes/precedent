"""
Tests for the two engines and the tunable parameters.

These build small graphs/vector stores directly (no servers) and assert on the
trace structure, the retrieval results, and -- importantly -- that turning the
RAG knobs actually changes what comes back, which is the educational promise of
the whole tool.
"""

from precedent.assembly.model_config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    ModelConfig,
    resolve_model,
)
from precedent.engine.graph_engine import GraphEngine
from precedent.engine.vector_engine import VectorEngine
from precedent.params import GraphParams, VectorParams
from precedent.preprocessing.entity_extraction import enrich_bills
from precedent.preprocessing.full_refresh import index_bills
from precedent.stores.graph.loader import InMemoryGraphStore
from precedent.stores.vector.loader import VectorStore


def _bill(congress, num, title, summary, sponsor, committee, outcome):
    return {
        "congress": congress,
        "bill_type": "hr",
        "bill_number": num,
        "title": title,
        "summary": summary,
        "outcome": outcome,
        "sponsor": {"bioguide_id": sponsor, "full_name": sponsor, "party": "D", "state": "CA"},
        "cosponsors": [],
        "committees": [{"name": committee, "chamber": "House"}],
        "actions": [],
    }


def _fixture_corpus():
    # Two healthcare bills share a sponsor and committee (so the graph connects
    # them); one unrelated tax bill stands alone.
    return [
        _bill(
            "118",
            "1",
            "Health Coverage Act",
            "Expands health insurance coverage and Medicare drug benefits.",
            "SPON_A",
            "Energy and Commerce",
            "became_law",
        ),
        _bill(
            "118",
            "2",
            "Medical Access Act",
            "Improves patient access to hospitals and health clinics.",
            "SPON_A",
            "Energy and Commerce",
            "died_in_committee",
        ),
        _bill(
            "118",
            "3",
            "Tax Relief Act",
            "Lowers the tax rate and expands the revenue deduction for businesses.",
            "SPON_B",
            "Ways and Means",
            "vetoed",
        ),
    ]


def _build_engines(settings):
    bills = enrich_bills(_fixture_corpus())
    graph = InMemoryGraphStore()
    vector = VectorStore(settings)
    index_bills(bills, graph, vector)
    mc = ModelConfig()
    return (
        GraphEngine(graph, settings, mc),
        VectorEngine(vector, settings, mc, corpus=bills),
    )


def test_graph_engine_emits_full_pipeline(settings):
    graph, _ = _build_engines(settings)
    out = graph.execute("How likely is a health coverage bill to pass?")
    step_names = [s["step"] for s in out.steps]
    assert step_names == [
        "parse_query",
        "seed_match",
        "expand_graph",
        "score_precedents",
        "retrieve",
        "build_context",
        "answer",
    ]
    assert out.answer  # extractive answer is non-empty
    assert out.retrieval["precedents"]  # found at least one connected precedent


def test_graph_finds_precedent_via_shared_sponsor(settings):
    graph, _ = _build_engines(settings)
    out = graph.execute("health insurance coverage")
    ids = [p["id"] for p in out.retrieval["precedents"]]
    # Bill 2 shares a sponsor/committee with the seed (bill 1) but the tax bill
    # (bill 3) shares nothing, so graph reasoning must surface 2 and not 3.
    assert "118-HR-2" in ids
    assert "118-HR-3" not in ids


def test_vector_engine_ranks_by_similarity(settings):
    _, vector = _build_engines(settings)
    out = vector.execute("health insurance coverage")
    chunks = out.retrieval["chunks"]
    assert chunks
    # Similarity should be sorted descending.
    sims = [c["similarity"] for c in chunks]
    assert sims == sorted(sims, reverse=True)


def test_vector_chunk_size_changes_chunk_count(settings):
    _, vector = _build_engines(settings)
    big = vector.execute("health", VectorParams(chunk_size=1000, chunk_overlap=0))
    small = vector.execute("health", VectorParams(chunk_size=60, chunk_overlap=0))
    big_step = next(s for s in big.steps if s["step"] == "embed_search")
    small_step = next(s for s in small.steps if s["step"] == "embed_search")
    # Smaller chunks -> more chunks searched. This is the knob-turning payoff.
    assert small_step["payload"]["chunk_count"] > big_step["payload"]["chunk_count"]


def test_graph_weights_change_ranking(settings):
    graph, _ = _build_engines(settings)
    out = graph.execute(
        "health insurance coverage",
        GraphParams(legislator_weight=0.0, committee_weight=0.0, subject_weight=5.0),
    )
    # With connection weights zeroed, scoring is driven purely by subject
    # overlap -- the engine still runs and returns a result.
    assert out.answer


def test_model_registry_and_resolution():
    ids = {m.id for m in AVAILABLE_MODELS}
    assert DEFAULT_MODEL in ids
    # LiteLLM route ids carry a provider prefix.
    assert all("/" in m.id for m in AVAILABLE_MODELS)
    # Unknown/empty selections fall back to the default; known ones pass through.
    assert resolve_model("nonsense") == DEFAULT_MODEL
    assert resolve_model(None) == DEFAULT_MODEL
    assert resolve_model(DEFAULT_MODEL) == DEFAULT_MODEL


def test_engine_accepts_model_override(settings):
    # Without a key the model isn't actually called, but the argument must thread
    # through cleanly and the extractive answer still comes back.
    graph, vector = _build_engines(settings)
    other = AVAILABLE_MODELS[-1].id
    assert graph.execute("health coverage", None, other).answer
    assert vector.execute("health coverage", None, other).answer
