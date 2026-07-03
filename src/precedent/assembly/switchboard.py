"""
The switchboard: the one place that wires the whole application together.

Everything with a lifetime longer than a single request -- the settings, the
two stores, the two engines, the query logger -- is constructed here, exactly
once, and handed out on request. Nothing else in the app constructs a store or
an engine directly; they ask the switchboard. That keeps the "how is this
deployed" decisions (which graph backend, which Chroma mode, is there an API
key) isolated to config + this assembly step, and makes the app trivial to
stand up in a test with an alternate configuration.

On construction it also loads the seed dataset into both stores if they're
empty, so a freshly started process can answer a query immediately without a
separate ingestion step.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from precedent.assembly.app_config import Settings, get_settings
from precedent.assembly.model_config import ModelConfig
from precedent.engine.graph_engine import GraphEngine
from precedent.engine.vector_engine import VectorEngine
from precedent.preprocessing.full_refresh import index_bills
from precedent.response.logger import QueryLogger
from precedent.stores.graph.loader import GraphStore, build_graph_store
from precedent.stores.vector.loader import VectorStore

logger = logging.getLogger(__name__)


class Switchboard:
    """Owns the long-lived objects and constructs them from Settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_config = ModelConfig()

        self.graph_store: GraphStore = build_graph_store(self.settings)
        self.vector_store = VectorStore(self.settings)
        self.query_logger = QueryLogger()

        # The enriched corpus, kept so the vector engine can re-chunk/re-embed
        # on the fly when a user turns the chunking or embedding knobs.
        self.bills: list[dict[str, Any]] = []
        self.corpus_source = "seed"  # "seed" or "ingested"; set during load
        self._load_seed_if_empty()

        self.graph_engine = GraphEngine(self.graph_store, self.settings, self.model_config)
        self.vector_engine = VectorEngine(
            self.vector_store, self.settings, self.model_config, corpus=self.bills
        )

    def _load_seed_if_empty(self) -> None:
        """
        Load the checked-in seed bills into both stores unless data is already
        present. This is what makes ``uvicorn ... main:app`` answer a query on a
        cold start with no manual ingestion -- the demo is always ready.

        The seed is always read and enriched into ``self.bills`` (the corpus the
        vector engine re-chunks for tunable queries), but only *indexed* into the
        stores when they're empty, so a persistent Chroma/Neo4j isn't re-loaded
        on every restart.
        """
        from precedent.preprocessing.entity_extraction import enrich_bills

        bills = self._read_corpus()
        if not bills:
            logger.warning(
                "No corpus found at %s or %s", self.settings.bills_path, self.settings.seed_path
            )
            return
        self.bills = enrich_bills(bills)
        self.corpus_source = "ingested" if self.settings.bills_path.exists() else "seed"

        # In the local/dev config (embedded Chroma + in-memory graph) rebuild
        # from scratch every boot so the index always reflects the current
        # chunking/embedding defaults. In the server/Neo4j config, respect
        # existing data and only load when empty, so restarts don't wipe a real
        # database.
        dev_mode = (
            self.settings.chroma_mode == "embedded" and self.settings.graph_backend == "memory"
        )
        already_loaded = self.graph_store.count_bills() > 0 and self.vector_store.count() > 0
        if already_loaded and not dev_mode:
            logger.info(
                "Stores already populated (%d bills), skipping seed index",
                self.graph_store.count_bills(),
            )
            return
        if dev_mode:
            self.vector_store.reset()
        index_bills(self.bills, self.graph_store, self.vector_store)

    def _read_corpus(self) -> list[dict[str, Any]]:
        """
        Load the corpus, preferring the ingested bills over the demo seed.

        If ``scripts/ingest.py`` has written data/bills.json, that real corpus is
        used; otherwise the checked-in 16-bill seed keeps the app runnable out of
        the box. One ingestion run flips the whole app to real data with no code
        or config change.
        """
        for path in (self.settings.bills_path, self.settings.seed_path):
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        return []

    def health(self) -> dict[str, Any]:
        """A readiness snapshot for the /health endpoint."""
        return {
            "status": "ok",
            "graph_backend": self.settings.graph_backend,
            "chroma_mode": self.settings.chroma_mode,
            "bills_in_graph": self.graph_store.count_bills(),
            "chunks_in_vector_store": self.vector_store.count(),
            "corpus_source": self.corpus_source,
            "llm_enabled": self.settings.llm_enabled,
            "model": self.model_config.model if self.settings.llm_enabled else None,
        }


@lru_cache
def get_switchboard() -> Switchboard:
    """Return the process-wide Switchboard singleton (built on first call)."""
    return Switchboard()
