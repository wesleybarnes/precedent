"""
The one function that loads bills into both stores.

Ingestion, whichever source it comes from, ends the same way: a list of parsed
bill dicts that need to be (a) enriched with subjects and (b) written into both
the graph store and the vector store so the two engines can retrieve over the
same underlying facts. ``index_bills`` is that shared final step, used by both
the live GovInfo refresh and the seed-data loader, so there is exactly one code
path that decides how a bill becomes searchable.

Keeping this separate from the GovInfo-specific ingestion in
``ingestion/govinfo.py`` means the seed loader and a future data source (a CSV,
a different API) can reuse it without dragging in the GovInfo client.
"""

from __future__ import annotations

import logging
from typing import Any

from precedent.preprocessing.entity_extraction import enrich_bills
from precedent.stores.graph.loader import GraphStore
from precedent.stores.vector.loader import VectorStore

logger = logging.getLogger(__name__)


def index_bills(
    bills: list[dict[str, Any]],
    graph_store: GraphStore,
    vector_store: VectorStore,
) -> int:
    """
    Enrich and load a batch of bills into both stores. Returns the count loaded.

    Order matters only in that enrichment happens first: the graph's seed
    matching and the chunk text both read the ``subjects``/``summary`` fields,
    so those must be populated before either store sees the bill. After that the
    two stores are independent -- the graph indexes entities and relationships,
    the vector store indexes embedded chunks -- but they index the *same* bills,
    which is what makes the side-by-side comparison fair.
    """
    if not bills:
        logger.warning("index_bills called with no bills")
        return 0

    enrich_bills(bills)
    graph_store.add_bills(bills)
    vector_store.add_bills(bills)
    logger.info("Indexed %d bills into graph + vector stores", len(bills))
    return len(bills)
