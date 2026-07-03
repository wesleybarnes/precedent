"""
Chroma client construction -- embedded or server, chosen by configuration.

The vector store can run two ways, and this module is the single place that
knows the difference:

* ``embedded`` (default): ``chromadb.PersistentClient`` keeps the index in a
  local directory, in-process, with no server to start. This is what lets the
  app run on a laptop with nothing but ``pip install``.
* ``server``: ``chromadb.HttpClient`` talks to the Chroma container in
  docker-compose (``CHROMA_URL``).

Either way the caller gets back a client and a collection configured for cosine
similarity; nothing downstream cares which mode produced it.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import chromadb
from chromadb.config import Settings as ChromaSettings

from precedent.assembly.app_config import Settings

logger = logging.getLogger(__name__)


def build_chroma_client(settings: Settings):
    """Return a Chroma client for the configured mode (embedded or server)."""
    if settings.chroma_mode == "server":
        url = settings.chroma_url or "http://localhost:8000"
        parsed = urlparse(url)
        logger.info("Using Chroma server at %s", url)
        return chromadb.HttpClient(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8000,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using embedded Chroma at %s", settings.chroma_persist_dir)
    return chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(client, name: str):
    """
    Get-or-create the bill-chunk collection, configured for cosine distance.

    We supply embeddings ourselves (see embedding.py) rather than letting Chroma
    compute them, so the collection is created with no embedding function -- the
    ``hnsw:space=cosine`` metadata is the only index configuration it needs.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
