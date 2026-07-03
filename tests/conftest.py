"""
Shared test fixtures.

Every test runs against the local, no-server configuration: the in-memory graph
backend and an embedded Chroma index pointed at a throwaway temp directory, so
tests never touch a real database and never collide with the app's own on-disk
Chroma. The seed dataset is the corpus, so tests exercise the same data the demo
shows.
"""

import pytest

from precedent.assembly.app_config import Settings
from precedent.assembly.switchboard import Switchboard


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Local-only settings with an isolated, per-test Chroma directory."""
    return Settings(
        graph_backend="memory",
        chroma_mode="embedded",
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection="test_chunks",
        # Point at a nonexistent ingested-corpus path so tests always run on the
        # small, deterministic seed -- never on whatever data/bills.json happens
        # to hold after an ingestion run.
        bills_path=tmp_path / "no_ingested_bills.json",
        anthropic_api_key=None,  # force the extractive path -- deterministic tests
    )


@pytest.fixture
def switchboard(settings) -> Switchboard:
    """A fully wired switchboard over the seed corpus, isolated per test."""
    return Switchboard(settings=settings)
