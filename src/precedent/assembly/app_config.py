"""
Application configuration, loaded once from the environment.

Every tunable that changes between "running on my laptop" and "running in
docker-compose / production" lives here as a single ``Settings`` object, read
from environment variables (and a local ``.env`` file) via pydantic-settings.
Nothing else in the codebase should call ``os.getenv`` directly -- they ask
the switchboard for the already-constructed ``Settings`` instead, so there is
exactly one place that knows how the app is wired.

The most important switches here are the two that let the whole system run
with no external servers at all:

    GRAPH_BACKEND = "memory"   -> in-process NetworkX graph, no Neo4j needed
    GRAPH_BACKEND = "neo4j"    -> connect to a real Neo4j (docker-compose)

    CHROMA_MODE   = "embedded" -> in-process Chroma PersistentClient, no server
    CHROMA_MODE   = "server"   -> connect to a Chroma HTTP server (CHROMA_URL)

Defaults are chosen so ``uvicorn precedent.api.main:app`` works on a fresh
checkout with nothing else installed or running. Point the backends at real
servers only when you actually have them.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: this file is src/precedent/assembly/app_config.py, so four
# .parent hops land on the project root. Used to anchor the default on-disk
# locations (seed data, the embedded Chroma directory) so they resolve the
# same way no matter what working directory the process was launched from.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime configuration for Precedent, read from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data ingestion (only needed when refreshing from the real API) ---
    govinfo_api_key: str | None = None

    # --- LLM (Claude via the Anthropic SDK) ---
    # Left as None on a fresh checkout: the engines detect the missing key and
    # fall back to an extractive answer instead of a generated one, so the app
    # still runs end-to-end. Set ANTHROPIC_API_KEY to enable generation.
    anthropic_api_key: str | None = None

    # --- Graph store ---
    graph_backend: Literal["memory", "neo4j"] = "memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "devpassword"

    # --- Vector store ---
    chroma_mode: Literal["embedded", "server"] = "embedded"
    chroma_url: str | None = None  # e.g. "http://localhost:8000" when mode="server"
    chroma_persist_dir: Path = _REPO_ROOT / "data" / "chroma"
    chroma_collection: str = "bill_chunks"

    # --- Corpus ---
    # A small, checked-in set of parsed bills so the app has something to serve
    # without anyone having to run a GovInfo ingestion first.
    seed_path: Path = _REPO_ROOT / "data" / "seed_bills.json"
    # Where `scripts/ingest.py` writes the real, ingested corpus. When this file
    # exists it is loaded in preference to the seed, so a single ingestion run
    # upgrades the whole app from 16 demo bills to hundreds of real ones.
    bills_path: Path = _REPO_ROOT / "data" / "bills.json"

    # --- Retrieval knobs ---
    top_k: int = 6  # how many precedent bills / chunks each engine returns

    # --- API server ---
    # Origins allowed to call the API from a browser. The Vite dev server runs
    # on 5173; the containerised frontend is served from 8081.
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:8081",
        "http://localhost:3000",
    ]

    @property
    def llm_enabled(self) -> bool:
        """True when an Anthropic key is present, so generation can run."""
        return bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    """
    Return the process-wide Settings singleton.

    Cached so every caller shares one instance -- reading the environment and
    parsing ``.env`` happens exactly once, and tests can override the cache by
    calling ``get_settings.cache_clear()`` after patching the environment.
    """
    return Settings()
