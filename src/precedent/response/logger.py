"""
Append-only query logging.

Every comparison the app runs is written as one JSON line to a log file. It's a
lightweight audit trail -- useful for seeing what people asked, eyeballing how
the two engines answered the same question over time, and building an eval set
from real queries later. Deliberately JSONL (one self-contained record per line)
so it can be tailed live and parsed incrementally without loading the whole
file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default log location: a data/ dir at the repo root (this file is
# src/precedent/response/logger.py, so three .parent hops reach the repo root).
_DEFAULT_LOG = Path(__file__).resolve().parents[3] / "data" / "query_log.jsonl"


class QueryLogger:
    """Writes one JSON record per query to a JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _DEFAULT_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        query: str,
        graph_answer: str,
        vector_answer: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one comparison record. Never raises -- logging must not break a query."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "graph_answer": graph_answer,
            "vector_answer": vector_answer,
            **(extra or {}),
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:  # pragma: no cover - disk issues shouldn't fail a request
            logger.warning("Failed to write query log: %s", exc)
