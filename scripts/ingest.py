"""
Ingest real bills from GovInfo into data/bills.json.

This replaces the 16-bill demo seed with as much real legislative data as you
want. It discovers BILLSTATUS packages for a Congress, fetches and parses each
one, enriches it with derived subjects, and writes the lot to data/bills.json --
which the app loads in preference to the seed on next start (so GraphRAG builds
its relationships over the full, real corpus).

Because every bill is a separate API call and GovInfo rate-limits default keys
(~1000 requests/hour), this is bounded by ``--limit``. Run it a few times with a
rising limit, or overnight with a large one, to grow the corpus.

Usage
-----
    export GOVINFO_API_KEY=...            # or put it in .env
    python scripts/ingest.py --congress 118 --limit 300
    python scripts/ingest.py --congress 119 --since 2025-01-01 --limit 500 --append

The package id itself encodes the Congress (e.g. "BILLSTATUS-118hr1"), so we
filter by Congress *before* fetching -- no wasted requests on other Congresses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from precedent.preprocessing.entity_extraction import enrich_bills
from precedent.preprocessing.ingestion.utils import IngestionUtils
from precedent.preprocessing.parsers.billstatus import parse_billstatus

logger = logging.getLogger(__name__)

COLLECTIONS_URL = "https://api.govinfo.gov/collections/BILLSTATUS/{start}"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "bills.json"


def discover_ids_for_congress(
    api_key: str, congress: int, since: str, limit: int, max_pages: int = 200
) -> list[str]:
    """
    Page through the collections endpoint, keeping only this Congress's package
    ids, until we have ``limit`` of them (or run out of pages).

    Filtering on the package-id prefix means we never fetch a bill from another
    Congress just to discover it doesn't belong -- the id carries everything the
    discovery step needs.
    """
    prefix = f"BILLSTATUS-{congress}"
    url = COLLECTIONS_URL.format(start=since)
    params = {"api_key": api_key, "pageSize": "1000", "offsetMark": "*"}
    ids: list[str] = []
    bar = tqdm(desc=f"Discovering Congress {congress}", unit="pkg")

    for _ in range(max_pages):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("Discovery request failed: %s", exc)
            break
        data = resp.json()
        for pkg in data.get("packages", []):
            pid = pkg.get("packageId", "")
            if pid.startswith(prefix):
                ids.append(pid)
                bar.update(1)
                if len(ids) >= limit:
                    bar.close()
                    return ids
        url = data.get("nextPage")
        params = {"api_key": api_key}
        if not url:
            break
    bar.close()
    return ids


def ingest(congress: int, since: str, limit: int, sleep: float, max_summary: int) -> list[dict]:
    """Discover, fetch, and parse up to ``limit`` bills for one Congress."""
    load_dotenv()
    api_key = os.getenv("GOVINFO_API_KEY")
    if not api_key:
        raise SystemExit("GOVINFO_API_KEY not set (put it in .env or export it).")

    ingestor = IngestionUtils(api_key)
    package_ids = discover_ids_for_congress(api_key, congress, since, limit)
    logger.info("Discovered %d package ids for Congress %d", len(package_ids), congress)

    bills: list[dict] = []
    for pid in tqdm(package_ids, desc="Fetching + parsing", unit="bill", colour="green"):
        xml = ingestor.fetch(pid)
        if xml is None:
            continue
        parsed = parse_billstatus(xml)
        if parsed and str(parsed.get("congress")) == str(congress):
            # Some bill summaries run to 100k+ chars (e.g. the NDAA). Keep the
            # lead so the vector index stays a manageable size and boots fast --
            # the opening paragraphs carry the substance a topical search needs.
            if max_summary and parsed.get("summary"):
                parsed["summary"] = parsed["summary"][:max_summary]
            bills.append(parsed)
        if sleep:
            time.sleep(sleep)
    return bills


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Ingest real bills from GovInfo.")
    ap.add_argument("--congress", type=int, default=118, help="Congress number (e.g. 118)")
    ap.add_argument(
        "--since",
        default=None,
        help="ISO date lower bound, e.g. 2023-01-01 (defaults to the Congress start)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="max bills to fetch (bounded only by your GovInfo quota + time)",
    )
    ap.add_argument(
        "--max-summary-chars",
        type=int,
        default=2000,
        help="truncate each summary to this many chars (0 = keep full)",
    )
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between fetches")
    ap.add_argument(
        "--append",
        action="store_true",
        help="merge into existing data/bills.json instead of replacing it",
    )
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    # Default the date bound to the Congress's own start year so discovery starts
    # in the right window (each Congress begins Jan 3 of an odd year).
    since = args.since or f"{1789 + (args.congress - 1) * 2}-01-01"
    since_iso = f"{since}T00:00:00Z" if "T" not in since else since

    bills = ingest(args.congress, since_iso, args.limit, args.sleep, args.max_summary_chars)
    enrich_bills(bills)

    if args.append and args.out.exists():
        existing = json.loads(args.out.read_text())
        seen = {(b["congress"], b["bill_type"], b["bill_number"]) for b in bills}
        merged = bills + [
            b for b in existing if (b["congress"], b["bill_type"], b["bill_number"]) not in seen
        ]
        bills = merged

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bills, indent=2))
    with_summary = sum(1 for b in bills if b.get("summary"))
    print(f"\nWrote {len(bills)} bills to {args.out} ({with_summary} with summaries).")
    print("Restart the API to load them:  uvicorn precedent.api.main:app --port 8080")


if __name__ == "__main__":
    main()
