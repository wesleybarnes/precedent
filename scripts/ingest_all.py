#!/usr/bin/env python3
"""Ingest all available historical Congresses into the corpus.

GovInfo's BILLSTATUS collection only covers the 113th Congress (2013) onward --
there is no bill-status data before that -- so "all history" here means 113-118.

It:
- Fetches bills for each Congress in EARLIEST_CONGRESS..LATEST_CONGRESS
- Deduplicates automatically with --append
- Respects rate limits (default small delay between requests)
- Can be resumed if interrupted (just re-run)

Usage:
    export GOVINFO_API_KEY=...
    python scripts/ingest_all.py
"""

import subprocess
import sys

# BILLSTATUS coverage begins with the 113th Congress; 118 is the most recent
# complete one. Bump LATEST_CONGRESS to 119 once that Congress has data you want.
EARLIEST_CONGRESS = 113
LATEST_CONGRESS = 118

# Recent Congresses hold 10k-15k+ bills each. This caps how many we pull per
# Congress so a full run stays within a sane time/quota budget. Raise it if you
# want deeper coverage and have the API quota to spend.
LIMIT_PER_CONGRESS = 500  # bills per congress
SLEEP_BETWEEN_REQUESTS = 0.1  # seconds


def ingest_congress(congress: int, append: bool = False) -> bool:
    """Ingest one Congress. Returns True if successful."""
    cmd = [
        sys.executable,
        "scripts/ingest.py",
        f"--congress={congress}",
        f"--limit={LIMIT_PER_CONGRESS}",
        f"--sleep={SLEEP_BETWEEN_REQUESTS}",
    ]
    if append:
        cmd.append("--append")

    print(f"\n{'='*60}")
    print(f"Congress {congress} ({1788 + 2*congress}-{1790 + 2*congress})")
    print("=" * 60)
    return subprocess.run(cmd).returncode == 0


def main() -> None:
    # Append onto the existing corpus for every Congress after the first, so a
    # re-run (or a run on top of your current Congress-118 data) dedupes cleanly.
    for c in range(EARLIEST_CONGRESS, LATEST_CONGRESS + 1):
        if not ingest_congress(c, append=True):
            print(f"\n⚠️  Congress {c} failed. Resume later with:")
            print("   python scripts/ingest_all.py")
            return

    total = LATEST_CONGRESS - EARLIEST_CONGRESS + 1
    print("\n" + "=" * 60)
    print(f"✓ All {total} available Congresses ({EARLIEST_CONGRESS}-{LATEST_CONGRESS}) ingested!")
    print("=" * 60)
    print("\nRestart the API to load the data:")
    print("  uvicorn precedent.api.main:app --port 8080")
    print("(or just ./dev.sh)")


if __name__ == "__main__":
    main()
