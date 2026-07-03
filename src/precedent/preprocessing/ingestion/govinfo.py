import logging
import os
from tqdm import tqdm

import requests
from dotenv import load_dotenv

# Adjust this import path to match your actual installed package name.
from precedent.preprocessing.ingestion.utils import IngestionUtils
from precedent.preprocessing.parsers.billstatus import parse_billstatus

logger = logging.getLogger(__name__)

GOVINFO_COLLECTIONS_URL = "https://api.govinfo.gov/collections/BILLSTATUS/{start}"


def discover_billstatus_packages(
    api_key: str,
    start_date: str,
    page_size: int = 100,
) -> list[str]:
    """
    Find every BILLSTATUS package ID modified since a given date.

    GovInfo's /collections endpoint does not support filtering by
    Congress at the request level -- it only filters by date. So this
    function discovers everything modified since start_date, across
    all Congresses, and Congress-specific filtering happens later,
    after parsing (since each parsed bill already carries its own
    "congress" field, extracted by billstatus.py).

    This follows GovInfo's documented pagination pattern: the first
    request must include offsetMark=* to start at the beginning, and
    each response's "nextPage" field is a complete URL (including its
    own offsetMark) to follow for the next page.

    Parameters
    ----------
    api_key : str
        Your GovInfo API key.
    start_date : str
        ISO-format date (e.g. "2025-01-01T00:00:00Z") -- only packages
        modified on or after this date are returned.
    page_size : int
        How many results to request per page.

    Returns
    -------
    list[str]
        Every package ID found, e.g. ["BILLSTATUS-118hr1", ...].
        Returns an empty list if the request fails outright.
    """
    package_ids: list[str] = []
    url = GOVINFO_COLLECTIONS_URL.format(start=start_date)
    params = {
        "api_key": api_key,
        "pageSize": str(page_size),
        "offsetMark": "*",
    }
    page_bar = tqdm(desc="Discovering BILLSTATUS packages", unit="page", dynamic_ncols=True)

    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch package listing from {url}: {e}")
            break

        data = response.json()

        for package in data.get("packages", []):
            package_id = package.get("packageId")
            if package_id:
                package_ids.append(package_id)

        page_bar.update(1)
        page_bar.set_postfix(packages=len(package_ids))

        url = data.get("nextPage")
        params = {"api_key": api_key}

    page_bar.close()

    logger.info(f"Discovered {len(package_ids)} BILLSTATUS packages")
    return package_ids


def ingest_billstatus_for_congress(
    ingestor: IngestionUtils,
    congress: int,
    start_date: str,
) -> list[dict]:
    """
    Run a full BILLSTATUS ingestion pass for one Congress.

    Discovers every package modified since start_date (across all
    Congresses -- GovInfo's API can't filter this at the request
    level), fetches each one's raw XML, parses it, and keeps only the
    bills whose own "congress" field matches the one requested. Any
    individual package that fails to fetch or fails to parse is logged
    and skipped -- one bad package never aborts the whole run.

    Parameters
    ----------
    ingestor : IngestionUtils
        An already-constructed IngestionUtils instance, reused across
        every package in this loop rather than recreated each time.
    congress : int
        The Congress number to keep, e.g. 118. Used to filter the
        already-parsed results, not the discovery request itself.
    start_date : str
        Date bound passed through to discover_billstatus_packages.

    Returns
    -------
    list[dict]
        Parsed bill dicts belonging to the requested Congress.
    """
    package_ids = discover_billstatus_packages(
        api_key=ingestor.api_key,
        start_date=start_date,
    )

    parsed_bills = []

    progress = tqdm(
        package_ids,
        desc=f"Congress {congress}",
        unit="pkg",
        colour="green",
        dynamic_ncols=True,
    )

    for package_id in progress:
        xml_content = ingestor.fetch(package_id)

        if xml_content is None:
            continue

        parsed = parse_billstatus(xml_content)

        if not parsed:
            logger.warning(f"Parsed empty result for {package_id}, skipping")
            continue

        if parsed.get("congress") != str(congress):
            continue

        parsed_bills.append(parsed)

        progress.set_postfix(
            matched=len(parsed_bills),
            remaining=len(package_ids) - progress.n,
        )

    logger.info(
        f"Successfully parsed {len(parsed_bills)} of {len(package_ids)} "
        f"BILLSTATUS packages, matching Congress {congress}"
    )
    return parsed_bills


def run_full_refresh(
    congresses: list[int],
    start_date: str,
) -> dict[int, list[dict]]:
    """
    Top-level entry point: ingest BILLSTATUS data for a list of
    Congresses.

    This is the function your nightly cron job (or you, manually, while
    developing) actually calls. It creates the IngestionUtils instance
    exactly once and reuses it for every Congress and every package in
    the run -- the same "construct once, call .fetch() many times"
    pattern from when you first built IngestionUtils.

    Parameters
    ----------
    congresses : list[int]
        Which Congresses to keep, e.g. [118, 119].
    start_date : str
        Date bound applied to discovery -- only packages modified on
        or after this date are considered at all.

    Returns
    -------
    dict[int, list[dict]]
        Parsed bills, keyed by Congress number.
    """
    load_dotenv()
    api_key = os.getenv("GOVINFO_API_KEY")
    ingestor = IngestionUtils(api_key)

    results = {}
    for congress in congresses:
        logger.info(f"Starting ingestion for Congress {congress}")
        results[congress] = ingest_billstatus_for_congress(ingestor, congress, start_date)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_full_refresh(congresses=[119], start_date="2026-06-20T00:00:00Z")
    print(f"Got {len(results[119])} bills")
    if results[119]:
        print(results[119][0])  # eyeball one real parsed bill
