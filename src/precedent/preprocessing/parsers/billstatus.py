"""
Parser for BILLSTATUS XML files from GovInfo.

BILLSTATUS XML contains the structured metadata about a bill's lifecycle:
sponsors, cosponsors, committees, and the full history of actions taken on
the bill. This is the data that populates the graph database (sponsors,
committees, and actions become nodes/edges) -- it is NOT the actual text of
the bill itself (that lives in the BILLS collection and is handled
separately, in bills.py).

Schema note
-----------
GovInfo changed the BILLSTATUS schema in late 2022 (see
https://github.com/usgpo/bill-status/issues/200). Older files (roughly
108th-117th Congress, pre-2022 reprocessing) use different tag names and
nesting than newer files. The differences relevant to this parser:

    field          | old schema                        | new schema
    ---------------|------------------------------------|--------------------
    bill number    | billNumber                         | number
    bill type      | billType                           | type
    committees     | committees/billCommittees/item     | committees/item
    subjects       | subjects/billSubjects/...           | subjects/legislativeSubjects
    summaries      | summaries/billSummaries/item        | summaries/summary

Rather than writing two separate parsers, every lookup below tries the new
path first, then falls back to the old path if nothing was found. This
keeps one parser working across the full historical range of files.
"""

import logging
import re
from datetime import date

from lxml import etree
from precedent.preprocessing.parsers.xml_helpers import get_text, find_items

logger = logging.getLogger(__name__)

# The 1st Congress began January 3, 1789, and each numbered Congress since
# has run for a fixed two-year term, ending on January 3rd two years later.
# This lets us compute the end date of *any* Congress number from a single
# anchor point, instead of hardcoding a lookup table that would need a
# manual update every two years.
_FIRST_CONGRESS_START_YEAR = 1789
_FIRST_CONGRESS_NUMBER = 1
_CONGRESS_TERM_YEARS = 2


def parse_billstatus(xml_content: str, as_of: date | None = None) -> dict:
    """
    Parse a BILLSTATUS XML string into a structured dictionary.

    Parameters
    ----------
    xml_content : str
        Raw XML text, exactly as returned by IngestionUtils.fetch()
        for a BILLSTATUS package.
    as_of : date, optional
        The date to treat as "today" when deciding whether a bill's
        Congress has ended. Defaults to the real current date. Exposed
        as a parameter (rather than calling date.today() deep inside
        the outcome logic) so tests can pass a fixed date and get a
        deterministic result instead of one that silently changes
        depending on when the test happens to run.

    Returns
    -------
    dict
        Fields needed to populate the graph: congress, bill_type,
        bill_number, title, sponsor, cosponsors, committees, actions,
        latest_action, and a derived outcome label. Returns an empty
        dict if the XML cannot be parsed at all, so a single bad file
        doesn't crash a batch ingestion run.
    """
    try:
        root = etree.fromstring(xml_content.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        logger.warning(f"Failed to parse BILLSTATUS XML: {e}")
        return {}

    bill = root.find("bill")
    if bill is None:
        logger.warning("No <bill> element found in BILLSTATUS XML")
        return {}

    congress = get_text(bill, "congress")
    today = as_of if as_of is not None else date.today()
    congress_ended = _has_congress_ended(congress, today)

    return {
        "congress": congress,
        "bill_type": get_text(bill, "type", "billType"),
        "bill_number": get_text(bill, "number", "billNumber"),
        "title": get_text(bill, "title"),
        "introduced_date": get_text(bill, "introducedDate"),
        "sponsor": _parse_sponsor(bill),
        "cosponsors": _parse_cosponsors(bill),
        "committees": _parse_committees(bill),
        "subjects": _parse_subjects(bill),
        "summary": _parse_summary(bill),
        "actions": _parse_actions(bill),
        "latest_action": _parse_latest_action(bill),
        "outcome": _derive_outcome(bill, congress_ended),
    }


def _parse_summary(bill) -> str:
    """
    Extract the bill's plain-language summary -- the text the vector engine
    chunks and embeds.

    BILLSTATUS carries one or more CRS summaries under <summaries>. Newer files
    nest them as <summaries><summary>, older ones as
    <summaries><billSummaries><item>. We take the *last* (most recent) summary's
    <text>, which is HTML, and strip the tags down to plain prose. Returns "" if
    the bill has no summary yet (common for freshly introduced bills), in which
    case the title alone carries the bill in the vector index.
    """
    items = find_items(bill, "summaries/summary", "summaries/billSummaries/item")
    if not items:
        return ""
    # The HTML summary lives under <cdata><text> in the modern schema and under
    # a direct <text> in the old one -- try the modern nesting first.
    raw = get_text(items[-1], "cdata/text", "text") or ""
    # Summary text is HTML: drop tags, unescape the few common entities, and
    # collapse the whitespace the tag removal leaves behind.
    text = re.sub(r"<[^>]+>", " ", raw)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#8217;", "'")
        .replace("&quot;", '"')
    )
    return re.sub(r"\s+", " ", text).strip()


def _parse_subjects(bill) -> list[str]:
    """
    Extract the bill's legislative subjects and policy area.

    These populate the graph's subject-based seed matching. The policy area is a
    single high-level bucket; legislative subjects are the finer tags. Both the
    new (subjects/legislativeSubjects) and old
    (subjects/billSubjects/legislativeSubjects) nestings are handled.
    """
    subjects: list[str] = []
    policy_area = get_text(bill, "policyArea/name", "subjects/billSubjects/policyArea/name")
    if policy_area:
        subjects.append(policy_area)
    for item in find_items(
        bill,
        "subjects/legislativeSubjects/item",
        "subjects/billSubjects/legislativeSubjects/item",
    ):
        name = get_text(item, "name")
        if name and name not in subjects:
            subjects.append(name)
    return subjects


def _congress_end_date(congress: str | None) -> date | None:
    """
    Compute the end date of a numbered Congress.

    Every Congress runs for exactly two years and ends on January 3rd.
    The 119th Congress, for example, runs Jan 3 2025 - Jan 3 2027.
    Computed from the 1st Congress's known start year rather than
    hardcoded, so this keeps working for the 120th, 121st, etc.
    without any future edits.

    Returns None if `congress` is missing or not a valid integer, so
    callers can treat "we don't know" as a distinct case rather than
    crashing on bad/missing data.
    """
    if not congress:
        return None
    try:
        congress_num = int(congress)
    except ValueError:
        logger.warning(f"Non-numeric congress value: {congress!r}")
        return None

    start_year = _FIRST_CONGRESS_START_YEAR + (
        (congress_num - _FIRST_CONGRESS_NUMBER) * _CONGRESS_TERM_YEARS
    )
    end_year = start_year + _CONGRESS_TERM_YEARS
    return date(end_year, 1, 3)


def _has_congress_ended(congress: str | None, today: date) -> bool:
    """
    Decide whether the given Congress's term has already ended as of
    `today`. Returns False (i.e. "assume still pending") if the end
    date can't be determined at all -- an unknown state should not be
    silently treated as "died", since that's a much stronger claim.
    """
    end_date = _congress_end_date(congress)
    if end_date is None:
        return False
    return today > end_date


def _parse_sponsor(bill) -> dict | None:
    """
    Extract the primary sponsor of the bill.

    BILLSTATUS lists sponsors under <sponsors><item>...</item></sponsors>.
    In practice there is exactly one sponsor (cosponsors are a separate,
    much longer list) -- so this returns a single dict, not a list, to
    make downstream code (e.g. creating one SPONSORED_BY edge) simpler.
    """
    sponsor_item = bill.find("sponsors/item")
    if sponsor_item is None:
        return None

    return {
        "bioguide_id": get_text(sponsor_item, "bioguideId"),
        "full_name": get_text(sponsor_item, "fullName"),
        "party": get_text(sponsor_item, "party"),
        "state": get_text(sponsor_item, "state"),
    }


def _parse_cosponsors(bill) -> list[dict]:
    """
    Extract every cosponsor of the bill.

    Unlike the single sponsor, cosponsors can number in the dozens or
    hundreds, so this loops over every <item> under <cosponsors> and
    returns a list -- this is the "repeated item" pattern, the same
    shape you'll see again in committees and actions below.
    """
    cosponsors = []
    for item in bill.findall("cosponsors/item"):
        cosponsors.append(
            {
                "bioguide_id": get_text(item, "bioguideId"),
                "full_name": get_text(item, "fullName"),
                "party": get_text(item, "party"),
                "state": get_text(item, "state"),
            }
        )
    return cosponsors


def _parse_committees(bill) -> list[dict]:
    """
    Extract the committees the bill was referred to.

    Committees are repeated <item> elements, but the container around
    them moved between schema versions: old files nest an extra
    <billCommittees> layer (committees/billCommittees/item), new files
    don't (committees/item). find_items() tries the new path first.
    """
    committees = []
    for item in find_items(bill, "committees/item", "committees/billCommittees/item"):
        committees.append(
            {
                "name": get_text(item, "name"),
                "chamber": get_text(item, "chamber"),
            }
        )
    return committees


def _parse_actions(bill) -> list[dict]:
    """
    Extract the full chronological action history of the bill.

    This is the single most important field for your graph, since it's
    where a bill's actual outcome lives -- introduced, referred, passed,
    vetoed, became law, etc. are all just entries in this list, not a
    separate "status" field BILLSTATUS hands you directly.
    """
    actions = []
    for item in bill.findall("actions/item"):
        actions.append(
            {
                "date": get_text(item, "actionDate"),
                "text": get_text(item, "text"),
                "type": get_text(item, "type"),
            }
        )
    return actions


def _parse_latest_action(bill) -> dict | None:
    """
    Extract the single most recent action on the bill.

    BILLSTATUS provides this as a separate, already-extracted field
    (<latestAction>) specifically so consumers don't have to sort the
    full actions list just to answer "what's the current status?" --
    worth pulling out separately here for the same reason.
    """
    latest = bill.find("latestAction")
    if latest is None:
        return None

    return {
        "date": get_text(latest, "actionDate"),
        "text": get_text(latest, "text"),
    }


def _derive_outcome(bill, congress_ended: bool) -> str:
    """
    Derive an outcome label by scanning the action history for known
    phrases, disambiguated by whether the bill's Congress has ended.

    BILLSTATUS does not hand you a single "did this become law" field --
    that has to be inferred by reading through the action text. This
    function does that inference, kept separate from _parse_actions
    since it's a different kind of work (interpretation, not extraction).

    Why `congress_ended` is a parameter, not computed in here: this
    function only has access to the <bill> XML element, which has no
    idea what today's date is. Whether a Congress has ended is a fact
    about the outside world, not about the bill itself, so it's the
    caller's job (parse_billstatus) to work that out once and hand it
    down -- that keeps this function a pure mapping from
    (XML content, one boolean) -> outcome label, which is easy to unit
    test with fixed inputs and no mocking of "today" required.

    The order of these checks matters: a bill that became law will also
    have "passed house" and "passed senate" earlier in its history, so
    the strongest outcome (became_law) must be checked first, or a
    later, weaker match would incorrectly win.
    """
    actions = bill.findall("actions/item")
    all_text = " ".join(
        (item.find("text").text or "") for item in actions if item.find("text") is not None
    ).lower()

    if "became public law" in all_text:
        return "became_law"
    if "vetoed" in all_text:
        return "vetoed"

    passed_senate = "passed senate" in all_text
    passed_house = "passed house" in all_text

    if passed_senate and passed_house:
        # Passed both chambers but never became law and wasn't vetoed
        # above -- e.g. stuck in conference, or a pocket veto. Only a
        # real "died" outcome once its Congress is actually over;
        # otherwise it's still alive and could yet become law.
        return (
            "died_after_passing_both_chambers" if congress_ended else "passed_both_chambers_pending"
        )

    if passed_senate or passed_house:
        return "died_after_passing_one_chamber" if congress_ended else "passed_one_chamber_pending"

    # Never passed either chamber. This is the case that used to be a
    # single catch-all ("pending_or_died_in_committee") -- now split by
    # whether the bill's Congress has actually ended.
    return "died_in_committee" if congress_ended else "pending_in_committee"
