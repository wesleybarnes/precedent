"""
Canonical data shapes shared across the whole pipeline.

The parsers (billstatus.py etc.) emit plain dicts, and the stores and engines
consume them. Rather than introduce a heavy ORM, this module pins down the one
dict shape everyone agrees on -- a "bill record" -- plus the couple of helpers
(a stable id, a display id) that every layer needs and that would otherwise get
re-implemented slightly differently in five places.

A bill record is a dict with these keys (all the parser already produces, plus
``subjects`` and ``summary`` which the seed data and entity extraction add):

    congress        str   e.g. "118"
    bill_type       str   e.g. "hr", "s", "hjres"
    bill_number     str   e.g. "1"
    title           str
    introduced_date str | None   ISO date
    sponsor         dict | None  {bioguide_id, full_name, party, state}
    cosponsors      list[dict]   same shape as sponsor
    committees      list[dict]   {name, chamber}
    subjects        list[str]    policy areas / legislative subjects
    summary         str          plain-language summary (the vector-RAG text)
    actions         list[dict]   {date, text, type}
    latest_action   dict | None  {date, text}
    outcome         str          derived label from billstatus._derive_outcome
"""

from typing import Any

# Human-readable labels for the outcome codes billstatus.py derives, used in
# both the graph tooltips and the generated answer context so nobody has to
# decode "died_after_passing_one_chamber" by eye.
OUTCOME_LABELS: dict[str, str] = {
    "became_law": "Became law",
    "vetoed": "Vetoed",
    "died_after_passing_both_chambers": "Passed both chambers, then died",
    "passed_both_chambers_pending": "Passed both chambers (pending)",
    "died_after_passing_one_chamber": "Passed one chamber, then died",
    "passed_one_chamber_pending": "Passed one chamber (pending)",
    "died_in_committee": "Died in committee",
    "pending_in_committee": "Pending in committee",
}

# Whether an outcome counts as "the bill made it into law", used by the graph
# engine to compute a base rate over retrieved precedents.
_ENACTED_OUTCOMES = {"became_law"}


def bill_id(bill: dict[str, Any]) -> str:
    """
    Build the stable, unique id for a bill: ``{congress}-{TYPE}-{number}``.

    Example: congress "118", type "hr", number "1"  ->  "118-HR-1".

    This is the primary key everywhere -- the graph node id, the vector store
    metadata key, and the id the frontend renders. Normalising the type to
    upper-case here (rather than trusting whatever case the source used) keeps
    the id deterministic so the same bill never lands under two different ids.
    """
    return f"{bill['congress']}-{str(bill['bill_type']).upper()}-{bill['bill_number']}"


def outcome_label(outcome: str | None) -> str:
    """Map a derived outcome code to a human-readable label (safe on None)."""
    if not outcome:
        return "Unknown"
    return OUTCOME_LABELS.get(outcome, outcome)


# Bill-type codes are opaque ("hres", "sjres"). These map them to the citation
# form people recognise and to a plain-English name, so the UI never shows a
# user a bare "118-HRES-1102" without a way to understand it.
_BILL_TYPE_SHORT = {
    "hr": "H.R.",
    "s": "S.",
    "hres": "H.Res.",
    "sres": "S.Res.",
    "hjres": "H.J.Res.",
    "sjres": "S.J.Res.",
    "hconres": "H.Con.Res.",
    "sconres": "S.Con.Res.",
}
_BILL_TYPE_NAME = {
    "hr": "House Bill",
    "s": "Senate Bill",
    "hres": "House Resolution",
    "sres": "Senate Resolution",
    "hjres": "House Joint Resolution",
    "sjres": "Senate Joint Resolution",
    "hconres": "House Concurrent Resolution",
    "sconres": "Senate Concurrent Resolution",
}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def bill_type_short(bill_type: str | None) -> str:
    """'hres' -> 'H.Res.' (the citation form); unknown types pass through upper-cased."""
    return _BILL_TYPE_SHORT.get((bill_type or "").lower(), (bill_type or "").upper())


def bill_type_name(bill_type: str | None) -> str:
    """'hres' -> 'House Resolution' (plain English)."""
    return _BILL_TYPE_NAME.get((bill_type or "").lower(), bill_type or "bill")


def bill_label(bill: dict[str, Any]) -> str:
    """
    A human-readable citation for a bill, e.g. 'H.Res. 1102 (118th)'.

    This is what the UI shows instead of the machine id '118-HRES-1102' -- it's
    the form the bill is actually cited by, so a user recognises it.
    """
    congress = bill.get("congress")
    try:
        congress_str = f" ({_ordinal(int(congress))})" if congress else ""
    except (TypeError, ValueError):
        congress_str = f" ({congress})" if congress else ""
    return f"{bill_type_short(bill.get('bill_type'))} {bill.get('bill_number')}{congress_str}"


def is_enacted(outcome: str | None) -> bool:
    """True if this outcome means the bill became law."""
    return outcome in _ENACTED_OUTCOMES


def enactment_rate(bills: list[dict[str, Any]]) -> float | None:
    """
    Share of the given bills that became law, or None for an empty list.

    Used as the GraphRAG "base rate" signal: once the graph has selected a set
    of precedent bills connected to the query, what fraction of *those* were
    enacted is a grounded, structural answer to "how likely is this to pass?".
    """
    if not bills:
        return None
    enacted = sum(1 for b in bills if is_enacted(b.get("outcome")))
    return enacted / len(bills)
