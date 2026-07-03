"""
Deriving the ``subjects`` field that the graph's seed-matching relies on.

BILLSTATUS XML carries a subjects list, but it isn't always populated and older
schema versions nest it differently. Rather than depend on that being present,
this module derives a compact set of subject keywords from whatever text a bill
does have (its title and summary), normalising them into the ``subjects`` list
the rest of the pipeline expects.

This is deliberately a lightweight keyword pass, not a trained NER model: the
subjects it produces are the *entry point* into the graph (see
stores/graph/queries.py), and the graph traversal -- not this step -- is what
does the real precedent-finding. Keeping it simple also keeps it inspectable,
which matters for a project whose whole point is showing how retrieval works.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# A small controlled vocabulary of policy areas. When one of these terms (or a
# listed synonym) appears in a bill's text, that policy area is attached as a
# subject. A controlled list keeps subjects consistent across bills so two bills
# about the same area actually share a subject string (and thus a seed match),
# instead of one saying "healthcare" and the other "health care".
_POLICY_AREAS: dict[str, list[str]] = {
    "taxation": ["tax", "taxation", "irs", "revenue", "deduction", "credit"],
    "healthcare": ["health", "healthcare", "medicare", "medicaid", "insurance", "drug"],
    "immigration": ["immigration", "immigrant", "visa", "border", "asylum", "citizenship"],
    "defense": ["defense", "military", "armed", "veterans", "security", "weapon"],
    "environment": ["environment", "climate", "emissions", "energy", "pollution", "clean"],
    "education": ["education", "school", "student", "college", "teacher", "tuition"],
    "technology": ["technology", "internet", "data", "privacy", "cyber", "artificial"],
    "labor": ["labor", "wage", "worker", "employment", "union", "workplace"],
    "trade": ["trade", "tariff", "export", "import", "commerce", "customs"],
    "finance": ["bank", "finance", "financial", "securities", "credit", "loan"],
    "agriculture": ["agriculture", "farm", "crop", "rural", "food", "livestock"],
    "housing": ["housing", "rent", "mortgage", "homeless", "affordable", "tenant"],
}


def extract_subjects(bill: dict[str, Any], max_subjects: int = 4) -> list[str]:
    """
    Derive a bill's policy-area subjects from its title and summary.

    Counts how many vocabulary terms of each policy area appear, and returns the
    areas with the most hits (up to ``max_subjects``). If the bill already came
    with subjects from BILLSTATUS, those are preserved and merged in -- this
    only *adds* signal, it never discards what the source provided.
    """
    text = f"{bill.get('title', '')} {bill.get('summary', '')}".lower()
    words = set(re.findall(r"[a-z]+", text))

    hits: Counter[str] = Counter()
    for area, terms in _POLICY_AREAS.items():
        overlap = sum(1 for term in terms if term in words)
        if overlap:
            hits[area] = overlap

    derived = [area for area, _ in hits.most_common(max_subjects)]

    existing = [s.lower() for s in bill.get("subjects", []) if isinstance(s, str)]
    # Preserve order, drop duplicates: existing subjects first, then derived.
    merged: list[str] = []
    for subject in existing + derived:
        if subject not in merged:
            merged.append(subject)
    return merged


def enrich_bills(bills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Attach derived subjects to a batch of bills in place, returning the list.

    Called once during ingestion (and when loading seed data) so that every
    bill carries a populated ``subjects`` field before it reaches the stores.
    """
    for bill in bills:
        bill["subjects"] = extract_subjects(bill)
    return bills
