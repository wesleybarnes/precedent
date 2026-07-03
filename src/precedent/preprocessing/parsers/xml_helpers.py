"""
Shared XML parsing helpers, used by every collection-specific parser
(billstatus.py, bills.py, billsum.py).

These functions know nothing about any particular GovInfo collection's
schema -- they're generic "safely pull text or repeated items out of an
XML element" utilities. Schema-specific knowledge (which tags to look
for, what fields a bill has) belongs in the individual parser files that
import these, not here.
"""


def get_text(element, *tags: str) -> str | None:
    """
    Safely extract the text content of a child element, trying each
    given path in order until one is found.

    Accepting multiple tags (instead of just one) is what lets a single
    call site handle multiple schema versions of the same field -- e.g.
    get_text(bill, "number", "billNumber") tries the modern path first,
    then falls back to an older path if the modern one isn't present.

    Returns None if none of the given paths exist, instead of raising
    an error. Many fields are genuinely optional -- treating a missing
    field as an error would crash the parser on perfectly normal input.
    """
    for tag in tags:
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
    return None


def find_items(element, *paths: str) -> list:
    """
    Find repeated <item> elements, trying each given container path in
    order until one returns results.

    This is the equivalent of get_text(), but for lists rather than
    single values -- needed when the whole container path can differ
    between schema versions, not just a single tag name.
    """
    for path in paths:
        items = element.findall(path)
        if items:
            return items
    return []
