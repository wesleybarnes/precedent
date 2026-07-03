"""
Retrieval metrics for comparing the two engines against a labelled golden set.

These are the standard information-retrieval measures, applied to the ranked
list of *precedent bill ids* each engine returns (the graph engine's precedents,
or the distinct bills behind the vector engine's top chunks). Keeping them here
as small, readable functions -- rather than pulling in a metrics library -- fits
the project's teaching goal: a learner can see exactly how precision@k or MRR is
computed.
"""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k retrieved."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    hits = len(top & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item (0 if none retrieved)."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def jaccard(a: list[str], b: list[str]) -> float:
    """Overlap between two engines' retrieved sets -- how much they agree."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
