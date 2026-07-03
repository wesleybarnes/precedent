"""
Run the golden-set evaluation: score both engines' retrieval and print a
side-by-side report.

This is how you'd answer "is GraphRAG actually retrieving better precedents than
vector search on this data?" quantitatively, instead of by eyeballing one query
in the UI. For each labelled query it pulls each engine's retrieved precedent
bills, scores them against the human-labelled relevant set, and averages
precision@k, recall@k, and MRR across the set -- plus how much the two engines
overlap (Jaccard), which is a proxy for how *differently* they reason.

Run:  python eval/run_eval.py   (uses the local no-server backends + seed data)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow both `python -m eval.run_eval` and `python eval/run_eval.py` by ensuring
# the repo root (which holds the `eval` package) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import (  # noqa: E402
    average,
    jaccard,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from precedent.assembly.switchboard import Switchboard

K = 6
_GOLDEN = Path(__file__).resolve().parent / "golden_set.json"


def _distinct(seq: list[str]) -> list[str]:
    """Order-preserving de-dup -- vector chunks can repeat a bill id."""
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def graph_retrieved(sb: Switchboard, query: str) -> list[str]:
    out = sb.graph_engine.execute(query)
    return [p["id"] for p in out.retrieval.get("precedents", [])]


def vector_retrieved(sb: Switchboard, query: str) -> list[str]:
    out = sb.vector_engine.execute(query)
    return _distinct([c["bill_id"] for c in out.retrieval.get("chunks", [])])


def main() -> None:
    with open(_GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    sb = Switchboard()
    rows = []
    g_p, g_r, g_mrr = [], [], []
    v_p, v_r, v_mrr = [], [], []
    overlaps = []

    for case in golden:
        query = case["query"]
        relevant = set(case["relevant"])

        g = graph_retrieved(sb, query)
        v = vector_retrieved(sb, query)

        gp, gr, gm = (
            precision_at_k(g, relevant, K),
            recall_at_k(g, relevant, K),
            reciprocal_rank(g, relevant),
        )
        vp, vr, vm = (
            precision_at_k(v, relevant, K),
            recall_at_k(v, relevant, K),
            reciprocal_rank(v, relevant),
        )

        g_p.append(gp)
        g_r.append(gr)
        g_mrr.append(gm)
        v_p.append(vp)
        v_r.append(vr)
        v_mrr.append(vm)
        overlaps.append(jaccard(g, v))

        rows.append((query[:44], gp, gr, vp, vr, jaccard(g, v)))

    print(f"\nGolden-set evaluation  (k={K}, {len(golden)} queries)\n")
    header = f"{'query':<46}{'G P@k':>7}{'G R@k':>7}{'V P@k':>7}{'V R@k':>7}{'overlap':>9}"
    print(header)
    print("-" * len(header))
    for q, gp, gr, vp, vr, ov in rows:
        print(f"{q:<46}{gp:>7.2f}{gr:>7.2f}{vp:>7.2f}{vr:>7.2f}{ov:>9.2f}")

    print("-" * len(header))
    print(
        f"{'AVERAGE':<46}{average(g_p):>7.2f}{average(g_r):>7.2f}"
        f"{average(v_p):>7.2f}{average(v_r):>7.2f}{average(overlaps):>9.2f}"
    )
    print(f"\nMRR   graph={average(g_mrr):.3f}   vector={average(v_mrr):.3f}")
    print(
        "Overlap is the mean Jaccard of the two engines' retrieved sets: "
        "lower means they are reasoning more differently.\n"
    )


if __name__ == "__main__":
    main()
