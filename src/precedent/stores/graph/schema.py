"""
The knowledge-graph schema, and the one function that turns a parsed bill dict
into graph nodes and edges.

There are two graph backends (Neo4j and an in-process NetworkX store), and the
single most important rule is that they load *identical* graphs -- otherwise a
GraphRAG answer would depend on which backend happened to be running. That is
guaranteed by having both backends call ``decompose_bill`` from here: the shape
of the graph is defined once, in this file, and the backends only differ in how
they persist the nodes/edges this function hands them.

Graph shape
-----------
    (Legislator)-[:SPONSORED]->(Bill)
    (Legislator)-[:COSPONSORED]->(Bill)
    (Committee)-[:REVIEWED]->(Bill)

Subjects are stored as a property on the Bill node rather than as their own
nodes: they drive the *entry point* into the graph (lexical match against the
query) but the graph *reasoning* happens over the shared-legislator and
shared-committee edges, which is what distinguishes GraphRAG from plain
keyword search.
"""

from dataclasses import dataclass, field
from typing import Any

from precedent.models import bill_id

# Node labels and relationship types, named once so query-building code and the
# loaders can refer to them symbolically instead of sprinkling string literals.
BILL = "Bill"
LEGISLATOR = "Legislator"
COMMITTEE = "Committee"

SPONSORED = "SPONSORED"
COSPONSORED = "COSPONSORED"
REVIEWED = "REVIEWED"

# Cypher run once at load time against a real Neo4j to make id lookups fast and
# to reject duplicate nodes. Harmless to re-run (IF NOT EXISTS), so the loader
# can call these on every startup without tracking whether they already ran.
CONSTRAINTS: list[str] = [
    f"CREATE CONSTRAINT bill_id IF NOT EXISTS " f"FOR (b:{BILL}) REQUIRE b.id IS UNIQUE",
    f"CREATE CONSTRAINT legislator_id IF NOT EXISTS "
    f"FOR (l:{LEGISLATOR}) REQUIRE l.bioguide_id IS UNIQUE",
    f"CREATE CONSTRAINT committee_name IF NOT EXISTS "
    f"FOR (c:{COMMITTEE}) REQUIRE c.name IS UNIQUE",
]


@dataclass
class Node:
    """A graph node: a label, a stable key, and a bag of display properties."""

    label: str
    key: str  # the unique id within its label (bill id / bioguide id / name)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A directed, typed relationship between two node keys."""

    rel_type: str
    src_label: str
    src_key: str
    dst_label: str
    dst_key: str


def decompose_bill(bill: dict[str, Any]) -> tuple[list[Node], list[Edge]]:
    """
    Turn one parsed bill dict into the nodes and edges it contributes.

    Returns every node the bill touches (the bill itself, its sponsor, each
    cosponsor, each committee) and every edge between them. Nodes are emitted
    with a stable ``key`` so that when two bills share a sponsor, both call
    sites produce the *same* Legislator node key -- the backends de-duplicate
    on that key, which is exactly what creates the shared-sponsor edges the
    GraphRAG traversal later walks.

    A missing sponsor or an entity with no usable key is simply skipped rather
    than raised on: real BILLSTATUS data has gaps, and one incomplete bill
    should never abort a whole load.
    """
    bid = bill_id(bill)
    nodes: list[Node] = [
        Node(
            label=BILL,
            key=bid,
            properties={
                "id": bid,
                "congress": bill.get("congress"),
                "bill_type": bill.get("bill_type"),
                "bill_number": bill.get("bill_number"),
                "title": bill.get("title"),
                "introduced_date": bill.get("introduced_date"),
                "outcome": bill.get("outcome"),
                "subjects": bill.get("subjects", []),
                "summary": bill.get("summary", ""),
            },
        )
    ]
    edges: list[Edge] = []

    def add_legislator(person: dict[str, Any], rel: str) -> None:
        bioguide = person.get("bioguide_id")
        if not bioguide:
            return
        nodes.append(
            Node(
                label=LEGISLATOR,
                key=bioguide,
                properties={
                    "bioguide_id": bioguide,
                    "full_name": person.get("full_name"),
                    "party": person.get("party"),
                    "state": person.get("state"),
                },
            )
        )
        edges.append(Edge(rel, LEGISLATOR, bioguide, BILL, bid))

    if bill.get("sponsor"):
        add_legislator(bill["sponsor"], SPONSORED)
    for cosponsor in bill.get("cosponsors", []):
        add_legislator(cosponsor, COSPONSORED)

    for committee in bill.get("committees", []):
        name = committee.get("name")
        if not name:
            continue
        nodes.append(
            Node(
                label=COMMITTEE,
                key=name,
                properties={"name": name, "chamber": committee.get("chamber")},
            )
        )
        edges.append(Edge(REVIEWED, COMMITTEE, name, BILL, bid))

    return nodes, edges
