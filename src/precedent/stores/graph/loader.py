"""
The graph store: one interface, two interchangeable backends.

``GraphStore`` is the abstract contract the rest of the app codes against. It
deliberately exposes small, uniform *primitives* (who sponsored this bill,
which bills did this legislator touch, ...) rather than a big "do GraphRAG"
method. The actual GraphRAG traversal is composed from these primitives in
queries.py, so the interesting algorithm lives in exactly one readable place
and runs unchanged on either backend.

Two backends implement the contract:

* ``InMemoryGraphStore`` -- a plain-Python adjacency-map graph. No server, no
  driver, nothing to install. This is what runs on a laptop and in the unit
  tests, and it's what makes the whole app demonstrable without Docker.

* ``Neo4jGraphStore`` -- backs the same primitives with Cypher against a real
  Neo4j (the one in docker-compose). Selected when GRAPH_BACKEND=neo4j.

``build_graph_store`` reads Settings and hands back whichever one is configured,
so no caller ever names a concrete class.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from precedent.assembly.app_config import Settings
from precedent.stores.graph.schema import (
    BILL,
    COMMITTEE,
    CONSTRAINTS,
    LEGISLATOR,
    Edge,
    Node,
    decompose_bill,
)

logger = logging.getLogger(__name__)


class GraphStore(ABC):
    """Abstract contract for a legislative knowledge graph."""

    @abstractmethod
    def add_bills(self, bills: list[dict[str, Any]]) -> None:
        """Load (or upsert) a batch of parsed bill dicts into the graph."""

    @abstractmethod
    def all_bill_ids(self) -> list[str]:
        """Every Bill node id currently in the graph."""

    @abstractmethod
    def get_bill(self, bid: str) -> dict[str, Any] | None:
        """The stored property bag for one Bill node, or None if absent."""

    @abstractmethod
    def legislators_of(self, bid: str) -> set[str]:
        """Bioguide ids of the sponsor and every cosponsor of a bill."""

    @abstractmethod
    def committees_of(self, bid: str) -> set[str]:
        """Names of every committee that reviewed a bill."""

    @abstractmethod
    def bills_of_legislator(self, bioguide: str) -> set[str]:
        """Every bill this legislator sponsored or cosponsored."""

    @abstractmethod
    def bills_of_committee(self, name: str) -> set[str]:
        """Every bill this committee reviewed."""

    @abstractmethod
    def legislator(self, bioguide: str) -> dict[str, Any] | None:
        """Property bag for a Legislator node."""

    @abstractmethod
    def committee(self, name: str) -> dict[str, Any] | None:
        """Property bag for a Committee node."""

    def count_bills(self) -> int:
        """How many bills are loaded -- handy for a /health readiness check."""
        return len(self.all_bill_ids())

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release any backend resources (a no-op for the in-memory store)."""


class InMemoryGraphStore(GraphStore):
    """
    A NetworkX-free adjacency-map graph that lives entirely in the process.

    Everything is kept in plain dicts and sets, which is all the GraphRAG
    primitives need: forward maps (bill -> its legislators / committees) and
    reverse maps (legislator / committee -> its bills). Loading a bill just
    means merging its decomposed nodes/edges into these maps, de-duplicating
    entities by key so shared sponsors and committees naturally become the
    connective tissue the traversal walks.
    """

    def __init__(self) -> None:
        self._bills: dict[str, dict[str, Any]] = {}
        self._legislators: dict[str, dict[str, Any]] = {}
        self._committees: dict[str, dict[str, Any]] = {}
        self._bill_legislators: dict[str, set[str]] = defaultdict(set)
        self._bill_committees: dict[str, set[str]] = defaultdict(set)
        self._legislator_bills: dict[str, set[str]] = defaultdict(set)
        self._committee_bills: dict[str, set[str]] = defaultdict(set)

    def add_bills(self, bills: list[dict[str, Any]]) -> None:
        for bill in bills:
            nodes, edges = decompose_bill(bill)
            self._merge_nodes(nodes)
            self._merge_edges(edges)
        logger.info("InMemoryGraphStore now holds %d bills", len(self._bills))

    def _merge_nodes(self, nodes: list[Node]) -> None:
        for node in nodes:
            if node.label == BILL:
                self._bills[node.key] = node.properties
            elif node.label == LEGISLATOR:
                # Keep the richest record seen: later bills may fill in a name
                # or party that an earlier appearance left blank.
                self._legislators.setdefault(node.key, {}).update(
                    {k: v for k, v in node.properties.items() if v is not None}
                )
            elif node.label == COMMITTEE:
                self._committees.setdefault(node.key, {}).update(
                    {k: v for k, v in node.properties.items() if v is not None}
                )

    def _merge_edges(self, edges: list[Edge]) -> None:
        for edge in edges:
            if edge.src_label == LEGISLATOR:
                self._bill_legislators[edge.dst_key].add(edge.src_key)
                self._legislator_bills[edge.src_key].add(edge.dst_key)
            elif edge.src_label == COMMITTEE:
                self._bill_committees[edge.dst_key].add(edge.src_key)
                self._committee_bills[edge.src_key].add(edge.dst_key)

    def all_bill_ids(self) -> list[str]:
        return list(self._bills)

    def get_bill(self, bid: str) -> dict[str, Any] | None:
        return self._bills.get(bid)

    def legislators_of(self, bid: str) -> set[str]:
        return set(self._bill_legislators.get(bid, set()))

    def committees_of(self, bid: str) -> set[str]:
        return set(self._bill_committees.get(bid, set()))

    def bills_of_legislator(self, bioguide: str) -> set[str]:
        return set(self._legislator_bills.get(bioguide, set()))

    def bills_of_committee(self, name: str) -> set[str]:
        return set(self._committee_bills.get(name, set()))

    def legislator(self, bioguide: str) -> dict[str, Any] | None:
        return self._legislators.get(bioguide)

    def committee(self, name: str) -> dict[str, Any] | None:
        return self._committees.get(name)


class Neo4jGraphStore(GraphStore):
    """
    The same primitives, backed by Cypher against a real Neo4j server.

    Each primitive is one small parameterised query. That keeps the class a
    thin translation layer -- the GraphRAG algorithm in queries.py issues the
    same sequence of primitive calls regardless of backend, so behaviour stays
    identical to the in-memory store.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        # Imported lazily so that a laptop running the default in-memory
        # backend never needs the neo4j driver import to succeed.
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self._driver.session() as session:
            for statement in CONSTRAINTS:
                session.run(statement)

    def add_bills(self, bills: list[dict[str, Any]]) -> None:
        with self._driver.session() as session:
            for bill in bills:
                nodes, edges = decompose_bill(bill)
                session.execute_write(self._write_bill, nodes, edges)
        logger.info("Neo4jGraphStore loaded %d bills", len(bills))

    @staticmethod
    def _write_bill(tx, nodes: list[Node], edges: list[Edge]) -> None:
        for node in nodes:
            key_field = {
                BILL: "id",
                LEGISLATOR: "bioguide_id",
                COMMITTEE: "name",
            }[node.label]
            tx.run(
                f"MERGE (n:{node.label} {{{key_field}: $key}}) SET n += $props",
                key=node.key,
                props=node.properties,
            )
        for edge in edges:
            src_key_field = {
                BILL: "id",
                LEGISLATOR: "bioguide_id",
                COMMITTEE: "name",
            }[edge.src_label]
            tx.run(
                f"MATCH (s:{edge.src_label} {{{src_key_field}: $src}}) "
                f"MATCH (d:{BILL} {{id: $dst}}) "
                f"MERGE (s)-[:{edge.rel_type}]->(d)",
                src=edge.src_key,
                dst=edge.dst_key,
            )

    def all_bill_ids(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run(f"MATCH (b:{BILL}) RETURN b.id AS id")
            return [record["id"] for record in result]

    def get_bill(self, bid: str) -> dict[str, Any] | None:
        with self._driver.session() as session:
            record = session.run(f"MATCH (b:{BILL} {{id: $id}}) RETURN b", id=bid).single()
            return dict(record["b"]) if record else None

    def legislators_of(self, bid: str) -> set[str]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (l:{LEGISLATOR})-[:SPONSORED|COSPONSORED]->(b:{BILL} {{id: $id}}) "
                f"RETURN l.bioguide_id AS k",
                id=bid,
            )
            return {r["k"] for r in result}

    def committees_of(self, bid: str) -> set[str]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (c:{COMMITTEE})-[:REVIEWED]->(b:{BILL} {{id: $id}}) " f"RETURN c.name AS k",
                id=bid,
            )
            return {r["k"] for r in result}

    def bills_of_legislator(self, bioguide: str) -> set[str]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (l:{LEGISLATOR} {{bioguide_id: $k}})-[:SPONSORED|COSPONSORED]->(b:{BILL}) "
                f"RETURN b.id AS id",
                k=bioguide,
            )
            return {r["id"] for r in result}

    def bills_of_committee(self, name: str) -> set[str]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (c:{COMMITTEE} {{name: $k}})-[:REVIEWED]->(b:{BILL}) " f"RETURN b.id AS id",
                k=name,
            )
            return {r["id"] for r in result}

    def legislator(self, bioguide: str) -> dict[str, Any] | None:
        with self._driver.session() as session:
            record = session.run(
                f"MATCH (l:{LEGISLATOR} {{bioguide_id: $k}}) RETURN l", k=bioguide
            ).single()
            return dict(record["l"]) if record else None

    def committee(self, name: str) -> dict[str, Any] | None:
        with self._driver.session() as session:
            record = session.run(f"MATCH (c:{COMMITTEE} {{name: $k}}) RETURN c", k=name).single()
            return dict(record["c"]) if record else None

    def close(self) -> None:
        self._driver.close()


def build_graph_store(settings: Settings) -> GraphStore:
    """Construct whichever backend Settings selects, defaulting to in-memory."""
    if settings.graph_backend == "neo4j":
        logger.info("Using Neo4j graph backend at %s", settings.neo4j_uri)
        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    logger.info("Using in-memory graph backend")
    return InMemoryGraphStore()
