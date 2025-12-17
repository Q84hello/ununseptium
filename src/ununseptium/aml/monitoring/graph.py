"""Graph-based monitoring for AML.

Provides graph analysis for transaction networks to detect
suspicious patterns and relationships.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ununseptium.aml.transactions import Transaction


class GraphNode(BaseModel):
    """A node in the transaction graph.

    Attributes:
        id: Node identifier (party ID).
        node_type: Type of node (individual, business).
        in_degree: Number of incoming edges.
        out_degree: Number of outgoing edges.
        total_in: Total amount received.
        total_out: Total amount sent.
        first_seen: First transaction timestamp.
        last_seen: Last transaction timestamp.
        properties: Additional properties.
    """

    id: str
    node_type: str = "unknown"
    in_degree: int = 0
    out_degree: int = 0
    total_in: float = 0.0
    total_out: float = 0.0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """An edge in the transaction graph.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        transaction_count: Number of transactions.
        total_amount: Total amount transferred.
        first_transaction: First transaction timestamp.
        last_transaction: Last transaction timestamp.
        transaction_ids: List of transaction IDs.
    """

    source: str
    target: str
    transaction_count: int = 0
    total_amount: float = 0.0
    first_transaction: datetime | None = None
    last_transaction: datetime | None = None
    transaction_ids: list[str] = Field(default_factory=list)


class GraphPattern(BaseModel):
    """A detected graph pattern.

    Attributes:
        id: Pattern identifier.
        pattern_type: Type of pattern detected.
        node_ids: Involved node IDs.
        edge_count: Number of edges in pattern.
        confidence: Detection confidence.
        description: Pattern description.
        evidence: Evidence details.
        detected_at: Detection timestamp.
    """

    id: str = Field(default_factory=lambda: f"GP-{uuid4().hex[:8].upper()}")
    pattern_type: str
    node_ids: list[str] = Field(default_factory=list)
    edge_count: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class GraphMonitor:
    """Graph-based transaction monitoring.

    Builds and analyzes transaction graphs to detect:
    - Hub nodes (money collectors/distributors)
    - Fan patterns (one-to-many or many-to-one)
    - Cycles (round-tripping)
    - Dense subgraphs (coordinated activity)

    Example:
        ```python
        from ununseptium.aml.monitoring import GraphMonitor

        monitor = GraphMonitor()

        # Build graph from transactions
        for txn in transactions:
            monitor.add_transaction(txn)

        # Detect patterns
        patterns = monitor.detect_patterns()
        ```
    """

    def __init__(self) -> None:
        """Initialize the graph monitor."""
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[tuple[str, str], GraphEdge] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._reverse_adjacency: dict[str, set[str]] = defaultdict(set)

    def add_transaction(self, transaction: Transaction) -> None:
        """Add a transaction to the graph.

        Args:
            transaction: Transaction to add.
        """
        sender_id = transaction.sender.id if transaction.sender else transaction.sender_id
        receiver_id = transaction.receiver.id if transaction.receiver else transaction.receiver_id

        if not sender_id or not receiver_id:
            return

        amount = float(transaction.amount)
        timestamp = transaction.timestamp

        # Update sender node
        if sender_id not in self._nodes:
            self._nodes[sender_id] = GraphNode(id=sender_id)

        sender_node = self._nodes[sender_id]
        sender_node.out_degree += 1
        sender_node.total_out += amount
        if sender_node.first_seen is None or timestamp < sender_node.first_seen:
            sender_node.first_seen = timestamp
        if sender_node.last_seen is None or timestamp > sender_node.last_seen:
            sender_node.last_seen = timestamp

        # Update receiver node
        if receiver_id not in self._nodes:
            self._nodes[receiver_id] = GraphNode(id=receiver_id)

        receiver_node = self._nodes[receiver_id]
        receiver_node.in_degree += 1
        receiver_node.total_in += amount
        if receiver_node.first_seen is None or timestamp < receiver_node.first_seen:
            receiver_node.first_seen = timestamp
        if receiver_node.last_seen is None or timestamp > receiver_node.last_seen:
            receiver_node.last_seen = timestamp

        # Update edge
        edge_key = (sender_id, receiver_id)
        if edge_key not in self._edges:
            self._edges[edge_key] = GraphEdge(source=sender_id, target=receiver_id)

        edge = self._edges[edge_key]
        edge.transaction_count += 1
        edge.total_amount += amount
        edge.transaction_ids.append(transaction.id)
        if edge.first_transaction is None or timestamp < edge.first_transaction:
            edge.first_transaction = timestamp
        if edge.last_transaction is None or timestamp > edge.last_transaction:
            edge.last_transaction = timestamp

        # Update adjacency
        self._adjacency[sender_id].add(receiver_id)
        self._reverse_adjacency[receiver_id].add(sender_id)

    def detect_patterns(
        self,
        *,
        hub_threshold: int = 10,
        fan_threshold: int = 5,
    ) -> list[GraphPattern]:
        """Detect suspicious patterns in the graph.

        Args:
            hub_threshold: Minimum connections for hub detection.
            fan_threshold: Minimum edges for fan pattern.

        Returns:
            List of detected patterns.
        """
        patterns: list[GraphPattern] = []

        # Detect hub nodes
        patterns.extend(self._detect_hubs(hub_threshold))

        # Detect fan-in patterns
        patterns.extend(self._detect_fan_in(fan_threshold))

        # Detect fan-out patterns
        patterns.extend(self._detect_fan_out(fan_threshold))

        # Detect cycles
        patterns.extend(self._detect_cycles())

        return patterns

    def _detect_hubs(self, threshold: int) -> list[GraphPattern]:
        """Detect hub nodes with high connectivity."""
        patterns: list[GraphPattern] = []

        for node_id, node in self._nodes.items():
            total_degree = node.in_degree + node.out_degree

            if total_degree >= threshold:
                # Hub node detected
                connected = list(self._adjacency[node_id] | self._reverse_adjacency[node_id])

                patterns.append(
                    GraphPattern(
                        pattern_type="hub_node",
                        node_ids=[node_id, *connected[:10]],
                        edge_count=total_degree,
                        confidence=min(total_degree / (threshold * 2), 1.0),
                        description=f"Hub node with {total_degree} connections",
                        evidence={
                            "in_degree": node.in_degree,
                            "out_degree": node.out_degree,
                            "total_in": node.total_in,
                            "total_out": node.total_out,
                        },
                    )
                )

        return patterns

    def _detect_fan_in(self, threshold: int) -> list[GraphPattern]:
        """Detect fan-in patterns (many to one)."""
        patterns: list[GraphPattern] = []

        for node_id, node in self._nodes.items():
            if node.in_degree >= threshold:
                senders = list(self._reverse_adjacency[node_id])

                patterns.append(
                    GraphPattern(
                        pattern_type="fan_in",
                        node_ids=[node_id, *senders[:10]],
                        edge_count=node.in_degree,
                        confidence=min(node.in_degree / (threshold * 2), 1.0),
                        description=f"Fan-in: {node.in_degree} senders to one receiver",
                        evidence={
                            "receiver": node_id,
                            "sender_count": len(senders),
                            "total_received": node.total_in,
                        },
                    )
                )

        return patterns

    def _detect_fan_out(self, threshold: int) -> list[GraphPattern]:
        """Detect fan-out patterns (one to many)."""
        patterns: list[GraphPattern] = []

        for node_id, node in self._nodes.items():
            if node.out_degree >= threshold:
                receivers = list(self._adjacency[node_id])

                patterns.append(
                    GraphPattern(
                        pattern_type="fan_out",
                        node_ids=[node_id, *receivers[:10]],
                        edge_count=node.out_degree,
                        confidence=min(node.out_degree / (threshold * 2), 1.0),
                        description=f"Fan-out: one sender to {node.out_degree} receivers",
                        evidence={
                            "sender": node_id,
                            "receiver_count": len(receivers),
                            "total_sent": node.total_out,
                        },
                    )
                )

        return patterns

    def _detect_cycles(self, max_length: int = 4) -> list[GraphPattern]:
        """Detect cycles in the graph."""
        patterns: list[GraphPattern] = []
        visited_cycles: set[frozenset[str]] = set()

        for start_node in self._nodes:
            cycles = self._find_cycles_from(start_node, max_length)

            for cycle in cycles:
                cycle_set = frozenset(cycle)
                if cycle_set not in visited_cycles:
                    visited_cycles.add(cycle_set)

                    patterns.append(
                        GraphPattern(
                            pattern_type="cycle",
                            node_ids=cycle,
                            edge_count=len(cycle),
                            confidence=0.9,
                            description=f"Cycle of length {len(cycle)}",
                            evidence={"cycle_length": len(cycle)},
                        )
                    )

        return patterns

    def _find_cycles_from(self, start: str, max_length: int) -> list[list[str]]:
        """Find cycles starting from a node using DFS."""
        cycles: list[list[str]] = []

        def dfs(current: str, path: list[str], depth: int) -> None:
            if depth > max_length:
                return

            for neighbor in self._adjacency.get(current, set()):
                if neighbor == start and len(path) >= 2:
                    cycles.append([*path])
                elif neighbor not in path and depth < max_length:
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()

        dfs(start, [start], 1)
        return cycles

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier.

        Returns:
            GraphNode if found.
        """
        return self._nodes.get(node_id)

    def get_edge(self, source: str, target: str) -> GraphEdge | None:
        """Get an edge by source and target.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            GraphEdge if found.
        """
        return self._edges.get((source, target))

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with graph metrics.
        """
        if not self._nodes:
            return {"node_count": 0, "edge_count": 0}

        in_degrees = [n.in_degree for n in self._nodes.values()]
        out_degrees = [n.out_degree for n in self._nodes.values()]

        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "avg_in_degree": sum(in_degrees) / len(in_degrees),
            "avg_out_degree": sum(out_degrees) / len(out_degrees),
            "max_in_degree": max(in_degrees),
            "max_out_degree": max(out_degrees),
        }
