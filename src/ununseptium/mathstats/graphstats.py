"""Graph statistics for network analysis.

Provides graph features and community detection for AML.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class NodeFeatures(BaseModel):
    """Features computed for a graph node.

    Attributes:
        node_id: Node identifier.
        degree: Total degree.
        in_degree: Incoming edges.
        out_degree: Outgoing edges.
        clustering_coeff: Local clustering coefficient.
        pagerank: PageRank score.
        betweenness: Betweenness centrality.
        eigenvector: Eigenvector centrality.
    """

    node_id: str
    degree: int = 0
    in_degree: int = 0
    out_degree: int = 0
    clustering_coeff: float = Field(default=0.0, ge=0.0, le=1.0)
    pagerank: float = 0.0
    betweenness: float = 0.0
    eigenvector: float = 0.0


class Community(BaseModel):
    """A detected community.

    Attributes:
        id: Community identifier.
        nodes: Member node IDs.
        size: Number of nodes.
        density: Internal edge density.
        modularity_contribution: Contribution to modularity.
    """

    id: int
    nodes: list[str]
    size: int = 0
    density: float = 0.0
    modularity_contribution: float = 0.0


class TemporalPattern(BaseModel):
    """Temporal pattern in transactions.

    Attributes:
        pattern_type: Type of pattern.
        nodes: Involved nodes.
        timespan: Duration of pattern.
        count: Number of occurrences.
        evidence: Pattern evidence.
    """

    pattern_type: str
    nodes: list[str]
    timespan: timedelta
    count: int
    evidence: dict[str, Any] = Field(default_factory=dict)


class GraphFeatures:
    """Compute graph features for nodes.

    Example:
        ```python
        from ununseptium.mathstats import GraphFeatures

        features = GraphFeatures()

        # Add edges
        features.add_edge("A", "B")
        features.add_edge("B", "C")
        features.add_edge("A", "C")

        # Compute all features
        node_features = features.compute("A")
        print(f"Clustering: {node_features.clustering_coeff}")
        ```
    """

    def __init__(self) -> None:
        """Initialize feature extractor."""
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._reverse_adj: dict[str, set[str]] = defaultdict(set)
        self._edge_weights: dict[tuple[str, str], float] = {}

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
    ) -> None:
        """Add an edge.

        Args:
            source: Source node.
            target: Target node.
            weight: Edge weight.
        """
        self._adjacency[source].add(target)
        self._reverse_adj[target].add(source)
        self._edge_weights[(source, target)] = weight

    def compute(self, node_id: str) -> NodeFeatures:
        """Compute features for a node.

        Args:
            node_id: Node identifier.

        Returns:
            NodeFeatures for the node.
        """
        out_neighbors = self._adjacency.get(node_id, set())
        in_neighbors = self._reverse_adj.get(node_id, set())

        return NodeFeatures(
            node_id=node_id,
            degree=len(out_neighbors) + len(in_neighbors),
            in_degree=len(in_neighbors),
            out_degree=len(out_neighbors),
            clustering_coeff=self._clustering_coefficient(node_id),
            pagerank=self._pagerank().get(node_id, 0.0),
        )

    def compute_all(self) -> dict[str, NodeFeatures]:
        """Compute features for all nodes.

        Returns:
            Dictionary mapping node ID to features.
        """
        all_nodes = set(self._adjacency.keys()) | set(self._reverse_adj.keys())
        pageranks = self._pagerank()

        features = {}
        for node_id in all_nodes:
            out_neighbors = self._adjacency.get(node_id, set())
            in_neighbors = self._reverse_adj.get(node_id, set())

            features[node_id] = NodeFeatures(
                node_id=node_id,
                degree=len(out_neighbors) + len(in_neighbors),
                in_degree=len(in_neighbors),
                out_degree=len(out_neighbors),
                clustering_coeff=self._clustering_coefficient(node_id),
                pagerank=pageranks.get(node_id, 0.0),
            )

        return features

    def _clustering_coefficient(self, node_id: str) -> float:
        """Compute local clustering coefficient."""
        neighbors = self._adjacency.get(node_id, set()) | self._reverse_adj.get(
            node_id, set()
        )
        k = len(neighbors)

        if k < 2:
            return 0.0

        # Count edges between neighbors
        edges = 0
        neighbor_list = list(neighbors)

        for i, n1 in enumerate(neighbor_list):
            for n2 in neighbor_list[i + 1 :]:
                if n2 in self._adjacency.get(n1, set()):
                    edges += 1
                if n1 in self._adjacency.get(n2, set()):
                    edges += 1

        max_edges = k * (k - 1)
        return edges / max_edges if max_edges > 0 else 0.0

    def _pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """Compute PageRank scores."""
        all_nodes = list(
            set(self._adjacency.keys()) | set(self._reverse_adj.keys())
        )
        n = len(all_nodes)

        if n == 0:
            return {}

        node_idx = {node: i for i, node in enumerate(all_nodes)}
        pr = np.ones(n) / n

        for _ in range(max_iter):
            new_pr = np.ones(n) * (1 - damping) / n

            for node in all_nodes:
                out_neighbors = self._adjacency.get(node, set())

                if out_neighbors:
                    contribution = pr[node_idx[node]] * damping / len(out_neighbors)
                    for neighbor in out_neighbors:
                        if neighbor in node_idx:
                            new_pr[node_idx[neighbor]] += contribution

            if np.abs(new_pr - pr).sum() < tol:
                break

            pr = new_pr

        return {node: float(pr[node_idx[node]]) for node in all_nodes}


class CommunityDetector:
    """Detect communities in graphs.

    Uses label propagation algorithm.

    Example:
        ```python
        from ununseptium.mathstats import CommunityDetector

        detector = CommunityDetector()

        edges = [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F")]
        communities = detector.detect(edges)
        ```
    """

    def __init__(self, max_iter: int = 100) -> None:
        """Initialize detector.

        Args:
            max_iter: Maximum iterations for label propagation.
        """
        self.max_iter = max_iter

    def detect(
        self,
        edges: list[tuple[str, str]],
    ) -> list[Community]:
        """Detect communities using label propagation.

        Args:
            edges: List of (source, target) edges.

        Returns:
            List of detected communities.
        """
        # Build adjacency
        adjacency: dict[str, set[str]] = defaultdict(set)
        for s, t in edges:
            adjacency[s].add(t)
            adjacency[t].add(s)

        nodes = list(adjacency.keys())
        if not nodes:
            return []

        # Initialize labels
        labels = {node: i for i, node in enumerate(nodes)}

        # Label propagation
        for _ in range(self.max_iter):
            changed = False

            for node in np.random.permutation(nodes):
                neighbors = adjacency[node]
                if not neighbors:
                    continue

                # Count neighbor labels
                label_counts: dict[int, int] = defaultdict(int)
                for neighbor in neighbors:
                    label_counts[labels[neighbor]] += 1

                # Assign most common label
                max_count = max(label_counts.values())
                candidates = [l for l, c in label_counts.items() if c == max_count]
                new_label = np.random.choice(candidates)

                if new_label != labels[node]:
                    labels[node] = new_label
                    changed = True

            if not changed:
                break

        # Group by label
        communities_dict: dict[int, list[str]] = defaultdict(list)
        for node, label in labels.items():
            communities_dict[label].append(node)

        # Create Community objects
        communities = []
        for i, (label, members) in enumerate(communities_dict.items()):
            density = self._compute_density(members, adjacency)
            communities.append(
                Community(
                    id=i,
                    nodes=members,
                    size=len(members),
                    density=density,
                )
            )

        return communities

    def _compute_density(
        self,
        nodes: list[str],
        adjacency: dict[str, set[str]],
    ) -> float:
        """Compute internal edge density of a community."""
        node_set = set(nodes)
        n = len(nodes)

        if n < 2:
            return 1.0

        internal_edges = 0
        for node in nodes:
            for neighbor in adjacency.get(node, set()):
                if neighbor in node_set:
                    internal_edges += 1

        # Undirected: divide by 2
        internal_edges //= 2
        max_edges = n * (n - 1) // 2

        return internal_edges / max_edges if max_edges > 0 else 0.0


class TemporalMotifs:
    """Detect temporal patterns in transaction sequences.

    Example:
        ```python
        from ununseptium.mathstats import TemporalMotifs
        from datetime import datetime, timedelta

        detector = TemporalMotifs()

        events = [
            ("A", "B", datetime(2024, 1, 1, 10, 0)),
            ("B", "C", datetime(2024, 1, 1, 10, 5)),
            ("C", "A", datetime(2024, 1, 1, 10, 10)),
        ]

        patterns = detector.find_motifs(events)
        ```
    """

    def __init__(self, max_delta: timedelta = timedelta(hours=24)) -> None:
        """Initialize detector.

        Args:
            max_delta: Maximum time between events in a motif.
        """
        self.max_delta = max_delta

    def find_motifs(
        self,
        events: list[tuple[str, str, datetime]],
    ) -> list[TemporalPattern]:
        """Find temporal motifs in event sequence.

        Args:
            events: List of (source, target, timestamp) events.

        Returns:
            List of detected patterns.
        """
        if not events:
            return []

        patterns: list[TemporalPattern] = []

        # Sort by timestamp
        sorted_events = sorted(events, key=lambda x: x[2])

        # Find triangular motifs (A->B->C->A)
        patterns.extend(self._find_triangles(sorted_events))

        # Find fan motifs (A->B, A->C, A->D in short time)
        patterns.extend(self._find_fans(sorted_events))

        # Find chains (A->B->C->D)
        patterns.extend(self._find_chains(sorted_events))

        return patterns

    def _find_triangles(
        self,
        events: list[tuple[str, str, datetime]],
    ) -> list[TemporalPattern]:
        """Find triangular patterns."""
        patterns: list[TemporalPattern] = []

        for i, (s1, t1, ts1) in enumerate(events):
            for s2, t2, ts2 in events[i + 1 :]:
                if ts2 - ts1 > self.max_delta:
                    break

                if s2 == t1:  # A->B followed by B->?
                    for s3, t3, ts3 in events[i + 1 :]:
                        if ts3 - ts1 > self.max_delta:
                            break

                        if s3 == t2 and t3 == s1:  # B->C followed by C->A
                            patterns.append(
                                TemporalPattern(
                                    pattern_type="triangle",
                                    nodes=[s1, t1, t2],
                                    timespan=ts3 - ts1,
                                    count=1,
                                    evidence={
                                        "sequence": [
                                            (s1, t1),
                                            (s2, t2),
                                            (s3, t3),
                                        ]
                                    },
                                )
                            )

        return patterns

    def _find_fans(
        self,
        events: list[tuple[str, str, datetime]],
    ) -> list[TemporalPattern]:
        """Find fan-out patterns."""
        patterns: list[TemporalPattern] = []

        # Group by source within time windows
        by_source: dict[str, list[tuple[str, datetime]]] = defaultdict(list)

        for s, t, ts in events:
            by_source[s].append((t, ts))

        for source, targets in by_source.items():
            if len(targets) < 3:
                continue

            # Sort by time
            targets.sort(key=lambda x: x[1])

            # Sliding window
            for i in range(len(targets) - 2):
                window = [targets[i]]
                for j in range(i + 1, len(targets)):
                    if targets[j][1] - targets[i][1] <= self.max_delta:
                        window.append(targets[j])
                    else:
                        break

                if len(window) >= 3:
                    patterns.append(
                        TemporalPattern(
                            pattern_type="fan_out",
                            nodes=[source, *[t for t, _ in window]],
                            timespan=window[-1][1] - window[0][1],
                            count=len(window),
                        )
                    )
                    break  # Only one pattern per source

        return patterns

    def _find_chains(
        self,
        events: list[tuple[str, str, datetime]],
    ) -> list[TemporalPattern]:
        """Find sequential chain patterns."""
        patterns: list[TemporalPattern] = []

        for i, (s1, t1, ts1) in enumerate(events):
            chain = [(s1, t1, ts1)]
            current_target = t1
            current_time = ts1

            for s2, t2, ts2 in events[i + 1 :]:
                if ts2 - current_time > self.max_delta:
                    break

                if s2 == current_target:
                    chain.append((s2, t2, ts2))
                    current_target = t2
                    current_time = ts2

                    if len(chain) >= 4:
                        break

            if len(chain) >= 3:
                patterns.append(
                    TemporalPattern(
                        pattern_type="chain",
                        nodes=[s for s, _, _ in chain] + [chain[-1][1]],
                        timespan=chain[-1][2] - chain[0][2],
                        count=len(chain),
                    )
                )

        return patterns
