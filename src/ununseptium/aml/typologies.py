"""Typology detection for AML.

Provides pattern recognition for money laundering typologies
including structuring, layering, and other suspicious patterns.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ununseptium.aml.transactions import Transaction, TransactionBatch


class TypologyType(str, Enum):
    """Types of money laundering typologies."""

    STRUCTURING = "structuring"
    SMURFING = "smurfing"
    LAYERING = "layering"
    ROUND_TRIPPING = "round_tripping"
    VELOCITY = "velocity"
    UNUSUAL_PATTERN = "unusual_pattern"
    HIGH_RISK_GEOGRAPHY = "high_risk_geography"
    SHELL_COMPANY = "shell_company"
    TRADE_BASED = "trade_based"
    CRYPTO_MIXING = "crypto_mixing"
    CUSTOM = "custom"


class TypologyMatch(BaseModel):
    """A detected typology match.

    Attributes:
        id: Match identifier.
        typology_type: Type of typology.
        typology_id: ID of the matched typology rule.
        confidence: Match confidence (0.0 to 1.0).
        transaction_ids: Involved transaction IDs.
        party_ids: Involved party IDs.
        amount_total: Total amount involved.
        time_span: Time span of the pattern.
        evidence: Evidence details.
        detected_at: Detection timestamp.
    """

    id: str = Field(default_factory=lambda: f"TM-{uuid4().hex[:8].upper()}")
    typology_type: TypologyType
    typology_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    transaction_ids: list[str] = Field(default_factory=list)
    party_ids: list[str] = Field(default_factory=list)
    amount_total: Decimal = Decimal("0")
    time_span: timedelta | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class TypologyRule(BaseModel):
    """A rule for detecting a typology.

    Attributes:
        field: Transaction field to evaluate.
        operator: Comparison operator.
        value: Value to compare against.
    """

    field: str
    operator: str
    value: Any


class Typology(BaseModel):
    """A money laundering typology definition.

    Attributes:
        id: Typology identifier.
        name: Human-readable name.
        typology_type: Type classification.
        description: Detailed description.
        rules: Detection rules.
        threshold: Detection threshold.
        time_window: Time window for pattern detection.
        min_transactions: Minimum transactions for pattern.
        enabled: Whether typology is active.
        severity: Severity level (1-10).
    """

    id: str = Field(default_factory=lambda: f"TYP-{uuid4().hex[:8].upper()}")
    name: str
    typology_type: TypologyType
    description: str = ""
    rules: list[TypologyRule] = Field(default_factory=list)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    time_window: timedelta = timedelta(days=30)
    min_transactions: int = Field(default=2, ge=1)
    enabled: bool = True
    severity: int = Field(default=5, ge=1, le=10)


class TypologyDetector:
    """Detect money laundering typologies in transactions.

    Analyzes transaction patterns to identify potential
    money laundering activities.

    Example:
        ```python
        from ununseptium.aml import TypologyDetector, TransactionBatch

        detector = TypologyDetector()

        # Analyze a batch of transactions
        matches = detector.detect(batch)

        for match in matches:
            print(f"Detected: {match.typology_type} (confidence: {match.confidence})")
        ```
    """

    def __init__(self) -> None:
        """Initialize the detector with default typologies."""
        self._typologies: list[Typology] = []
        self._load_default_typologies()

    def _load_default_typologies(self) -> None:
        """Load default typology definitions."""
        self._typologies = [
            Typology(
                id="TYP-STRUCT-001",
                name="Structuring (Smurfing)",
                typology_type=TypologyType.STRUCTURING,
                description="Multiple transactions just below reporting threshold",
                threshold=0.8,
                time_window=timedelta(days=7),
                min_transactions=3,
                severity=8,
            ),
            Typology(
                id="TYP-VELOCITY-001",
                name="High Velocity",
                typology_type=TypologyType.VELOCITY,
                description="Unusually high transaction frequency",
                threshold=0.7,
                time_window=timedelta(days=1),
                min_transactions=10,
                severity=6,
            ),
            Typology(
                id="TYP-ROUND-001",
                name="Round Tripping",
                typology_type=TypologyType.ROUND_TRIPPING,
                description="Funds returning to origin through intermediaries",
                threshold=0.75,
                time_window=timedelta(days=30),
                min_transactions=3,
                severity=9,
            ),
            Typology(
                id="TYP-LAYER-001",
                name="Layering",
                typology_type=TypologyType.LAYERING,
                description="Complex transaction chains to obscure source",
                threshold=0.7,
                time_window=timedelta(days=14),
                min_transactions=5,
                severity=8,
            ),
        ]

    def add_typology(self, typology: Typology) -> None:
        """Add a custom typology.

        Args:
            typology: Typology definition to add.
        """
        self._typologies.append(typology)

    def detect(
        self,
        batch: TransactionBatch,
        *,
        party_id: str | None = None,
    ) -> list[TypologyMatch]:
        """Detect typologies in a transaction batch.

        Args:
            batch: Transaction batch to analyze.
            party_id: Optional filter by party.

        Returns:
            List of detected typology matches.
        """
        matches: list[TypologyMatch] = []

        transactions = list(batch)
        if party_id:
            transactions = batch.by_party(party_id)

        for typology in self._typologies:
            if not typology.enabled:
                continue

            detected = self._detect_typology(typology, transactions)
            if detected:
                matches.append(detected)

        return matches

    def detect_single(
        self,
        transaction: Transaction,
        history: list[Transaction],
    ) -> list[TypologyMatch]:
        """Detect typologies for a single transaction with history.

        Args:
            transaction: New transaction to analyze.
            history: Historical transactions for context.

        Returns:
            List of detected typology matches.
        """
        all_transactions = [*history, transaction]
        matches: list[TypologyMatch] = []

        for typology in self._typologies:
            if not typology.enabled:
                continue

            detected = self._detect_typology(typology, all_transactions)
            if detected and transaction.id in detected.transaction_ids:
                matches.append(detected)

        return matches

    def _detect_typology(
        self,
        typology: Typology,
        transactions: list[Transaction],
    ) -> TypologyMatch | None:
        """Detect a specific typology in transactions.

        Args:
            typology: Typology to detect.
            transactions: Transactions to analyze.

        Returns:
            TypologyMatch if detected, None otherwise.
        """
        if len(transactions) < typology.min_transactions:
            return None

        # Dispatch to specific detection methods
        if typology.typology_type == TypologyType.STRUCTURING:
            return self._detect_structuring(typology, transactions)
        if typology.typology_type == TypologyType.VELOCITY:
            return self._detect_velocity(typology, transactions)
        if typology.typology_type == TypologyType.ROUND_TRIPPING:
            return self._detect_round_tripping(typology, transactions)

        return None

    def _detect_structuring(
        self,
        typology: Typology,
        transactions: list[Transaction],
    ) -> TypologyMatch | None:
        """Detect structuring patterns.

        Structuring: Multiple cash transactions just below
        the reporting threshold (e.g., $10,000).
        """
        threshold = Decimal("10000")
        lower_bound = Decimal("8000")

        # Filter to cash transactions in suspicious range
        suspicious = [
            txn
            for txn in transactions
            if txn.transaction_type.value == "cash" and lower_bound <= txn.amount < threshold
        ]

        if len(suspicious) < typology.min_transactions:
            return None

        # Check time clustering
        suspicious.sort(key=lambda t: t.timestamp)

        for i in range(len(suspicious) - typology.min_transactions + 1):
            window = suspicious[i : i + typology.min_transactions]
            time_span = window[-1].timestamp - window[0].timestamp

            if time_span <= typology.time_window:
                total = sum(t.amount for t in window)
                party_ids = set()
                for t in window:
                    if t.sender:
                        party_ids.add(t.sender.id)
                    if t.sender_id:
                        party_ids.add(t.sender_id)

                return TypologyMatch(
                    typology_type=typology.typology_type,
                    typology_id=typology.id,
                    confidence=0.85,
                    transaction_ids=[t.id for t in window],
                    party_ids=list(party_ids),
                    amount_total=total,
                    time_span=time_span,
                    evidence={
                        "pattern": "structuring",
                        "transaction_count": len(window),
                        "average_amount": float(total / len(window)),
                    },
                )

        return None

    def _detect_velocity(
        self,
        typology: Typology,
        transactions: list[Transaction],
    ) -> TypologyMatch | None:
        """Detect high velocity patterns."""
        if len(transactions) < typology.min_transactions:
            return None

        # Group by sender
        by_sender: dict[str, list[Transaction]] = {}
        for txn in transactions:
            sender_id = txn.sender.id if txn.sender else txn.sender_id
            if sender_id:
                if sender_id not in by_sender:
                    by_sender[sender_id] = []
                by_sender[sender_id].append(txn)

        for sender_id, sender_txns in by_sender.items():
            if len(sender_txns) < typology.min_transactions:
                continue

            sender_txns.sort(key=lambda t: t.timestamp)
            time_span = sender_txns[-1].timestamp - sender_txns[0].timestamp

            if time_span <= typology.time_window:
                total = sum(t.amount for t in sender_txns)

                return TypologyMatch(
                    typology_type=typology.typology_type,
                    typology_id=typology.id,
                    confidence=0.75,
                    transaction_ids=[t.id for t in sender_txns],
                    party_ids=[sender_id],
                    amount_total=total,
                    time_span=time_span,
                    evidence={
                        "pattern": "velocity",
                        "transaction_count": len(sender_txns),
                        "time_span_hours": time_span.total_seconds() / 3600,
                    },
                )

        return None

    def _detect_round_tripping(
        self,
        typology: Typology,
        transactions: list[Transaction],
    ) -> TypologyMatch | None:
        """Detect round-tripping patterns.

        Round-tripping: Funds moving in a circle back to origin.
        """
        # Build transaction graph
        edges: list[tuple[str, str, Transaction]] = []
        for txn in transactions:
            sender = txn.sender.id if txn.sender else txn.sender_id
            receiver = txn.receiver.id if txn.receiver else txn.receiver_id
            if sender and receiver:
                edges.append((sender, receiver, txn))

        if len(edges) < typology.min_transactions:
            return None

        # Find cycles (simplified: look for A->B->...->A patterns)
        adjacency: dict[str, list[tuple[str, Transaction]]] = {}
        for sender, receiver, txn in edges:
            if sender not in adjacency:
                adjacency[sender] = []
            adjacency[sender].append((receiver, txn))

        # DFS for cycles
        for start_node in adjacency:
            cycle_txns = self._find_cycle(start_node, adjacency, typology.min_transactions)
            if cycle_txns:
                total = sum(t.amount for t in cycle_txns)
                party_ids = set()
                for t in cycle_txns:
                    if t.sender:
                        party_ids.add(t.sender.id)
                    if t.receiver:
                        party_ids.add(t.receiver.id)

                return TypologyMatch(
                    typology_type=typology.typology_type,
                    typology_id=typology.id,
                    confidence=0.9,
                    transaction_ids=[t.id for t in cycle_txns],
                    party_ids=list(party_ids),
                    amount_total=total,
                    evidence={
                        "pattern": "round_tripping",
                        "cycle_length": len(cycle_txns),
                    },
                )

        return None

    def _find_cycle(
        self,
        start: str,
        adjacency: dict[str, list[tuple[str, Transaction]]],
        min_length: int,
    ) -> list[Transaction] | None:
        """Find a cycle starting from a node."""
        visited: set[str] = set()
        path: list[Transaction] = []

        def dfs(node: str) -> bool:
            if node in visited:
                return node == start and len(path) >= min_length

            visited.add(node)
            for neighbor, txn in adjacency.get(node, []):
                path.append(txn)
                if dfs(neighbor):
                    return True
                path.pop()
            visited.remove(node)
            return False

        for neighbor, txn in adjacency.get(start, []):
            path = [txn]
            visited = {start}
            if dfs(neighbor):
                return path

        return None
