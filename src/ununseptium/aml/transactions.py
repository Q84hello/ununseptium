"""Transaction models and processing for AML.

Provides transaction data models, parsing, and batch processing
capabilities for anti-money laundering analysis.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Iterator
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class TransactionType(str, Enum):
    """Types of financial transactions."""

    CREDIT = "credit"
    DEBIT = "debit"
    TRANSFER = "transfer"
    WIRE = "wire"
    ACH = "ach"
    CHECK = "check"
    CASH = "cash"
    CARD = "card"
    CRYPTO = "crypto"
    OTHER = "other"


class TransactionStatus(str, Enum):
    """Status of a transaction."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERSED = "reversed"
    FLAGGED = "flagged"


class Party(BaseModel):
    """A party to a transaction.

    Attributes:
        id: Party identifier.
        name: Party name.
        account_number: Account number.
        bank_code: Bank/routing code.
        country: Country code.
        party_type: Type of party (individual, business).
    """

    id: str
    name: str | None = None
    account_number: str | None = None
    bank_code: str | None = None
    country: str | None = None
    party_type: str = "individual"


class Transaction(BaseModel):
    """Financial transaction model.

    Represents a single financial transaction for AML analysis.

    Attributes:
        id: Unique transaction identifier.
        transaction_type: Type of transaction.
        amount: Transaction amount.
        currency: ISO 4217 currency code.
        sender: Sending party.
        receiver: Receiving party.
        timestamp: Transaction timestamp.
        status: Transaction status.
        reference: External reference.
        description: Transaction description.
        channel: Transaction channel.
        location: Geographic location.
        risk_score: Computed risk score.
        flags: Applied flags.
        metadata: Additional data.
    """

    id: str = Field(default_factory=lambda: f"TXN-{uuid4().hex[:12].upper()}")
    transaction_type: TransactionType = TransactionType.TRANSFER
    amount: Decimal = Field(ge=Decimal("0"))
    currency: str = Field(default="USD", min_length=3, max_length=3)
    sender: Party | None = None
    sender_id: str | None = None
    receiver: Party | None = None
    receiver_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: TransactionStatus = TransactionStatus.COMPLETED
    reference: str | None = None
    description: str | None = None
    channel: str | None = None
    location: str | None = None
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code is uppercase."""
        return v.upper()

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v: Any) -> Decimal:
        """Coerce amount to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    def is_high_value(self, threshold: Decimal = Decimal("10000")) -> bool:
        """Check if transaction is high value.

        Args:
            threshold: Amount threshold.

        Returns:
            True if amount exceeds threshold.
        """
        return self.amount >= threshold

    def is_cross_border(self) -> bool:
        """Check if transaction is cross-border.

        Returns:
            True if sender and receiver are in different countries.
        """
        sender_country = self.sender.country if self.sender else None
        receiver_country = self.receiver.country if self.receiver else None

        if sender_country and receiver_country:
            return sender_country != receiver_country
        return False


class TransactionBatch(BaseModel):
    """A batch of transactions for processing.

    Attributes:
        id: Batch identifier.
        transactions: List of transactions.
        created_at: Batch creation time.
        source: Data source.
        metadata: Additional batch data.
    """

    id: str = Field(default_factory=lambda: f"BATCH-{uuid4().hex[:8].upper()}")
    transactions: list[Transaction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of transactions."""
        return len(self.transactions)

    def __iter__(self) -> Iterator[Transaction]:
        """Iterate over transactions."""
        return iter(self.transactions)

    def add(self, transaction: Transaction) -> None:
        """Add a transaction to the batch.

        Args:
            transaction: Transaction to add.
        """
        self.transactions.append(transaction)

    def total_amount(self, currency: str | None = None) -> Decimal:
        """Calculate total amount in batch.

        Args:
            currency: Filter by currency.

        Returns:
            Total transaction amount.
        """
        total = Decimal("0")
        for txn in self.transactions:
            if currency is None or txn.currency == currency:
                total += txn.amount
        return total

    def by_party(self, party_id: str) -> list[Transaction]:
        """Get transactions involving a party.

        Args:
            party_id: Party identifier.

        Returns:
            List of matching transactions.
        """
        return [
            txn
            for txn in self.transactions
            if (txn.sender and txn.sender.id == party_id)
            or (txn.receiver and txn.receiver.id == party_id)
            or txn.sender_id == party_id
            or txn.receiver_id == party_id
        ]

    def date_range(self) -> tuple[datetime | None, datetime | None]:
        """Get date range of transactions.

        Returns:
            Tuple of (earliest, latest) timestamps.
        """
        if not self.transactions:
            return None, None

        timestamps = [txn.timestamp for txn in self.transactions]
        return min(timestamps), max(timestamps)


class TransactionParser:
    """Parse transactions from various formats.

    Provides parsing capabilities for transaction data from
    CSV, JSON, and other common formats.

    Example:
        ```python
        from ununseptium.aml import TransactionParser

        parser = TransactionParser()

        # Parse from dict
        txn = parser.from_dict({
            "id": "TXN-001",
            "amount": 1000.00,
            "currency": "USD",
            "sender_id": "A001",
            "receiver_id": "A002"
        })

        # Parse batch from list
        batch = parser.parse_batch([...])
        ```
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._field_mapping: dict[str, str] = {
            "transaction_id": "id",
            "txn_id": "id",
            "amt": "amount",
            "ccy": "currency",
            "from_id": "sender_id",
            "to_id": "receiver_id",
            "txn_type": "transaction_type",
            "tx_type": "transaction_type",
            "date": "timestamp",
            "datetime": "timestamp",
        }

    def from_dict(self, data: dict[str, Any]) -> Transaction:
        """Parse a transaction from a dictionary.

        Args:
            data: Dictionary with transaction data.

        Returns:
            Parsed Transaction instance.
        """
        # Apply field mapping
        normalized = {}
        for key, value in data.items():
            mapped_key = self._field_mapping.get(key, key)
            normalized[mapped_key] = value

        # Handle party objects
        if "sender_id" in normalized and "sender" not in normalized:
            normalized["sender"] = Party(id=normalized["sender_id"])
        if "receiver_id" in normalized and "receiver" not in normalized:
            normalized["receiver"] = Party(id=normalized["receiver_id"])

        return Transaction.model_validate(normalized)

    def parse_batch(
        self,
        data: list[dict[str, Any]],
        *,
        source: str | None = None,
    ) -> TransactionBatch:
        """Parse a batch of transactions.

        Args:
            data: List of transaction dictionaries.
            source: Data source identifier.

        Returns:
            TransactionBatch with parsed transactions.
        """
        transactions = [self.from_dict(item) for item in data]
        return TransactionBatch(transactions=transactions, source=source)

    def add_field_mapping(self, source_field: str, target_field: str) -> None:
        """Add a custom field mapping.

        Args:
            source_field: Source field name in input data.
            target_field: Target field name in Transaction model.
        """
        self._field_mapping[source_field] = target_field
