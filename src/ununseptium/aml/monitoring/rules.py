"""Rule-based monitoring for AML.

Provides configurable rule engine for transaction monitoring.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ununseptium.aml.transactions import Transaction


class RuleOperator(str, Enum):
    """Operators for rule conditions."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    MATCHES = "matches"


class RuleCondition(BaseModel):
    """A condition in a rule.

    Attributes:
        field: Transaction field to evaluate.
        operator: Comparison operator.
        value: Value to compare against.
    """

    field: str
    operator: RuleOperator
    value: Any


class RuleSeverity(str, Enum):
    """Severity of rule match."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Rule(BaseModel):
    """A monitoring rule.

    Attributes:
        id: Rule identifier.
        name: Rule name.
        description: Rule description.
        conditions: Rule conditions (AND).
        severity: Match severity.
        score: Rule score (0.0 to 1.0).
        enabled: Whether rule is active.
        tags: Rule tags.
        metadata: Additional data.
    """

    id: str = Field(default_factory=lambda: f"RULE-{uuid4().hex[:8].upper()}")
    name: str
    description: str = ""
    conditions: list[RuleCondition] = Field(default_factory=list)
    severity: RuleSeverity = RuleSeverity.MEDIUM
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuleResult(BaseModel):
    """Result of rule evaluation.

    Attributes:
        rule_id: ID of the matched rule.
        rule_name: Name of the rule.
        matched: Whether rule matched.
        score: Match score.
        severity: Match severity.
        transaction_id: Evaluated transaction ID.
        matched_conditions: Conditions that matched.
        evaluated_at: Evaluation timestamp.
    """

    rule_id: str
    rule_name: str
    matched: bool
    score: float = 0.0
    severity: RuleSeverity = RuleSeverity.MEDIUM
    transaction_id: str
    matched_conditions: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class RuleEngine:
    """Rule-based transaction monitoring engine.

    Evaluates transactions against configurable rules.

    Example:
        ```python
        from ununseptium.aml.monitoring import RuleEngine, Rule, RuleCondition

        engine = RuleEngine()

        # Add a high-value transaction rule
        rule = Rule(
            name="High Value Transaction",
            conditions=[
                RuleCondition(field="amount", operator="gte", value=10000)
            ],
            severity="high"
        )
        engine.add_rule(rule)

        # Evaluate transaction
        results = engine.evaluate(transaction)
        ```
    """

    def __init__(self) -> None:
        """Initialize the rule engine."""
        self._rules: dict[str, Rule] = {}
        self._custom_evaluators: dict[str, Callable[[Transaction], bool]] = {}

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine.

        Args:
            rule: Rule to add.
        """
        self._rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine.

        Args:
            rule_id: ID of rule to remove.

        Returns:
            True if removed, False if not found.
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def add_custom_evaluator(
        self,
        rule_id: str,
        evaluator: Callable[[Transaction], bool],
    ) -> None:
        """Add a custom evaluator function for a rule.

        Args:
            rule_id: Rule ID to attach evaluator to.
            evaluator: Function that returns True if rule matches.
        """
        self._custom_evaluators[rule_id] = evaluator

    def evaluate(self, transaction: Transaction) -> list[RuleResult]:
        """Evaluate a transaction against all rules.

        Args:
            transaction: Transaction to evaluate.

        Returns:
            List of RuleResults for matching rules.
        """
        results: list[RuleResult] = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            result = self._evaluate_rule(rule, transaction)
            if result.matched:
                results.append(result)

        return results

    def evaluate_batch(
        self,
        transactions: list[Transaction],
    ) -> dict[str, list[RuleResult]]:
        """Evaluate multiple transactions.

        Args:
            transactions: Transactions to evaluate.

        Returns:
            Dictionary mapping transaction ID to results.
        """
        return {txn.id: self.evaluate(txn) for txn in transactions}

    def _evaluate_rule(self, rule: Rule, transaction: Transaction) -> RuleResult:
        """Evaluate a single rule against a transaction."""
        matched_conditions: list[str] = []

        # Check custom evaluator first
        if rule.id in self._custom_evaluators:
            if self._custom_evaluators[rule.id](transaction):
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    matched=True,
                    score=rule.score,
                    severity=rule.severity,
                    transaction_id=transaction.id,
                    matched_conditions=["custom_evaluator"],
                )

        # Evaluate standard conditions (AND logic)
        all_matched = len(rule.conditions) > 0

        for condition in rule.conditions:
            if self._evaluate_condition(condition, transaction):
                matched_conditions.append(condition.field)
            else:
                all_matched = False

        return RuleResult(
            rule_id=rule.id,
            rule_name=rule.name,
            matched=all_matched,
            score=rule.score if all_matched else 0.0,
            severity=rule.severity,
            transaction_id=transaction.id,
            matched_conditions=matched_conditions,
        )

    def _evaluate_condition(
        self,
        condition: RuleCondition,
        transaction: Transaction,
    ) -> bool:
        """Evaluate a single condition."""
        # Get field value (supports nested fields with dot notation)
        value = self._get_field_value(transaction, condition.field)

        if value is None:
            return False

        target = condition.value

        # Handle Decimal comparisons
        if isinstance(value, Decimal):
            target = Decimal(str(target))

        op = condition.operator

        if op == RuleOperator.EQUALS:
            return value == target
        if op == RuleOperator.NOT_EQUALS:
            return value != target
        if op == RuleOperator.GREATER_THAN:
            return value > target
        if op == RuleOperator.GREATER_THAN_OR_EQUAL:
            return value >= target
        if op == RuleOperator.LESS_THAN:
            return value < target
        if op == RuleOperator.LESS_THAN_OR_EQUAL:
            return value <= target
        if op == RuleOperator.IN:
            return value in target
        if op == RuleOperator.NOT_IN:
            return value not in target
        if op == RuleOperator.CONTAINS:
            return target in str(value)
        if op == RuleOperator.MATCHES:
            import re

            return bool(re.match(target, str(value)))

        return False

    def _get_field_value(self, transaction: Transaction, field: str) -> Any:
        """Get a field value from a transaction."""
        parts = field.split(".")
        obj: Any = transaction

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None

        return obj

    def list_rules(
        self,
        *,
        enabled_only: bool = False,
        tags: list[str] | None = None,
    ) -> list[Rule]:
        """List rules with optional filtering.

        Args:
            enabled_only: Only return enabled rules.
            tags: Filter by tags.

        Returns:
            List of matching rules.
        """
        rules = list(self._rules.values())

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        if tags:
            rules = [r for r in rules if any(t in r.tags for t in tags)]

        return rules
