"""Data integrity verification.

Provides integrity checking and consistency validation
for data and configurations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ununseptium.core.canonical import deterministic_hash, verify_hash


class IntegrityRecord(BaseModel):
    """Record of integrity check.

    Attributes:
        resource_id: Resource identifier.
        resource_type: Type of resource.
        hash_value: Computed hash.
        algorithm: Hash algorithm.
        checked_at: Check timestamp.
        is_valid: Whether check passed.
        previous_hash: Previous known hash.
    """

    resource_id: str
    resource_type: str
    hash_value: str
    algorithm: str = "sha256"
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    is_valid: bool = True
    previous_hash: str | None = None


class IntegrityChecker:
    """Check data integrity.

    Computes and verifies hashes for data integrity.

    Example:
        ```python
        from ununseptium.security import IntegrityChecker

        checker = IntegrityChecker()

        # Compute hash for data
        data = {"key": "value", "list": [1, 2, 3]}
        record = checker.compute("resource-1", data)

        # Later: verify integrity
        is_valid = checker.verify("resource-1", data)
        ```
    """

    def __init__(self, algorithm: str = "sha256") -> None:
        """Initialize the checker.

        Args:
            algorithm: Hash algorithm.
        """
        self.algorithm = algorithm
        self._records: dict[str, IntegrityRecord] = {}

    def compute(
        self,
        resource_id: str,
        data: Any,
        *,
        resource_type: str = "data",
    ) -> IntegrityRecord:
        """Compute and store hash for data.

        Args:
            resource_id: Resource identifier.
            data: Data to hash.
            resource_type: Type of resource.

        Returns:
            IntegrityRecord with hash.
        """
        hash_value = deterministic_hash(data, algorithm=self.algorithm)

        previous = self._records.get(resource_id)
        previous_hash = previous.hash_value if previous else None

        record = IntegrityRecord(
            resource_id=resource_id,
            resource_type=resource_type,
            hash_value=hash_value,
            algorithm=self.algorithm,
            previous_hash=previous_hash,
        )

        self._records[resource_id] = record
        return record

    def verify(self, resource_id: str, data: Any) -> bool:
        """Verify data against stored hash.

        Args:
            resource_id: Resource identifier.
            data: Data to verify.

        Returns:
            True if hash matches.
        """
        record = self._records.get(resource_id)
        if not record:
            return False

        return verify_hash(data, record.hash_value, algorithm=self.algorithm)

    def verify_file(self, resource_id: str, path: Path | str) -> bool:
        """Verify file contents against stored hash.

        Args:
            resource_id: Resource identifier.
            path: File path.

        Returns:
            True if hash matches.
        """
        path = Path(path)
        if not path.exists():
            return False

        content = path.read_bytes()
        record = self._records.get(resource_id)

        if not record:
            return False

        import hashlib

        hasher = hashlib.new(self.algorithm)
        hasher.update(content)
        actual = hasher.hexdigest()

        stored = record.hash_value
        if ":" in stored:
            stored = stored.split(":", 1)[1]

        import hmac

        return hmac.compare_digest(actual, stored)

    def compute_file(
        self,
        resource_id: str,
        path: Path | str,
    ) -> IntegrityRecord:
        """Compute hash for file.

        Args:
            resource_id: Resource identifier.
            path: File path.

        Returns:
            IntegrityRecord with hash.
        """
        path = Path(path)
        content = path.read_bytes()

        import hashlib

        hasher = hashlib.new(self.algorithm)
        hasher.update(content)
        hash_value = f"{self.algorithm}:{hasher.hexdigest()}"

        previous = self._records.get(resource_id)
        previous_hash = previous.hash_value if previous else None

        record = IntegrityRecord(
            resource_id=resource_id,
            resource_type="file",
            hash_value=hash_value,
            algorithm=self.algorithm,
            previous_hash=previous_hash,
        )

        self._records[resource_id] = record
        return record

    def get_record(self, resource_id: str) -> IntegrityRecord | None:
        """Get integrity record.

        Args:
            resource_id: Resource identifier.

        Returns:
            Record if found.
        """
        return self._records.get(resource_id)


class ConsistencyRule(BaseModel):
    """A consistency validation rule.

    Attributes:
        name: Rule name.
        description: Rule description.
        check_type: Type of check.
        parameters: Rule parameters.
    """

    name: str
    description: str = ""
    check_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ConsistencyResult(BaseModel):
    """Result of consistency validation.

    Attributes:
        is_consistent: Whether data is consistent.
        rule_name: Rule that was checked.
        message: Result message.
        details: Detailed findings.
        checked_at: Check timestamp.
    """

    is_consistent: bool
    rule_name: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class ConsistencyValidator:
    """Validate data consistency.

    Checks data against consistency rules.

    Example:
        ```python
        from ununseptium.security import ConsistencyValidator

        validator = ConsistencyValidator()

        # Add rule
        validator.add_rule(ConsistencyRule(
            name="amount_positive",
            check_type="range",
            parameters={"field": "amount", "min": 0}
        ))

        # Validate
        results = validator.validate({"amount": 100})
        ```
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self._rules: list[ConsistencyRule] = []

    def add_rule(self, rule: ConsistencyRule) -> None:
        """Add a consistency rule.

        Args:
            rule: Rule to add.
        """
        self._rules.append(rule)

    def validate(self, data: dict[str, Any]) -> list[ConsistencyResult]:
        """Validate data against all rules.

        Args:
            data: Data to validate.

        Returns:
            List of ConsistencyResults.
        """
        results: list[ConsistencyResult] = []

        for rule in self._rules:
            result = self._check_rule(rule, data)
            results.append(result)

        return results

    def is_consistent(self, data: dict[str, Any]) -> bool:
        """Check if data passes all rules.

        Args:
            data: Data to check.

        Returns:
            True if all rules pass.
        """
        results = self.validate(data)
        return all(r.is_consistent for r in results)

    def _check_rule(
        self,
        rule: ConsistencyRule,
        data: dict[str, Any],
    ) -> ConsistencyResult:
        """Check a single rule."""
        check_type = rule.check_type
        params = rule.parameters

        if check_type == "range":
            return self._check_range(rule, data, params)
        if check_type == "required":
            return self._check_required(rule, data, params)
        if check_type == "format":
            return self._check_format(rule, data, params)

        return ConsistencyResult(
            is_consistent=True,
            rule_name=rule.name,
            message="Unknown check type, skipped",
        )

    def _check_range(
        self,
        rule: ConsistencyRule,
        data: dict[str, Any],
        params: dict[str, Any],
    ) -> ConsistencyResult:
        """Check range constraint."""
        field = params.get("field")
        min_val = params.get("min")
        max_val = params.get("max")

        value = data.get(field)

        if value is None:
            return ConsistencyResult(
                is_consistent=False,
                rule_name=rule.name,
                message=f"Field '{field}' not found",
            )

        if min_val is not None and value < min_val:
            return ConsistencyResult(
                is_consistent=False,
                rule_name=rule.name,
                message=f"Value {value} below minimum {min_val}",
                details={"field": field, "value": value, "min": min_val},
            )

        if max_val is not None and value > max_val:
            return ConsistencyResult(
                is_consistent=False,
                rule_name=rule.name,
                message=f"Value {value} above maximum {max_val}",
                details={"field": field, "value": value, "max": max_val},
            )

        return ConsistencyResult(
            is_consistent=True,
            rule_name=rule.name,
            message="Range check passed",
        )

    def _check_required(
        self,
        rule: ConsistencyRule,
        data: dict[str, Any],
        params: dict[str, Any],
    ) -> ConsistencyResult:
        """Check required fields."""
        fields = params.get("fields", [])
        missing = [f for f in fields if f not in data or data[f] is None]

        if missing:
            return ConsistencyResult(
                is_consistent=False,
                rule_name=rule.name,
                message=f"Missing required fields: {missing}",
                details={"missing": missing},
            )

        return ConsistencyResult(
            is_consistent=True,
            rule_name=rule.name,
            message="Required fields present",
        )

    def _check_format(
        self,
        rule: ConsistencyRule,
        data: dict[str, Any],
        params: dict[str, Any],
    ) -> ConsistencyResult:
        """Check field format."""
        import re

        field = params.get("field")
        pattern = params.get("pattern")

        value = data.get(field)

        if value is None:
            return ConsistencyResult(
                is_consistent=False,
                rule_name=rule.name,
                message=f"Field '{field}' not found",
            )

        if pattern and not re.match(pattern, str(value)):
            return ConsistencyResult(
                is_consistent=False,
                rule_name=rule.name,
                message=f"Field '{field}' does not match pattern",
                details={"field": field, "pattern": pattern},
            )

        return ConsistencyResult(
            is_consistent=True,
            rule_name=rule.name,
            message="Format check passed",
        )
