"""Canonical JSON serialization and deterministic hashing.

Provides consistent, reproducible JSON output for cryptographic operations
and audit trail integrity. All JSON is sorted by keys and uses consistent
formatting to ensure deterministic hashing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


class CanonicalJSONEncoder(json.JSONEncoder):
    """JSON encoder that produces canonical, deterministic output.

    Handles common Python types and ensures consistent serialization:
    - datetime/date: ISO 8601 format
    - Decimal: String representation
    - UUID: String representation
    - Enum: Value
    - Path: String
    - bytes: Base64 encoded
    - set/frozenset: Sorted list
    """

    def default(self, obj: Any) -> Any:
        """Encode non-standard types.

        Args:
            obj: Object to encode.

        Returns:
            JSON-serializable representation.

        Raises:
            TypeError: If object cannot be serialized.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, bytes):
            import base64

            return base64.b64encode(obj).decode("ascii")
        if isinstance(obj, (set, frozenset)):
            return sorted(obj, key=str)
        if hasattr(obj, "model_dump"):
            # Pydantic v2 models
            return obj.model_dump(mode="json")
        if hasattr(obj, "dict"):
            # Pydantic v1 models
            return obj.dict()
        return super().default(obj)


def canonical_json(
    data: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = False,
) -> str:
    """Serialize data to canonical JSON.

    Produces deterministic JSON output suitable for hashing:
    - Keys are sorted alphabetically
    - No trailing whitespace
    - Consistent number formatting
    - UTF-8 encoding by default

    Args:
        data: Data to serialize.
        indent: Number of spaces for indentation. None for compact output.
        ensure_ascii: If True, escape non-ASCII characters.

    Returns:
        Canonical JSON string.

    Example:
        ```python
        from ununseptium.core import canonical_json

        data = {"z": 1, "a": 2, "m": [3, 2, 1]}
        json_str = canonical_json(data)
        # Output: {"a":2,"m":[3,2,1],"z":1}
        ```
    """
    return json.dumps(
        data,
        cls=CanonicalJSONEncoder,
        sort_keys=True,
        separators=(",", ":") if indent is None else (",", ": "),
        indent=indent,
        ensure_ascii=ensure_ascii,
    )


def deterministic_hash(
    data: Any,
    *,
    algorithm: str = "sha256",
    prefix: bool = True,
) -> str:
    """Compute a deterministic hash of data.

    Uses canonical JSON serialization to ensure consistent hashing
    regardless of key ordering or whitespace.

    Args:
        data: Data to hash.
        algorithm: Hash algorithm (sha256, sha384, sha512).
        prefix: If True, prefix hash with algorithm name.

    Returns:
        Hexadecimal hash string, optionally prefixed with algorithm.

    Example:
        ```python
        from ununseptium.core import deterministic_hash

        data = {"name": "John", "age": 30}
        hash_value = deterministic_hash(data)
        # Output: sha256:a1b2c3...
        ```
    """
    json_bytes = canonical_json(data).encode("utf-8")
    hasher = hashlib.new(algorithm)
    hasher.update(json_bytes)
    hex_digest = hasher.hexdigest()

    if prefix:
        return f"{algorithm}:{hex_digest}"
    return hex_digest


def verify_hash(
    data: Any,
    expected_hash: str,
    *,
    algorithm: str | None = None,
) -> bool:
    """Verify data against an expected hash.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        data: Data to verify.
        expected_hash: Expected hash value.
        algorithm: Hash algorithm. If None, extracted from prefix.

    Returns:
        True if hash matches, False otherwise.

    Example:
        ```python
        from ununseptium.core.canonical import deterministic_hash, verify_hash

        data = {"name": "John"}
        hash_value = deterministic_hash(data)
        assert verify_hash(data, hash_value)  # True
        ```
    """
    import hmac

    # Extract algorithm from prefix if present
    if algorithm is None:
        if ":" in expected_hash:
            algorithm, expected_hash = expected_hash.split(":", 1)
        else:
            algorithm = "sha256"

    actual_hash = deterministic_hash(data, algorithm=algorithm, prefix=False)

    # Constant-time comparison
    return hmac.compare_digest(actual_hash, expected_hash)


def canonical_hash_chain(
    items: list[Any],
    *,
    algorithm: str = "sha256",
) -> list[str]:
    """Compute a hash chain over a list of items.

    Each hash includes the previous hash, creating a tamper-evident chain.

    Args:
        items: List of items to hash.
        algorithm: Hash algorithm.

    Returns:
        List of hash values forming the chain.

    Example:
        ```python
        from ununseptium.core.canonical import canonical_hash_chain

        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        chain = canonical_hash_chain(items)
        # Each hash depends on the previous, detecting any tampering
        ```
    """
    chain: list[str] = []
    prev_hash = ""

    for item in items:
        # Include previous hash in the data
        chain_data = {
            "prev_hash": prev_hash,
            "data": item,
        }
        current_hash = deterministic_hash(chain_data, algorithm=algorithm)
        chain.append(current_hash)
        prev_hash = current_hash

    return chain


def verify_hash_chain(
    items: list[Any],
    chain: list[str],
    *,
    algorithm: str | None = None,
) -> tuple[bool, int | None]:
    """Verify a hash chain for integrity.

    Args:
        items: List of items that were hashed.
        chain: List of hash values to verify.
        algorithm: Hash algorithm. If None, extracted from first hash.

    Returns:
        Tuple of (is_valid, first_invalid_index).
        If valid, returns (True, None).
        If invalid, returns (False, index of first mismatch).

    Example:
        ```python
        from ununseptium.core.canonical import canonical_hash_chain, verify_hash_chain

        items = [{"id": 1}, {"id": 2}]
        chain = canonical_hash_chain(items)
        is_valid, index = verify_hash_chain(items, chain)
        assert is_valid and index is None
        ```
    """
    if len(items) != len(chain):
        return False, min(len(items), len(chain))

    if not chain:
        return True, None

    # Extract algorithm from first hash if not provided
    if algorithm is None:
        if ":" in chain[0]:
            algorithm = chain[0].split(":", 1)[0]
        else:
            algorithm = "sha256"

    prev_hash = ""

    for i, (item, expected_hash) in enumerate(zip(items, chain, strict=True)):
        chain_data = {
            "prev_hash": prev_hash,
            "data": item,
        }
        actual_hash = deterministic_hash(chain_data, algorithm=algorithm)

        # Normalize expected hash (remove prefix if present)
        expected_normalized = expected_hash
        if ":" in expected_hash:
            expected_normalized = expected_hash.split(":", 1)[1]

        actual_normalized = actual_hash.split(":", 1)[1] if ":" in actual_hash else actual_hash

        import hmac

        if not hmac.compare_digest(actual_normalized, expected_normalized):
            return False, i

        prev_hash = actual_hash

    return True, None
