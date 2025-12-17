"""Core module providing configuration, errors, logging, and schema management."""

from ununseptium.core.canonical import canonical_json, deterministic_hash
from ununseptium.core.config import Settings, load_config
from ununseptium.core.errors import (
    ConfigurationError,
    IntegrityError,
    ModelError,
    SecurityError,
    UnunseptiumError,
    ValidationError,
)
from ununseptium.core.logging import get_logger, setup_logging
from ununseptium.core.schemas import SchemaRegistry, export_schema, validate_data

__all__ = [
    # Configuration
    "Settings",
    "load_config",
    # Errors
    "ConfigurationError",
    "IntegrityError",
    "ModelError",
    "SecurityError",
    "UnunseptiumError",
    "ValidationError",
    # Logging
    "get_logger",
    "setup_logging",
    # Schemas
    "SchemaRegistry",
    "export_schema",
    "validate_data",
    # Canonical
    "canonical_json",
    "deterministic_hash",
]
