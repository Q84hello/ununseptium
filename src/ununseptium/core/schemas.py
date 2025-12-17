"""JSON Schema registry and validation.

Provides centralized schema management for all data models,
with support for schema export, validation, and versioning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from ununseptium.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", bound=BaseModel)


class SchemaRegistry:
    """Registry for JSON schemas.

    Manages schemas for all data models, providing centralized
    access for validation, export, and documentation.

    Attributes:
        schemas: Dictionary of registered schemas by name.

    Example:
        ```python
        from ununseptium.core import SchemaRegistry
        from pydantic import BaseModel

        class Transaction(BaseModel):
            id: str
            amount: float

        registry = SchemaRegistry()
        registry.register("transaction", Transaction)

        schema = registry.get("transaction")
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty schema registry."""
        self._schemas: dict[str, dict[str, Any]] = {}
        self._models: dict[str, type[BaseModel]] = {}
        self._validators: dict[str, Callable[[Any], None]] = {}

    def register(
        self,
        name: str,
        model: type[BaseModel],
        *,
        version: str = "1.0.0",
    ) -> None:
        """Register a Pydantic model in the registry.

        Args:
            name: Schema name (unique identifier).
            model: Pydantic model class.
            version: Schema version string.

        Example:
            ```python
            from pydantic import BaseModel
            from ununseptium.core import SchemaRegistry

            class Identity(BaseModel):
                id: str
                name: str

            registry = SchemaRegistry()
            registry.register("identity", Identity, version="1.0.0")
            ```
        """
        schema = model.model_json_schema()
        schema["$id"] = f"urn:ununseptium:{name}:{version}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

        self._schemas[name] = schema
        self._models[name] = model

    def register_custom(
        self,
        name: str,
        schema: dict[str, Any],
        validator: Callable[[Any], None] | None = None,
    ) -> None:
        """Register a custom JSON schema.

        Args:
            name: Schema name (unique identifier).
            schema: JSON Schema dictionary.
            validator: Optional custom validation function.

        Example:
            ```python
            from ununseptium.core import SchemaRegistry

            custom_schema = {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"]
            }

            registry = SchemaRegistry()
            registry.register_custom("custom", custom_schema)
            ```
        """
        self._schemas[name] = schema
        if validator is not None:
            self._validators[name] = validator

    def get(self, name: str) -> dict[str, Any]:
        """Get a schema by name.

        Args:
            name: Schema name.

        Returns:
            JSON Schema dictionary.

        Raises:
            ValidationError: If schema is not found.
        """
        if name not in self._schemas:
            msg = f"Schema not found: {name}"
            raise ValidationError(msg, field="name", value=name)
        return self._schemas[name]

    def get_model(self, name: str) -> type[BaseModel]:
        """Get a Pydantic model by schema name.

        Args:
            name: Schema name.

        Returns:
            Pydantic model class.

        Raises:
            ValidationError: If model is not found.
        """
        if name not in self._models:
            msg = f"Model not found for schema: {name}"
            raise ValidationError(msg, field="name", value=name)
        return self._models[name]

    def validate(self, name: str, data: Any) -> T:
        """Validate data against a schema.

        Args:
            name: Schema name.
            data: Data to validate.

        Returns:
            Validated and parsed model instance.

        Raises:
            ValidationError: If validation fails.

        Example:
            ```python
            from ununseptium.core import SchemaRegistry

            registry = SchemaRegistry()
            # ... register schemas ...

            data = {"id": "123", "name": "John"}
            identity = registry.validate("identity", data)
            ```
        """
        if name in self._validators:
            try:
                self._validators[name](data)
            except Exception as e:
                msg = f"Validation failed for schema {name}: {e}"
                raise ValidationError(msg, field=name) from e

        if name in self._models:
            try:
                model = self._models[name]
                return model.model_validate(data)  # type: ignore[return-value]
            except Exception as e:
                msg = f"Model validation failed for schema {name}: {e}"
                raise ValidationError(msg, field=name) from e

        # For custom schemas without models, just return the data
        return data  # type: ignore[return-value]

    def list_schemas(self) -> list[str]:
        """List all registered schema names.

        Returns:
            List of schema names.
        """
        return list(self._schemas.keys())

    def export(
        self,
        output_dir: Path | str,
        *,
        format: str = "json",  # noqa: A002
        indent: int = 2,
    ) -> list[Path]:
        """Export all schemas to files.

        Args:
            output_dir: Directory to write schema files.
            format: Output format (json or yaml).
            indent: Indentation for output files.

        Returns:
            List of created file paths.

        Example:
            ```python
            from ununseptium.core import SchemaRegistry

            registry = SchemaRegistry()
            # ... register schemas ...

            files = registry.export("schemas/", format="json")
            ```
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files: list[Path] = []

        for name, schema in self._schemas.items():
            if format == "json":
                file_path = output_dir / f"{name}.schema.json"
                with file_path.open("w") as f:
                    json.dump(schema, f, indent=indent)
            elif format == "yaml":
                try:
                    import yaml
                except ImportError as err:
                    msg = "PyYAML is required for YAML export"
                    raise ValidationError(msg) from err

                file_path = output_dir / f"{name}.schema.yaml"
                with file_path.open("w") as f:
                    yaml.safe_dump(schema, f, default_flow_style=False)
            else:
                msg = f"Unsupported format: {format}"
                raise ValidationError(msg, field="format", value=format)

            created_files.append(file_path)

        return created_files


# Global registry instance
_global_registry: SchemaRegistry | None = None


def get_global_registry() -> SchemaRegistry:
    """Get the global schema registry.

    Returns:
        Global SchemaRegistry instance.
    """
    global _global_registry  # noqa: PLW0603
    if _global_registry is None:
        _global_registry = SchemaRegistry()
    return _global_registry


def validate_data(schema_name: str, data: Any) -> Any:
    """Validate data against a schema in the global registry.

    Args:
        schema_name: Name of the schema.
        data: Data to validate.

    Returns:
        Validated data or model instance.

    Raises:
        ValidationError: If validation fails.
    """
    return get_global_registry().validate(schema_name, data)


def export_schema(
    output_dir: Path | str,
    *,
    format: str = "json",  # noqa: A002
) -> list[Path]:
    """Export all schemas from the global registry.

    Args:
        output_dir: Directory to write schema files.
        format: Output format (json or yaml).

    Returns:
        List of created file paths.
    """
    return get_global_registry().export(output_dir, format=format)


def register_schema(
    name: str,
    model: type[BaseModel],
    *,
    version: str = "1.0.0",
) -> type[BaseModel]:
    """Decorator to register a model in the global registry.

    Args:
        name: Schema name.
        model: Pydantic model class.
        version: Schema version.

    Returns:
        The model class (for use as decorator).

    Example:
        ```python
        from pydantic import BaseModel
        from ununseptium.core.schemas import register_schema

        @register_schema("transaction", version="1.0.0")
        class Transaction(BaseModel):
            id: str
            amount: float
        ```
    """
    get_global_registry().register(name, model, version=version)
    return model
