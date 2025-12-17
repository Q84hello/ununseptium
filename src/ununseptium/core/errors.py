"""Exception hierarchy for ununseptium.

All exceptions inherit from UnunseptiumError, enabling unified exception handling
while allowing granular catching of specific error types.
"""

from __future__ import annotations

from typing import Any


class UnunseptiumError(Exception):
    """Base exception for all ununseptium errors.

    Attributes:
        message: Human-readable error description.
        code: Machine-readable error code for programmatic handling.
        details: Additional context about the error.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            code: Machine-readable error code.
            details: Additional error context.
        """
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__.upper()
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization.

        Returns:
            Dictionary with error code, message, and details.
        """
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(UnunseptiumError):
    """Raised when data validation fails.

    This includes schema validation errors, constraint violations,
    and invalid input data.

    Attributes:
        field: Name of the field that failed validation.
        value: The invalid value that was provided.
        constraint: Description of the violated constraint.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the validation error.

        Args:
            message: Human-readable error description.
            field: Name of the invalid field.
            value: The invalid value.
            constraint: Description of the violated constraint.
            code: Machine-readable error code.
            details: Additional error context.
        """
        details = details or {}
        if field is not None:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)
        if constraint is not None:
            details["constraint"] = constraint

        super().__init__(message, code=code or "VALIDATION_ERROR", details=details)
        self.field = field
        self.value = value
        self.constraint = constraint


class SecurityError(UnunseptiumError):
    """Raised when a security violation occurs.

    This includes access control failures, authentication errors,
    and encryption/decryption failures.

    Attributes:
        operation: The security operation that failed.
        principal: The identity attempting the operation.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        principal: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the security error.

        Args:
            message: Human-readable error description.
            operation: The failed security operation.
            principal: The identity attempting the operation.
            code: Machine-readable error code.
            details: Additional error context.
        """
        details = details or {}
        if operation is not None:
            details["operation"] = operation
        if principal is not None:
            details["principal"] = principal

        super().__init__(message, code=code or "SECURITY_ERROR", details=details)
        self.operation = operation
        self.principal = principal


class IntegrityError(UnunseptiumError):
    """Raised when data integrity checks fail.

    This includes hash mismatches, tampered audit logs,
    and corrupted data detection.

    Attributes:
        expected: The expected integrity value.
        actual: The actual computed value.
        resource: The resource that failed integrity check.
    """

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
        resource: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the integrity error.

        Args:
            message: Human-readable error description.
            expected: The expected integrity value.
            actual: The actual computed value.
            resource: The resource that failed verification.
            code: Machine-readable error code.
            details: Additional error context.
        """
        details = details or {}
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual
        if resource is not None:
            details["resource"] = resource

        super().__init__(message, code=code or "INTEGRITY_ERROR", details=details)
        self.expected = expected
        self.actual = actual
        self.resource = resource


class ModelError(UnunseptiumError):
    """Raised when model operations fail.

    This includes model loading failures, prediction errors,
    and model validation failures.

    Attributes:
        model_id: Identifier of the failing model.
        operation: The model operation that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        model_id: str | None = None,
        operation: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the model error.

        Args:
            message: Human-readable error description.
            model_id: Identifier of the failing model.
            operation: The failed model operation.
            code: Machine-readable error code.
            details: Additional error context.
        """
        details = details or {}
        if model_id is not None:
            details["model_id"] = model_id
        if operation is not None:
            details["operation"] = operation

        super().__init__(message, code=code or "MODEL_ERROR", details=details)
        self.model_id = model_id
        self.operation = operation


class ConfigurationError(UnunseptiumError):
    """Raised when configuration is invalid or missing.

    This includes missing required settings, invalid configuration files,
    and environment variable errors.

    Attributes:
        setting: The configuration setting that caused the error.
        source: The source of the configuration (file, env, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        setting: str | None = None,
        source: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the configuration error.

        Args:
            message: Human-readable error description.
            setting: The problematic configuration setting.
            source: The source of the configuration.
            code: Machine-readable error code.
            details: Additional error context.
        """
        details = details or {}
        if setting is not None:
            details["setting"] = setting
        if source is not None:
            details["source"] = source

        super().__init__(message, code=code or "CONFIGURATION_ERROR", details=details)
        self.setting = setting
        self.source = source


class WorkflowError(UnunseptiumError):
    """Raised when workflow execution fails.

    This includes step failures, timeout errors,
    and workflow state violations.

    Attributes:
        workflow_id: Identifier of the failing workflow.
        step: The step that caused the failure.
        state: The workflow state at failure.
    """

    def __init__(
        self,
        message: str,
        *,
        workflow_id: str | None = None,
        step: str | None = None,
        state: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the workflow error.

        Args:
            message: Human-readable error description.
            workflow_id: Identifier of the failing workflow.
            step: The failing step.
            state: The workflow state at failure.
            code: Machine-readable error code.
            details: Additional error context.
        """
        details = details or {}
        if workflow_id is not None:
            details["workflow_id"] = workflow_id
        if step is not None:
            details["step"] = step
        if state is not None:
            details["state"] = state

        super().__init__(message, code=code or "WORKFLOW_ERROR", details=details)
        self.workflow_id = workflow_id
        self.step = step
        self.state = state


class ScreeningError(UnunseptiumError):
    """Raised when screening operations fail.

    This includes watchlist lookup failures, matching errors,
    and screening service unavailability.

    Attributes:
        list_type: The type of screening list.
        entity_id: The entity being screened.
    """

    def __init__(
        self,
        message: str,
        *,
        list_type: str | None = None,
        entity_id: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the screening error.

        Args:
            message: Human-readable error description.
            list_type: The type of screening list.
            entity_id: The entity being screened.
            code: Machine-readable error code.
            details: Additional error context.
        """
        details = details or {}
        if list_type is not None:
            details["list_type"] = list_type
        if entity_id is not None:
            details["entity_id"] = entity_id

        super().__init__(message, code=code or "SCREENING_ERROR", details=details)
        self.list_type = list_type
        self.entity_id = entity_id
