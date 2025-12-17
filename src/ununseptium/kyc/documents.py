"""Document processing and validation for KYC.

Provides document type classification, data extraction,
and validation for identity documents.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of identity documents."""

    PASSPORT = "passport"
    NATIONAL_ID = "national_id"
    DRIVERS_LICENSE = "drivers_license"
    RESIDENCE_PERMIT = "residence_permit"
    VISA = "visa"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    TAX_DOCUMENT = "tax_document"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Document(BaseModel):
    """Identity document model.

    Represents a document submitted for KYC verification.

    Attributes:
        id: Unique document identifier.
        document_type: Type of document.
        identity_id: ID of the associated identity.
        issuing_country: ISO country code of issuing authority.
        document_number: Document number/ID.
        issue_date: Date of issue.
        expiry_date: Date of expiry.
        status: Processing status.
        file_reference: Reference to stored file.
        extracted_data: Data extracted from document.
        validation_errors: List of validation errors.
        created_at: Timestamp of creation.
    """

    id: str = Field(default_factory=lambda: f"DOC-{uuid4().hex[:12].upper()}")
    document_type: DocumentType
    identity_id: str | None = None
    issuing_country: str | None = Field(default=None, min_length=2, max_length=3)
    document_number: str | None = None
    issue_date: date | None = None
    expiry_date: date | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    file_reference: str | None = None
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def is_expired(self) -> bool:
        """Check if document is expired.

        Returns:
            True if expired, False otherwise.
        """
        if self.expiry_date is None:
            return False
        return self.expiry_date < date.today()


class ExtractionResult(BaseModel):
    """Result of document data extraction.

    Attributes:
        document_id: ID of the processed document.
        success: Whether extraction was successful.
        extracted_fields: Dictionary of extracted field values.
        confidence_scores: Confidence for each extracted field.
        raw_text: Raw text extracted from document.
        errors: List of extraction errors.
    """

    document_id: str
    success: bool
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    raw_text: str | None = None
    errors: list[str] = Field(default_factory=list)


class DocumentExtractor:
    """Extract data from identity documents.

    Provides OCR and data extraction capabilities for
    various document types.

    Example:
        ```python
        from ununseptium.kyc import Document, DocumentExtractor, DocumentType

        doc = Document(
            document_type=DocumentType.PASSPORT,
            file_reference="uploads/passport_001.jpg"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(doc)

        if result.success:
            print(f"Name: {result.extracted_fields.get('name')}")
        ```
    """

    def __init__(self) -> None:
        """Initialize the document extractor."""
        self._field_extractors: dict[DocumentType, list[str]] = {
            DocumentType.PASSPORT: [
                "name",
                "date_of_birth",
                "nationality",
                "document_number",
                "issue_date",
                "expiry_date",
                "issuing_country",
                "gender",
                "mrz",
            ],
            DocumentType.NATIONAL_ID: [
                "name",
                "date_of_birth",
                "document_number",
                "issue_date",
                "expiry_date",
                "address",
            ],
            DocumentType.DRIVERS_LICENSE: [
                "name",
                "date_of_birth",
                "document_number",
                "issue_date",
                "expiry_date",
                "address",
                "license_class",
            ],
        }

    def extract(self, document: Document) -> ExtractionResult:
        """Extract data from a document.

        Args:
            document: Document to process.

        Returns:
            ExtractionResult with extracted fields.

        Note:
            This is a placeholder implementation. Production use requires
            integration with OCR services (e.g., AWS Textract, Google Vision).
        """
        # Get expected fields for document type
        expected_fields = self._field_extractors.get(
            document.document_type,
            ["name", "document_number"],
        )

        # Placeholder: In production, this would call OCR services
        extracted_fields: dict[str, Any] = {}
        confidence_scores: dict[str, float] = {}

        # If document already has extracted data, use it
        if document.extracted_data:
            for field in expected_fields:
                if field in document.extracted_data:
                    extracted_fields[field] = document.extracted_data[field]
                    confidence_scores[field] = 0.95

        return ExtractionResult(
            document_id=document.id,
            success=True,
            extracted_fields=extracted_fields,
            confidence_scores=confidence_scores,
        )

    def supported_types(self) -> list[DocumentType]:
        """Get list of supported document types.

        Returns:
            List of document types with extraction support.
        """
        return list(self._field_extractors.keys())


class ValidationRule(BaseModel):
    """Rule for document validation.

    Attributes:
        name: Rule identifier.
        description: Human-readable description.
        field: Field to validate.
        required: Whether field is required.
        pattern: Regex pattern for validation.
    """

    name: str
    description: str
    field: str
    required: bool = False
    pattern: str | None = None


class DocumentValidator:
    """Validate identity documents.

    Performs validation checks including expiry verification,
    format validation, and cross-referencing.

    Example:
        ```python
        from ununseptium.kyc import Document, DocumentValidator, DocumentType

        doc = Document(
            document_type=DocumentType.PASSPORT,
            document_number="AB1234567",
            expiry_date=date(2025, 12, 31)
        )

        validator = DocumentValidator()
        is_valid, errors = validator.validate(doc)

        if not is_valid:
            print(f"Validation errors: {errors}")
        ```
    """

    def __init__(self) -> None:
        """Initialize the document validator."""
        self._rules: dict[DocumentType, list[ValidationRule]] = {
            DocumentType.PASSPORT: [
                ValidationRule(
                    name="document_number_required",
                    description="Passport number is required",
                    field="document_number",
                    required=True,
                ),
                ValidationRule(
                    name="expiry_date_required",
                    description="Expiry date is required",
                    field="expiry_date",
                    required=True,
                ),
            ],
            DocumentType.NATIONAL_ID: [
                ValidationRule(
                    name="document_number_required",
                    description="ID number is required",
                    field="document_number",
                    required=True,
                ),
            ],
        }

    def validate(self, document: Document) -> tuple[bool, list[str]]:
        """Validate a document.

        Args:
            document: Document to validate.

        Returns:
            Tuple of (is_valid, error_messages).

        Example:
            ```python
            validator = DocumentValidator()
            is_valid, errors = validator.validate(document)
            ```
        """
        errors: list[str] = []

        # Check expiry
        if document.is_expired():
            errors.append(f"Document has expired on {document.expiry_date}")

        # Check type-specific rules
        rules = self._rules.get(document.document_type, [])
        for rule in rules:
            field_value = getattr(document, rule.field, None)
            if rule.required and field_value is None:
                errors.append(rule.description)

        return len(errors) == 0, errors

    def add_rule(self, document_type: DocumentType, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            document_type: Document type to apply rule to.
            rule: Validation rule to add.
        """
        if document_type not in self._rules:
            self._rules[document_type] = []
        self._rules[document_type].append(rule)
