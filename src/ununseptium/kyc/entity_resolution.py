"""Entity resolution and matching for KYC.

Provides algorithms for matching and merging identity records
to resolve duplicate and related entities.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MatchConfidence(str, Enum):
    """Confidence level for entity matches."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class MatchScore(BaseModel):
    """Detailed match score breakdown.

    Attributes:
        overall: Overall match score (0.0 to 1.0).
        name_score: Name matching score.
        dob_score: Date of birth matching score.
        address_score: Address matching score.
        document_score: Document number matching score.
        confidence: Confidence classification.
        matched_fields: Fields that contributed to match.
    """

    overall: float = Field(ge=0.0, le=1.0)
    name_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dob_score: float = Field(default=0.0, ge=0.0, le=1.0)
    address_score: float = Field(default=0.0, ge=0.0, le=1.0)
    document_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: MatchConfidence = MatchConfidence.NONE
    matched_fields: list[str] = Field(default_factory=list)

    def model_post_init(self, _context: Any) -> None:
        """Compute confidence level after initialization."""
        if self.overall >= 0.9:
            object.__setattr__(self, "confidence", MatchConfidence.HIGH)
        elif self.overall >= 0.7:
            object.__setattr__(self, "confidence", MatchConfidence.MEDIUM)
        elif self.overall >= 0.5:
            object.__setattr__(self, "confidence", MatchConfidence.LOW)
        else:
            object.__setattr__(self, "confidence", MatchConfidence.NONE)


class MatchResult(BaseModel):
    """Result of matching two entities.

    Attributes:
        entity1_id: ID of first entity.
        entity2_id: ID of second entity.
        score: Detailed match score.
        is_match: Whether entities are considered a match.
        match_type: Type of match relationship.
        evidence: List of evidence supporting match.
        created_at: When match was computed.
    """

    entity1_id: str
    entity2_id: str
    score: MatchScore
    is_match: bool = False
    match_type: str = "potential"
    evidence: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResolvedEntity(BaseModel):
    """A resolved entity combining matched records.

    Attributes:
        id: Unique identifier for resolved entity.
        canonical_name: Best available name.
        source_ids: IDs of source records.
        merged_data: Combined data from sources.
        confidence: Confidence in resolution.
        resolution_notes: Notes about resolution.
        created_at: When entity was resolved.
        updated_at: Last update timestamp.
    """

    id: str = Field(default_factory=lambda: f"RE-{uuid4().hex[:12].upper()}")
    canonical_name: str
    source_ids: list[str] = Field(default_factory=list)
    merged_data: dict[str, Any] = Field(default_factory=dict)
    confidence: MatchConfidence = MatchConfidence.MEDIUM
    resolution_notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MatchConfig(BaseModel):
    """Configuration for entity matching.

    Attributes:
        name_weight: Weight for name matching (0.0 to 1.0).
        dob_weight: Weight for DOB matching.
        address_weight: Weight for address matching.
        document_weight: Weight for document matching.
        match_threshold: Minimum score to consider a match.
        high_confidence_threshold: Threshold for high confidence.
    """

    name_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    dob_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    address_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    document_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    match_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class EntityResolver:
    """Resolve and match entities.

    Provides entity matching and resolution capabilities
    for deduplication and relationship identification.

    Attributes:
        config: Matching configuration.

    Example:
        ```python
        from ununseptium.kyc import EntityResolver, Identity

        resolver = EntityResolver()

        identity1 = Identity(name="John Smith", date_of_birth="1985-03-15")
        identity2 = Identity(name="Jon Smith", date_of_birth="1985-03-15")

        result = resolver.match(identity1, identity2)
        if result.is_match:
            resolved = resolver.resolve([identity1, identity2])
        ```
    """

    def __init__(self, config: MatchConfig | None = None) -> None:
        """Initialize the resolver.

        Args:
            config: Matching configuration.
        """
        self.config = config or MatchConfig()

    def match(self, entity1: Any, entity2: Any) -> MatchResult:
        """Match two entities and compute similarity.

        Args:
            entity1: First entity (must have standard identity fields).
            entity2: Second entity.

        Returns:
            MatchResult with detailed scoring.
        """
        entity1_id = getattr(entity1, "id", str(uuid4()))
        entity2_id = getattr(entity2, "id", str(uuid4()))

        matched_fields: list[str] = []
        evidence: list[str] = []

        # Name matching
        name_score = self._match_names(
            getattr(entity1, "name", ""),
            getattr(entity2, "name", ""),
        )
        if name_score > 0.5:
            matched_fields.append("name")
            evidence.append(f"Name similarity: {name_score:.2%}")

        # Date of birth matching
        dob_score = self._match_dob(
            getattr(entity1, "date_of_birth", None),
            getattr(entity2, "date_of_birth", None),
        )
        if dob_score > 0:
            matched_fields.append("date_of_birth")
            evidence.append(f"DOB match: {dob_score:.2%}")

        # Address matching
        address_score = self._match_address(
            getattr(entity1, "address", None),
            getattr(entity2, "address", None),
        )
        if address_score > 0.5:
            matched_fields.append("address")
            evidence.append(f"Address similarity: {address_score:.2%}")

        # Document matching
        document_score = self._match_documents(
            getattr(entity1, "document_number", None),
            getattr(entity2, "document_number", None),
        )
        if document_score > 0:
            matched_fields.append("document_number")
            evidence.append(f"Document match: {document_score:.2%}")

        # Compute overall score
        overall = (
            self.config.name_weight * name_score
            + self.config.dob_weight * dob_score
            + self.config.address_weight * address_score
            + self.config.document_weight * document_score
        )

        score = MatchScore(
            overall=round(overall, 4),
            name_score=round(name_score, 4),
            dob_score=round(dob_score, 4),
            address_score=round(address_score, 4),
            document_score=round(document_score, 4),
            matched_fields=matched_fields,
        )

        is_match = overall >= self.config.match_threshold
        match_type = (
            "confirmed"
            if overall >= self.config.high_confidence_threshold
            else "potential"
            if is_match
            else "non_match"
        )

        return MatchResult(
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            score=score,
            is_match=is_match,
            match_type=match_type,
            evidence=evidence,
        )

    def resolve(self, entities: list[Any]) -> ResolvedEntity:
        """Resolve multiple entities into a single canonical entity.

        Args:
            entities: List of entities to resolve.

        Returns:
            ResolvedEntity with merged data.
        """
        if not entities:
            msg = "At least one entity required for resolution"
            raise ValueError(msg)

        source_ids = [getattr(e, "id", str(uuid4())) for e in entities]

        # Use first entity's name as canonical (could be more sophisticated)
        canonical_name = getattr(entities[0], "name", "Unknown")

        # Merge data from all entities
        merged_data: dict[str, Any] = {}
        for entity in entities:
            for field in ["name", "date_of_birth", "nationality", "address", "email", "phone"]:
                value = getattr(entity, field, None)
                if value is not None and field not in merged_data:
                    merged_data[field] = value

        # Compute confidence based on agreement
        confidence = self._compute_resolution_confidence(entities)

        return ResolvedEntity(
            canonical_name=canonical_name,
            source_ids=source_ids,
            merged_data=merged_data,
            confidence=confidence,
            resolution_notes=[f"Resolved from {len(entities)} source records"],
        )

    def find_duplicates(
        self,
        entities: list[Any],
        *,
        threshold: float | None = None,
    ) -> list[MatchResult]:
        """Find potential duplicate pairs in a list of entities.

        Args:
            entities: List of entities to check.
            threshold: Optional custom threshold.

        Returns:
            List of MatchResults for potential duplicates.
        """
        threshold = threshold or self.config.match_threshold
        duplicates: list[MatchResult] = []

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                result = self.match(entity1, entity2)
                if result.score.overall >= threshold:
                    duplicates.append(result)

        return duplicates

    def _match_names(self, name1: str, name2: str) -> float:
        """Match two names using fuzzy matching.

        Args:
            name1: First name.
            name2: Second name.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        if not name1 or not name2:
            return 0.0

        # Normalize names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        if n1 == n2:
            return 1.0

        # Token-based matching
        tokens1 = set(n1.split())
        tokens2 = set(n2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _match_dob(self, dob1: Any, dob2: Any) -> float:
        """Match two dates of birth.

        Args:
            dob1: First date.
            dob2: Second date.

        Returns:
            Match score (1.0 if exact, 0.0 otherwise).
        """
        if dob1 is None or dob2 is None:
            return 0.0

        # Convert to strings for comparison
        str1 = str(dob1)
        str2 = str(dob2)

        return 1.0 if str1 == str2 else 0.0

    def _match_address(self, addr1: Any, addr2: Any) -> float:
        """Match two addresses.

        Args:
            addr1: First address.
            addr2: Second address.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        if addr1 is None or addr2 is None:
            return 0.0

        # Get address components
        components = ["street", "city", "postal_code", "country"]
        matches = 0
        total = 0

        for comp in components:
            v1 = getattr(addr1, comp, None)
            v2 = getattr(addr2, comp, None)

            if v1 is not None and v2 is not None:
                total += 1
                if str(v1).lower() == str(v2).lower():
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _match_documents(self, doc1: str | None, doc2: str | None) -> float:
        """Match two document numbers.

        Args:
            doc1: First document number.
            doc2: Second document number.

        Returns:
            Match score (1.0 if exact, 0.0 otherwise).
        """
        if doc1 is None or doc2 is None:
            return 0.0

        # Normalize and compare
        d1 = doc1.upper().replace(" ", "").replace("-", "")
        d2 = doc2.upper().replace(" ", "").replace("-", "")

        return 1.0 if d1 == d2 else 0.0

    def _compute_resolution_confidence(self, entities: list[Any]) -> MatchConfidence:
        """Compute confidence in entity resolution.

        Args:
            entities: Resolved entities.

        Returns:
            Confidence level.
        """
        if len(entities) < 2:
            return MatchConfidence.HIGH

        # Check agreement on key fields
        names = {getattr(e, "name", "").lower() for e in entities}
        dobs = {str(getattr(e, "date_of_birth", "")) for e in entities}

        if len(names) == 1 and len(dobs) == 1:
            return MatchConfidence.HIGH
        if len(names) <= 2:
            return MatchConfidence.MEDIUM
        return MatchConfidence.LOW
