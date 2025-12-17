"""KYC module for identity verification and compliance workflows.

Provides comprehensive Know Your Customer (KYC) functionality including:
- Identity management and verification
- Document processing and validation
- Sanctions and PEP screening
- Entity resolution and matching
- Workflow orchestration
"""

from ununseptium.kyc.documents import (
    Document,
    DocumentExtractor,
    DocumentType,
    DocumentValidator,
    ExtractionResult,
)
from ununseptium.kyc.entity_resolution import (
    EntityResolver,
    MatchResult,
    MatchScore,
    ResolvedEntity,
)
from ununseptium.kyc.identity import (
    Identity,
    IdentityVerifier,
    RiskLevel,
    VerificationResult,
    VerificationStatus,
)
from ununseptium.kyc.screening import (
    ScreeningEngine,
    ScreeningMatch,
    ScreeningResult,
    WatchlistEntry,
    WatchlistMatcher,
    WatchlistType,
)
from ununseptium.kyc.workflows import (
    KYCWorkflow,
    WorkflowResult,
    WorkflowState,
    WorkflowStep,
)

__all__ = [
    # Identity
    "Identity",
    "IdentityVerifier",
    "RiskLevel",
    "VerificationResult",
    "VerificationStatus",
    # Documents
    "Document",
    "DocumentExtractor",
    "DocumentType",
    "DocumentValidator",
    "ExtractionResult",
    # Screening
    "ScreeningEngine",
    "ScreeningMatch",
    "ScreeningResult",
    "WatchlistEntry",
    "WatchlistMatcher",
    "WatchlistType",
    # Entity Resolution
    "EntityResolver",
    "MatchResult",
    "MatchScore",
    "ResolvedEntity",
    # Workflows
    "KYCWorkflow",
    "WorkflowResult",
    "WorkflowState",
    "WorkflowStep",
]
