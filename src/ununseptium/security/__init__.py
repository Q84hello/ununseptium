"""Security module for data protection and audit.

Provides comprehensive security functionality including:
- PII detection and masking
- Cryptographic operations
- Access control
- Tamper-evident audit logs
- Data integrity verification
"""

from ununseptium.security.access import AccessController, Permission, Role
from ununseptium.security.audit import AuditLog, AuditVerifier, HashChain
from ununseptium.security.crypto import Encryptor, Hasher, KeyManager
from ununseptium.security.integrity import ConsistencyValidator, IntegrityChecker
from ununseptium.security.pii import PIIDetector, PIIMasker, PIIType, PIIVault

__all__ = [
    # PII
    "PIIDetector",
    "PIIMasker",
    "PIIType",
    "PIIVault",
    # Crypto
    "Encryptor",
    "Hasher",
    "KeyManager",
    # Access
    "AccessController",
    "Permission",
    "Role",
    # Audit
    "AuditLog",
    "AuditVerifier",
    "HashChain",
    # Integrity
    "ConsistencyValidator",
    "IntegrityChecker",
]
