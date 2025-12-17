"""
Ununseptium: State-of-the-art RegTech and Cybersecurity Python Library.

Provides comprehensive tools for:
- KYC/AML automation
- Data security and PII management
- AI-driven risk analysis
- Scientific ML (PINN, Neural ODEs)
"""

from ununseptium.core.config import Settings, load_config
from ununseptium.core.errors import (
    IntegrityError,
    ModelError,
    SecurityError,
    UnunseptiumError,
    ValidationError,
)

__version__ = "1.0.0"
__author__ = "Olaf Laitinen"
__email__ = "olaf.laitinen@protonmail.com"

# Module imports for convenient access
from ununseptium import aml, ai, core, kyc, mathstats, security
from ununseptium import cli, model_zoo, plugins

__all__ = [
    # Version info
    "__author__",
    "__email__",
    "__version__",
    # Configuration
    "Settings",
    "load_config",
    # Errors
    "IntegrityError",
    "ModelError",
    "SecurityError",
    "UnunseptiumError",
    "ValidationError",
    # Modules
    "ai",
    "aml",
    "cli",
    "core",
    "kyc",
    "mathstats",
    "model_zoo",
    "plugins",
    "security",
]
