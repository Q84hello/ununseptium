"""AI module for machine learning and explainability.

Provides ML capabilities for risk assessment:
- Feature engineering
- Model interfaces
- Explainability (SHAP, reason codes)
- Model governance
- Scientific ML (PINN, Neural ODE)
"""

from ununseptium.ai.explain import (
    ExplanationResult,
    FeatureImportance,
    ReasonCodeGenerator,
)
from ununseptium.ai.features import FeatureEncoder, FeatureEngineer, FeatureSpec
from ununseptium.ai.governance import ModelCard, ModelRegistry, ModelValidator
from ununseptium.ai.models import EnsembleModel, ModelInterface, RiskScorer

__all__ = [
    # Features
    "FeatureEncoder",
    "FeatureEngineer",
    "FeatureSpec",
    # Models
    "EnsembleModel",
    "ModelInterface",
    "RiskScorer",
    # Explain
    "ExplanationResult",
    "FeatureImportance",
    "ReasonCodeGenerator",
    # Governance
    "ModelCard",
    "ModelRegistry",
    "ModelValidator",
]
