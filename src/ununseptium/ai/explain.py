"""Model explainability and reason codes.

Provides explainability for model predictions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class FeatureImportance(BaseModel):
    """Feature importance for a prediction.

    Attributes:
        feature_name: Name of the feature.
        importance: Importance score.
        contribution: Contribution to prediction.
        direction: Direction of influence (positive/negative).
    """

    feature_name: str
    importance: float
    contribution: float = 0.0
    direction: str = "positive"


class ReasonCode(BaseModel):
    """A reason code explaining a decision.

    Attributes:
        code: Unique reason code.
        description: Human-readable description.
        category: Reason category.
        severity: Severity (1-10).
        contributing_features: Features that triggered this reason.
    """

    code: str
    description: str
    category: str = "general"
    severity: int = Field(default=5, ge=1, le=10)
    contributing_features: list[str] = Field(default_factory=list)


class ExplanationResult(BaseModel):
    """Complete explanation for a prediction.

    Attributes:
        prediction_id: Associated prediction ID.
        model_id: Model that made prediction.
        feature_importances: Feature importance scores.
        reason_codes: Generated reason codes.
        summary: Natural language summary.
        created_at: Explanation timestamp.
        metadata: Additional data.
    """

    prediction_id: str
    model_id: str
    feature_importances: list[FeatureImportance] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def top_reasons(self) -> list[ReasonCode]:
        """Get top reasons by severity."""
        return sorted(self.reason_codes, key=lambda r: r.severity, reverse=True)[:3]


class ReasonCodeGenerator:
    """Generate reason codes from model predictions.

    Example:
        ```python
        from ununseptium.ai import ReasonCodeGenerator
        import numpy as np

        generator = ReasonCodeGenerator()

        # Define reason code rules
        generator.add_rule(
            code="HV001",
            description="High transaction amount",
            condition=lambda f: f.get("amount", 0) > 10000,
            category="transaction"
        )

        # Generate from features
        features = {"amount": 15000, "frequency": 5}
        reasons = generator.generate(features)
        ```
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self._rules: list[dict[str, Any]] = []
        self._categories: dict[str, list[str]] = {}

    def add_rule(
        self,
        code: str,
        description: str,
        condition: Any,
        *,
        category: str = "general",
        severity: int = 5,
        contributing_features: list[str] | None = None,
    ) -> None:
        """Add a reason code rule.

        Args:
            code: Reason code.
            description: Description.
            condition: Function that takes features and returns bool.
            category: Reason category.
            severity: Severity level.
            contributing_features: Features involved.
        """
        self._rules.append(
            {
                "code": code,
                "description": description,
                "condition": condition,
                "category": category,
                "severity": severity,
                "contributing_features": contributing_features or [],
            }
        )

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(code)

    def generate(
        self,
        features: dict[str, Any],
        *,
        max_reasons: int = 5,
    ) -> list[ReasonCode]:
        """Generate reason codes for features.

        Args:
            features: Feature dictionary.
            max_reasons: Maximum reasons to return.

        Returns:
            List of triggered reason codes.
        """
        reasons: list[ReasonCode] = []

        for rule in self._rules:
            try:
                if rule["condition"](features):
                    reasons.append(
                        ReasonCode(
                            code=rule["code"],
                            description=rule["description"],
                            category=rule["category"],
                            severity=rule["severity"],
                            contributing_features=rule["contributing_features"],
                        )
                    )
            except (KeyError, TypeError, ValueError):
                continue

        # Sort by severity and limit
        reasons.sort(key=lambda r: r.severity, reverse=True)
        return reasons[:max_reasons]

    def get_default_rules(self) -> None:
        """Load default AML-related reason code rules."""
        self.add_rule(
            code="AML001",
            description="Transaction amount exceeds threshold",
            condition=lambda f: f.get("amount", 0) >= 10000,
            category="amount",
            severity=7,
            contributing_features=["amount"],
        )

        self.add_rule(
            code="AML002",
            description="High transaction velocity",
            condition=lambda f: f.get("transaction_count", 0) >= 10,
            category="velocity",
            severity=6,
            contributing_features=["transaction_count"],
        )

        self.add_rule(
            code="AML003",
            description="High-risk geographic location",
            condition=lambda f: f.get("country") in ["HIGH_RISK"],
            category="geography",
            severity=8,
            contributing_features=["country"],
        )

        self.add_rule(
            code="AML004",
            description="New customer with high activity",
            condition=lambda f: (
                f.get("account_age_days", 365) < 30 and f.get("transaction_count", 0) > 5
            ),
            category="behavior",
            severity=7,
            contributing_features=["account_age_days", "transaction_count"],
        )

        self.add_rule(
            code="AML005",
            description="Round transaction amounts",
            condition=lambda f: f.get("amount", 0) > 0 and f.get("amount", 1) % 1000 == 0,
            category="pattern",
            severity=5,
            contributing_features=["amount"],
        )


class FeatureExplainer:
    """Explain predictions using feature importance.

    Provides SHAP-like feature importance computation.

    Example:
        ```python
        from ununseptium.ai.explain import FeatureExplainer

        explainer = FeatureExplainer()

        # Explain prediction
        features = np.array([0.5, 0.3, 0.8])
        feature_names = ["amount", "frequency", "velocity"]

        importances = explainer.explain(
            features,
            feature_names,
            prediction_score=0.7
        )
        ```
    """

    def __init__(self, baseline: np.ndarray | None = None) -> None:
        """Initialize the explainer.

        Args:
            baseline: Baseline feature values for comparison.
        """
        self._baseline = baseline

    def explain(
        self,
        features: np.ndarray,
        feature_names: list[str],
        prediction_score: float,
    ) -> list[FeatureImportance]:
        """Explain a prediction.

        Args:
            features: Feature values.
            feature_names: Feature names.
            prediction_score: Model prediction score.

        Returns:
            List of feature importances.
        """
        baseline = self._baseline
        if baseline is None:
            baseline = np.zeros_like(features)

        # Simple attribution: deviation from baseline weighted by feature
        deviations = features - baseline

        # Normalize to sum to prediction score
        abs_dev = np.abs(deviations)
        total = abs_dev.sum()

        if total > 0:
            contributions = (abs_dev / total) * prediction_score
        else:
            contributions = np.ones_like(features) * prediction_score / len(features)

        importances = []
        for i, name in enumerate(feature_names):
            importances.append(
                FeatureImportance(
                    feature_name=name,
                    importance=float(abs_dev[i] / total) if total > 0 else 0.0,
                    contribution=float(contributions[i]),
                    direction="positive" if deviations[i] >= 0 else "negative",
                )
            )

        # Sort by importance
        importances.sort(key=lambda x: x.importance, reverse=True)
        return importances

    def set_baseline(self, baseline: np.ndarray) -> None:
        """Set baseline for explanations.

        Args:
            baseline: Baseline feature values.
        """
        self._baseline = baseline

    def compute_baseline(self, data: np.ndarray) -> np.ndarray:
        """Compute baseline from training data.

        Args:
            data: Training data matrix.

        Returns:
            Baseline (mean) values.
        """
        self._baseline = np.mean(data, axis=0)
        return self._baseline
