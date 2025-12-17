"""Model interfaces for risk scoring.

Provides base model interfaces and ensemble capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Result of a model prediction.

    Attributes:
        score: Risk score (0.0 to 1.0).
        label: Predicted label.
        probabilities: Class probabilities.
        confidence: Prediction confidence.
        model_id: ID of model that made prediction.
        timestamp: Prediction timestamp.
        metadata: Additional prediction data.
    """

    score: float = Field(ge=0.0, le=1.0)
    label: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInterface(ABC):
    """Abstract base class for risk models.

    Example:
        ```python
        from ununseptium.ai import ModelInterface
        import numpy as np

        class MyModel(ModelInterface):
            def predict(self, features):
                return PredictionResult(score=0.5)

            def predict_batch(self, features_batch):
                return [self.predict(f) for f in features_batch]

        model = MyModel("my_model")
        result = model.predict(np.array([1.0, 2.0, 3.0]))
        ```
    """

    def __init__(self, model_id: str) -> None:
        """Initialize the model.

        Args:
            model_id: Unique model identifier.
        """
        self.model_id = model_id

    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionResult:
        """Make a single prediction.

        Args:
            features: Feature vector.

        Returns:
            PredictionResult.
        """
        ...

    @abstractmethod
    def predict_batch(
        self,
        features_batch: np.ndarray,
    ) -> list[PredictionResult]:
        """Make batch predictions.

        Args:
            features_batch: Feature matrix (n_samples, n_features).

        Returns:
            List of PredictionResults.
        """
        ...


class RiskScorer(ModelInterface):
    """Simple threshold-based risk scorer.

    Example:
        ```python
        from ununseptium.ai import RiskScorer
        import numpy as np

        scorer = RiskScorer("risk_scorer", thresholds=[0.3, 0.7])

        features = np.array([0.5])
        result = scorer.predict(features)
        print(f"Risk: {result.label}")  # "medium"
        ```
    """

    def __init__(
        self,
        model_id: str,
        thresholds: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Initialize the scorer.

        Args:
            model_id: Model identifier.
            thresholds: Score thresholds for labels.
            labels: Labels for threshold ranges.
        """
        super().__init__(model_id)
        self.thresholds = thresholds or [0.3, 0.7]
        self.labels = labels or ["low", "medium", "high"]

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Score based on first feature value.

        Args:
            features: Feature vector.

        Returns:
            PredictionResult.
        """
        # Use first feature as raw score
        raw_score = float(features[0]) if len(features) > 0 else 0.0

        # Normalize to [0, 1]
        score = max(0.0, min(1.0, raw_score))

        # Determine label
        label = self.labels[0]
        for i, threshold in enumerate(self.thresholds):
            if score >= threshold:
                label = self.labels[min(i + 1, len(self.labels) - 1)]

        return PredictionResult(
            score=score,
            label=label,
            model_id=self.model_id,
        )

    def predict_batch(
        self,
        features_batch: np.ndarray,
    ) -> list[PredictionResult]:
        """Batch scoring.

        Args:
            features_batch: Feature matrix.

        Returns:
            List of results.
        """
        return [self.predict(f) for f in features_batch]


class EnsembleModel(ModelInterface):
    """Ensemble of multiple models.

    Combines predictions from multiple models using
    averaging or voting.

    Example:
        ```python
        from ununseptium.ai import EnsembleModel, RiskScorer

        model1 = RiskScorer("model1", thresholds=[0.4])
        model2 = RiskScorer("model2", thresholds=[0.6])

        ensemble = EnsembleModel("ensemble", [model1, model2])

        result = ensemble.predict(np.array([0.5]))
        ```
    """

    def __init__(
        self,
        model_id: str,
        models: list[ModelInterface],
        *,
        weights: list[float] | None = None,
        aggregation: str = "mean",
    ) -> None:
        """Initialize the ensemble.

        Args:
            model_id: Ensemble identifier.
            models: List of base models.
            weights: Model weights for aggregation.
            aggregation: Aggregation method ('mean', 'median', 'max').
        """
        super().__init__(model_id)
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.aggregation = aggregation

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Make ensemble prediction.

        Args:
            features: Feature vector.

        Returns:
            Aggregated PredictionResult.
        """
        predictions = [model.predict(features) for model in self.models]
        scores = [p.score for p in predictions]

        if self.aggregation == "mean":
            final_score = float(np.average(scores, weights=self.weights))
        elif self.aggregation == "median":
            final_score = float(np.median(scores))
        elif self.aggregation == "max":
            final_score = float(np.max(scores))
        else:
            final_score = float(np.mean(scores))

        # Confidence from agreement
        score_std = float(np.std(scores))
        confidence = max(0.0, 1.0 - score_std)

        # Majority vote for label
        labels = [p.label for p in predictions if p.label]
        if labels:
            from collections import Counter

            label_counts = Counter(labels)
            label = label_counts.most_common(1)[0][0]
        else:
            label = None

        return PredictionResult(
            score=final_score,
            label=label,
            confidence=confidence,
            model_id=self.model_id,
            metadata={
                "individual_scores": scores,
                "n_models": len(self.models),
            },
        )

    def predict_batch(
        self,
        features_batch: np.ndarray,
    ) -> list[PredictionResult]:
        """Batch ensemble predictions.

        Args:
            features_batch: Feature matrix.

        Returns:
            List of results.
        """
        return [self.predict(f) for f in features_batch]

    def add_model(self, model: ModelInterface, weight: float = 1.0) -> None:
        """Add a model to the ensemble.

        Args:
            model: Model to add.
            weight: Model weight.
        """
        self.models.append(model)
        self.weights.append(weight)

        # Renormalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
