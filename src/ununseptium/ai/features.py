"""Feature engineering for ML models.

Provides feature extraction and transformation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    """Types of features."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BINARY = "binary"


class FeatureSpec(BaseModel):
    """Specification for a feature.

    Attributes:
        name: Feature name.
        feature_type: Type of feature.
        source_field: Source field(s) for extraction.
        transformer: Transformation to apply.
        default_value: Default when missing.
        description: Feature description.
    """

    name: str
    feature_type: FeatureType
    source_field: str | list[str]
    transformer: str | None = None
    default_value: Any = None
    description: str = ""


class FeatureResult(BaseModel):
    """Extracted feature result.

    Attributes:
        name: Feature name.
        value: Extracted value.
        encoded_value: Encoded numeric value.
        metadata: Additional metadata.
    """

    name: str
    value: Any
    encoded_value: float | list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureEngineer:
    """Feature engineering for AML/KYC data.

    Example:
        ```python
        from ununseptium.ai import FeatureEngineer, FeatureSpec, FeatureType

        engineer = FeatureEngineer()

        # Define features
        engineer.add_feature(FeatureSpec(
            name="amount_normalized",
            feature_type=FeatureType.NUMERIC,
            source_field="amount",
            transformer="log1p"
        ))

        # Extract features
        data = {"amount": 10000}
        features = engineer.extract(data)
        ```
    """

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self._specs: dict[str, FeatureSpec] = {}
        self._transformers: dict[str, Callable[[Any], float]] = {
            "log1p": lambda x: float(np.log1p(float(x))),
            "sqrt": lambda x: float(np.sqrt(float(x))),
            "zscore": lambda x: float(x),  # Requires fitting
            "minmax": lambda x: float(x),  # Requires fitting
            "identity": lambda x: float(x),
        }
        self._fitted_params: dict[str, dict[str, float]] = {}

    def add_feature(self, spec: FeatureSpec) -> None:
        """Add a feature specification.

        Args:
            spec: Feature specification.
        """
        self._specs[spec.name] = spec

    def add_features(self, specs: list[FeatureSpec]) -> None:
        """Add multiple feature specifications.

        Args:
            specs: Feature specifications.
        """
        for spec in specs:
            self.add_feature(spec)

    def fit(self, data: list[dict[str, Any]]) -> None:
        """Fit transformers that require statistics.

        Args:
            data: Training data.
        """
        for name, spec in self._specs.items():
            if spec.transformer in ("zscore", "minmax"):
                values = []
                for row in data:
                    val = self._get_value(row, spec.source_field)
                    if val is not None:
                        values.append(float(val))

                if values:
                    arr = np.array(values)
                    self._fitted_params[name] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }

    def extract(self, data: dict[str, Any]) -> dict[str, FeatureResult]:
        """Extract features from data.

        Args:
            data: Input data dictionary.

        Returns:
            Dictionary of feature results.
        """
        results: dict[str, FeatureResult] = {}

        for name, spec in self._specs.items():
            value = self._get_value(data, spec.source_field)

            if value is None:
                value = spec.default_value

            encoded = self._transform(name, spec, value)

            results[name] = FeatureResult(
                name=name,
                value=value,
                encoded_value=encoded,
            )

        return results

    def extract_vector(self, data: dict[str, Any]) -> np.ndarray:
        """Extract features as a numeric vector.

        Args:
            data: Input data.

        Returns:
            Feature vector.
        """
        results = self.extract(data)
        values = []

        for name in sorted(self._specs.keys()):
            result = results.get(name)
            if result and result.encoded_value is not None:
                if isinstance(result.encoded_value, list):
                    values.extend(result.encoded_value)
                else:
                    values.append(result.encoded_value)
            else:
                values.append(0.0)

        return np.array(values)

    def _get_value(
        self,
        data: dict[str, Any],
        source: str | list[str],
    ) -> Any:
        """Get value from data dictionary."""
        if isinstance(source, str):
            return data.get(source)

        # Multiple sources - return dict
        return {s: data.get(s) for s in source}

    def _transform(
        self,
        name: str,
        spec: FeatureSpec,
        value: Any,
    ) -> float | list[float] | None:
        """Transform a value using the specified transformer."""
        if value is None:
            return None

        transformer = spec.transformer or "identity"

        if transformer == "zscore" and name in self._fitted_params:
            params = self._fitted_params[name]
            std = params["std"]
            if std > 0:
                return (float(value) - params["mean"]) / std
            return 0.0

        if transformer == "minmax" and name in self._fitted_params:
            params = self._fitted_params[name]
            range_val = params["max"] - params["min"]
            if range_val > 0:
                return (float(value) - params["min"]) / range_val
            return 0.5

        if transformer in self._transformers:
            try:
                return self._transformers[transformer](value)
            except (ValueError, TypeError):
                return None

        return None

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            Feature names.
        """
        return sorted(self._specs.keys())


class FeatureEncoder:
    """Encode categorical and other non-numeric features.

    Example:
        ```python
        from ununseptium.ai import FeatureEncoder

        encoder = FeatureEncoder()

        # Fit on categories
        encoder.fit_categorical("country", ["US", "UK", "DE", "FR"])

        # Encode
        encoded = encoder.encode_categorical("country", "US")
        ```
    """

    def __init__(self) -> None:
        """Initialize the encoder."""
        self._category_maps: dict[str, dict[str, int]] = {}
        self._reverse_maps: dict[str, dict[int, str]] = {}

    def fit_categorical(
        self,
        name: str,
        categories: list[str],
    ) -> None:
        """Fit a categorical encoder.

        Args:
            name: Feature name.
            categories: Unique categories.
        """
        self._category_maps[name] = {cat: i for i, cat in enumerate(categories)}
        self._reverse_maps[name] = {i: cat for cat, i in self._category_maps[name].items()}

    def encode_categorical(
        self,
        name: str,
        value: str,
        *,
        unknown_value: int = -1,
    ) -> int:
        """Encode a categorical value.

        Args:
            name: Feature name.
            value: Value to encode.
            unknown_value: Value for unknown categories.

        Returns:
            Encoded integer.
        """
        cat_map = self._category_maps.get(name, {})
        return cat_map.get(value, unknown_value)

    def encode_onehot(
        self,
        name: str,
        value: str,
    ) -> list[float]:
        """Encode as one-hot vector.

        Args:
            name: Feature name.
            value: Value to encode.

        Returns:
            One-hot vector.
        """
        cat_map = self._category_maps.get(name, {})
        n_categories = len(cat_map)
        vector = [0.0] * n_categories

        idx = cat_map.get(value)
        if idx is not None:
            vector[idx] = 1.0

        return vector

    def decode_categorical(
        self,
        name: str,
        encoded: int,
    ) -> str | None:
        """Decode an encoded value.

        Args:
            name: Feature name.
            encoded: Encoded value.

        Returns:
            Original category.
        """
        return self._reverse_maps.get(name, {}).get(encoded)

    def encode_temporal(
        self,
        timestamp: datetime,
        *,
        include_cyclical: bool = True,
    ) -> list[float]:
        """Encode a timestamp as features.

        Args:
            timestamp: Timestamp to encode.
            include_cyclical: Include cyclical encodings.

        Returns:
            List of temporal features.
        """
        features = [
            float(timestamp.year),
            float(timestamp.month),
            float(timestamp.day),
            float(timestamp.hour),
            float(timestamp.weekday()),
        ]

        if include_cyclical:
            # Cyclical encoding for hour
            hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
            hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
            features.extend([hour_sin, hour_cos])

            # Cyclical encoding for day of week
            dow_sin = np.sin(2 * np.pi * timestamp.weekday() / 7)
            dow_cos = np.cos(2 * np.pi * timestamp.weekday() / 7)
            features.extend([dow_sin, dow_cos])

        return features
