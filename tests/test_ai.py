"""Comprehensive tests for AI module."""

from __future__ import annotations

import numpy as np
import pytest

from ununseptium.ai.features import (
    FeatureEncoder,
    FeatureEngineer,
    FeatureSpec,
    FeatureType,
)
from ununseptium.ai.models import (
    EnsembleModel,
    PredictionResult,
    RiskScorer,
)


class TestRiskScorer:
    """Test RiskScorer model."""

    def test_scorer_creation(self):
        """Test scorer instantiation."""
        scorer = RiskScorer("test_scorer")
        assert scorer is not None
        assert scorer.model_id == "test_scorer"

    def test_scorer_with_custom_thresholds(self):
        """Test scorer with custom thresholds."""
        scorer = RiskScorer(
            "test",
            thresholds=[0.25, 0.75],
            labels=["low", "medium", "high"]
        )
        assert scorer.thresholds == [0.25, 0.75]
        assert scorer.labels == ["low", "medium", "high"]

    def test_predict_low_risk(self):
        """Test low risk prediction."""
        scorer = RiskScorer("test", thresholds=[0.3, 0.7])
        features = np.array([0.2])
        
        result = scorer.predict(features)
        
        assert isinstance(result, PredictionResult)
        assert result.score == 0.2
        assert result.label == "low"
        assert result.model_id == "test"

    def test_predict_medium_risk(self):
        """Test medium risk prediction."""
        scorer = RiskScorer("test", thresholds=[0.3, 0.7])
        features = np.array([0.5])
        
        result = scorer.predict(features)
        
        assert result.score == 0.5
        assert result.label == "medium"

    def test_predict_high_risk(self):
        """Test high risk prediction."""
        scorer = RiskScorer("test", thresholds=[0.3, 0.7])
        features = np.array([0.9])
        
        result = scorer.predict(features)
        
        assert result.score == 0.9
        assert result.label == "high"

    def test_predict_batch(self):
        """Test batch predictions."""
        scorer = RiskScorer("test")
        features = np.array([[0.1], [0.5], [0.9]])
        
        results = scorer.predict_batch(features)
        
        assert len(results) == 3
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_predict_clips_score(self):
        """Test score clipping to [0,1]."""
        scorer = RiskScorer("test")
        
        # Test > 1
        result1 = scorer.predict(np.array([1.5]))
        assert result1.score == 1.0
        
        # Test < 0
        result2 = scorer.predict(np.array([-0.5]))
        assert result2.score == 0.0


class TestEnsembleModel:
    """Test EnsembleModel."""

    def test_ensemble_creation(self):
        """Test ensemble instantiation."""
        model1 = RiskScorer("model1")
        model2 = RiskScorer("model2")
        
        ensemble = EnsembleModel("ensemble", [model1, model2])
        
        assert ensemble is not None
        assert len(ensemble.models) == 2

    def test_ensemble_predict_mean(self):
        """Test ensemble with mean aggregation."""
        model1 = RiskScorer("model1", thresholds=[0.5])
        model2 = RiskScorer("model2", thresholds=[0.5])
        
        ensemble = EnsembleModel("ensemble", [model1, model2], aggregation="mean")
        
        result = ensemble.predict(np.array([0.6]))
        
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.score <= 1.0

    def test_ensemble_weights(self):
        """Test weighted ensemble."""
        model1 = RiskScorer("model1")
        model2 = RiskScorer("model2")
        
        ensemble = EnsembleModel(
            "ensemble",
            [model1, model2],
            weights=[0.7, 0.3]
        )
        
        assert ensemble.weights == [0.7, 0.3]

    def test_ensemble_add_model(self):
        """Test adding model to ensemble."""
        model1 = RiskScorer("model1")
        ensemble = EnsembleModel("ensemble", [model1])
        
        model2 = RiskScorer("model2")
        ensemble.add_model(model2, weight=1.0)
        
        assert len(ensemble.models) == 2
        # Weights should be renormalized
        assert sum(ensemble.weights) == pytest.approx(1.0)

    def test_ensemble_different_aggregations(self):
        """Test different aggregation methods."""
        model1 = RiskScorer("model1", thresholds=[0.3])
        model2 = RiskScorer("model2", thresholds=[0.7])
        features = np.array([0.5])
        
        # Mean
        ens_mean = EnsembleModel("mean", [model1, model2], aggregation="mean")
        result_mean = ens_mean.predict(features)
        assert result_mean.score == 0.5
        
        # Median
        ens_median = EnsembleModel("median", [model1, model2], aggregation="median")
        result_median = ens_median.predict(features)
        assert result_median.score == 0.5
        
        # Max
        ens_max = EnsembleModel("max", [model1, model2], aggregation="max")
        result_max = ens_max.predict(features)
        assert result_max.score == 0.5


class TestFeatureEngineer:
    """Test FeatureEngineer."""

    def test_engineer_creation(self):
        """Test engineer instantiation."""
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_add_feature(self):
        """Test adding feature spec."""
        engineer = FeatureEngineer()
        spec = FeatureSpec(
            name="amount",
            feature_type=FeatureType.NUMERIC,
            source_field="amount"
        )
        
        engineer.add_feature(spec)
        
        assert "amount" in engineer._specs

    def test_extract_simple_feature(self):
        """Test extracting simple feature."""
        engineer = FeatureEngineer()
        engineer.add_feature(FeatureSpec(
            name="amount",
            feature_type=FeatureType.NUMERIC,
            source_field="amount",
            transformer="identity"
        ))
        
        data = {"amount": 100.0}
        results = engineer.extract(data)
        
        assert "amount" in results
        assert results["amount"].value == 100.0
        assert results["amount"].encoded_value == 100.0

    def test_extract_vector(self):
        """Test extracting as vector."""
        engineer = FeatureEngineer()
        engineer.add_feature(FeatureSpec(
            name="amount",
            feature_type=FeatureType.NUMERIC,
            source_field="amount",
            transformer="identity"
        ))
        engineer.add_feature(FeatureSpec(
            name="count",
            feature_type=FeatureType.NUMERIC,
            source_field="count",
            transformer="identity"
        ))
        
        data = {"amount": 100.0, "count": 5.0}
        vector = engineer.extract_vector(data)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 2

    def test_transformer_log1p(self):
        """Test log1p transformer."""
        engineer = FeatureEngineer()
        engineer.add_feature(FeatureSpec(
            name="amount_log",
            feature_type=FeatureType.NUMERIC,
            source_field="amount",
            transformer="log1p"
        ))
        
        data = {"amount": 99.0}
        results = engineer.extract(data)
        
        expected = np.log1p(99.0)
        assert results["amount_log"].encoded_value == pytest.approx(expected)

    def test_default_value(self):
        """Test default value for missing field."""
        engineer = FeatureEngineer()
        engineer.add_feature(FeatureSpec(
            name="amount",
            feature_type=FeatureType.NUMERIC,
            source_field="amount",
            default_value=0.0
        ))
        
        data = {}  # Missing amount
        results = engineer.extract(data)
        
        assert results["amount"].value == 0.0

    def test_fit_zscore(self):
        """Test fitting z-score normalization."""
        engineer = FeatureEngineer()
        engineer.add_feature(FeatureSpec(
            name="amount_z",
            feature_type=FeatureType.NUMERIC,
            source_field="amount",
            transformer="zscore"
        ))
        
        # Fit on training data
        train_data = [
            {"amount": 100.0},
            {"amount": 200.0},
            {"amount": 300.0},
        ]
        engineer.fit(train_data)
        
        # Extract from new data
        data = {"amount": 200.0}
        results = engineer.extract(data)
        
        # Should be close to 0 (it's the mean)
        assert results["amount_z"].encoded_value == pytest.approx(0.0, abs=0.1)


class TestFeatureEncoder:
    """Test FeatureEncoder."""

    def test_encoder_creation(self):
        """Test encoder instantiation."""
        encoder = FeatureEncoder()
        assert encoder is not None

    def test_fit_categorical(self):
        """Test fitting categorical encoder."""
        encoder = FeatureEncoder()
        categories = ["US", "UK", "DE", "FR"]
        
        encoder.fit_categorical("country", categories)
        
        assert "country" in encoder._category_maps
        assert len(encoder._category_maps["country"]) == 4

    def test_encode_categorical(self):
        """Test encoding categorical value."""
        encoder = FeatureEncoder()
        encoder.fit_categorical("country", ["US", "UK", "DE"])
        
        encoded = encoder.encode_categorical("country", "UK")
        
        assert encoded == 1

    def test_encode_unknown_category(self):
        """Test encoding unknown category."""
        encoder = FeatureEncoder()
        encoder.fit_categorical("country", ["US", "UK"])
        
        encoded = encoder.encode_categorical("country", "FR")
        
        assert encoded == -1  # Unknown value

    def test_encode_onehot(self):
        """Test one-hot encoding."""
        encoder = FeatureEncoder()
        encoder.fit_categorical("country", ["US", "UK", "DE"])
        
        onehot = encoder.encode_onehot("country", "UK")
        
        assert onehot == [0.0, 1.0, 0.0]

    def test_decode_categorical(self):
        """Test decoding categorical value."""
        encoder = FeatureEncoder()
        encoder.fit_categorical("country", ["US", "UK", "DE"])
        
        decoded = encoder.decode_categorical("country", 1)
        
        assert decoded == "UK"

    def test_encode_temporal(self):
        """Test temporal encoding."""
        from datetime import UTC, datetime
        
        encoder = FeatureEncoder()
        timestamp = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        
        features = encoder.encode_temporal(timestamp, include_cyclical=True)
        
        assert len(features) == 9  # 5 basic + 4 cyclical
        assert features[0] == 2024.0  # year
        assert features[1] == 6.0     # month
        assert features[2] == 15.0    # day
        assert features[3] == 14.0    # hour

    def test_encode_temporal_without_cyclical(self):
        """Test temporal encoding without cyclical features."""
        from datetime import UTC, datetime
        
        encoder = FeatureEncoder()
        timestamp = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        
        features = encoder.encode_temporal(timestamp, include_cyclical=False)
        
        assert len(features) == 5
