"""Uncertainty quantification with conformal prediction.

Provides coverage-guaranteed prediction sets and calibration metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class PredictionSet(BaseModel):
    """A prediction set with coverage guarantee.

    Attributes:
        point_estimate: Point prediction.
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.
        confidence_level: Target confidence level (1 - alpha).
        set_size: Size of prediction set for classification.
        labels: Labels in prediction set (classification).
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = Field(ge=0.0, le=1.0)
    set_size: float | None = None
    labels: list[Any] | None = None

    @property
    def interval_width(self) -> float:
        """Width of the prediction interval."""
        return self.upper - self.lower


class CalibrationMetrics(BaseModel):
    """Calibration metrics for probabilistic predictions.

    Attributes:
        ece: Expected Calibration Error.
        mce: Maximum Calibration Error.
        brier: Brier score.
        reliability_data: Binned reliability data.
    """

    ece: float = Field(ge=0.0, le=1.0)
    mce: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0)
    reliability_data: list[dict[str, float]] = Field(default_factory=list)


class ConformalPredictor:
    """Conformal prediction for uncertainty quantification.

    Provides prediction sets/intervals with guaranteed coverage.

    Mathematical Foundation:
        For a miscoverage rate alpha, conformal prediction guarantees:
        P(Y in C(X)) >= 1 - alpha

    Example:
        ```python
        from ununseptium.mathstats import ConformalPredictor
        import numpy as np

        # Calibration data
        y_cal = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        y_hat_cal = np.array([0.15, 0.45, 0.35, 0.75, 0.25])

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(y_cal, y_hat_cal)

        # Get prediction interval
        pred_set = predictor.predict(0.5)
        print(f"Interval: [{pred_set.lower}, {pred_set.upper}]")
        ```
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """Initialize the predictor.

        Args:
            alpha: Miscoverage rate (1 - confidence level).
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        self._scores: np.ndarray | None = None
        self._quantile: float | None = None

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calibrate the predictor using hold-out data.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Calibrated quantile threshold.
        """
        # Compute nonconformity scores (absolute residuals)
        self._scores = np.abs(y_true - y_pred)

        # Compute quantile with finite sample correction
        n = len(self._scores)
        q = min(1.0, (1 - self.alpha) * (n + 1) / n)
        self._quantile = float(np.quantile(self._scores, q))

        return self._quantile

    def predict(self, y_pred: float | np.ndarray) -> PredictionSet | list[PredictionSet]:
        """Generate prediction set(s).

        Args:
            y_pred: Point prediction(s).

        Returns:
            PredictionSet or list of PredictionSets.
        """
        if self._quantile is None:
            msg = "Predictor not calibrated. Call calibrate() first."
            raise ValueError(msg)

        if isinstance(y_pred, np.ndarray):
            return [
                PredictionSet(
                    point_estimate=float(p),
                    lower=float(p - self._quantile),
                    upper=float(p + self._quantile),
                    confidence_level=self.confidence_level,
                )
                for p in y_pred
            ]

        return PredictionSet(
            point_estimate=float(y_pred),
            lower=float(y_pred - self._quantile),
            upper=float(y_pred + self._quantile),
            confidence_level=self.confidence_level,
        )

    def predict_classification(
        self,
        probs: np.ndarray,
        labels: list[Any] | None = None,
    ) -> PredictionSet:
        """Generate prediction set for classification.

        Uses adaptive prediction sets based on probability ordering.

        Args:
            probs: Class probabilities.
            labels: Class labels.

        Returns:
            PredictionSet with included labels.
        """
        if labels is None:
            labels = list(range(len(probs)))

        # Sort by probability descending
        order = np.argsort(-probs)
        cumsum = np.cumsum(probs[order])

        # Include classes until cumulative prob exceeds threshold
        threshold = 1 - self.alpha
        n_include = int(np.searchsorted(cumsum, threshold)) + 1

        included_indices = order[:n_include]
        included_labels = [labels[i] for i in included_indices]

        return PredictionSet(
            point_estimate=float(probs.max()),
            lower=0.0,
            upper=1.0,
            confidence_level=self.confidence_level,
            set_size=float(len(included_labels)),
            labels=included_labels,
        )


class OnlineConformalPredictor:
    """Online conformal prediction with adaptive calibration.

    Updates calibration as new data arrives.

    Example:
        ```python
        predictor = OnlineConformalPredictor(alpha=0.1, window_size=100)

        for y_true, y_pred in stream:
            pred_set = predictor.predict_and_update(y_pred, y_true)
        ```
    """

    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 100,
    ) -> None:
        """Initialize online predictor.

        Args:
            alpha: Miscoverage rate.
            window_size: Size of calibration window.
        """
        self.alpha = alpha
        self.window_size = window_size
        self._scores: list[float] = []

    def predict_and_update(
        self,
        y_pred: float,
        y_true: float | None = None,
    ) -> PredictionSet:
        """Predict and optionally update with true value.

        Args:
            y_pred: Point prediction.
            y_true: True value (for updating calibration).

        Returns:
            PredictionSet.
        """
        # Compute current quantile
        if self._scores:
            scores = np.array(self._scores)
            n = len(scores)
            q = min(1.0, (1 - self.alpha) * (n + 1) / n)
            quantile = float(np.quantile(scores, q))
        else:
            # Default before calibration
            quantile = 1.0

        pred_set = PredictionSet(
            point_estimate=y_pred,
            lower=y_pred - quantile,
            upper=y_pred + quantile,
            confidence_level=1 - self.alpha,
        )

        # Update with new observation
        if y_true is not None:
            score = abs(y_true - y_pred)
            self._scores.append(score)

            # Maintain window size
            if len(self._scores) > self.window_size:
                self._scores.pop(0)

        return pred_set


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute calibration metrics for probabilistic predictions.

    Args:
        y_true: True binary labels (0 or 1).
        y_probs: Predicted probabilities.
        n_bins: Number of bins for reliability diagram.

    Returns:
        CalibrationMetrics with ECE, MCE, and Brier score.
    """
    # Brier score
    brier = float(np.mean((y_probs - y_true) ** 2))

    # Binned calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_probs, bin_edges[1:-1])

    ece = 0.0
    mce = 0.0
    reliability_data = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_probs[mask])
            bin_size = np.sum(mask)

            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / len(y_true)) * gap
            mce = max(mce, gap)

            reliability_data.append({
                "bin": i,
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "count": int(bin_size),
                "gap": float(gap),
            })

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        brier=brier,
        reliability_data=reliability_data,
    )
