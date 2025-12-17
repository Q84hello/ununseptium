"""Sequential detection methods.

Provides change point and drift detection algorithms.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class ChangePointResult(BaseModel):
    """Result of change point detection.

    Attributes:
        detected: Whether change was detected.
        statistic: Test statistic value.
        threshold: Detection threshold.
        location: Estimated change location.
        detected_at: Detection timestamp.
    """

    detected: bool
    statistic: float
    threshold: float
    location: int | None = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class CUSUM:
    """Cumulative Sum (CUSUM) change detector.

    Detects mean shifts in a data stream.

    Mathematical Foundation:
        S_t^+ = max(0, S_{t-1}^+ + (x_t - mu_0 - k))
        S_t^- = max(0, S_{t-1}^- - (x_t - mu_0 - k))

    Alarms when S_t^+ or S_t^- exceeds threshold h.

    Example:
        ```python
        from ununseptium.mathstats import CUSUM

        detector = CUSUM(target_mean=100, threshold=5, slack=0.5)

        for value in data_stream:
            result = detector.update(value)
            if result.detected:
                print(f"Change detected at {detector.n}")
                detector.reset()
        ```
    """

    def __init__(
        self,
        target_mean: float = 0.0,
        threshold: float = 5.0,
        slack: float = 0.5,
    ) -> None:
        """Initialize CUSUM detector.

        Args:
            target_mean: Expected mean (mu_0).
            threshold: Detection threshold (h).
            slack: Allowable slack parameter (k).
        """
        self.target_mean = target_mean
        self.threshold = threshold
        self.slack = slack
        self._s_plus = 0.0
        self._s_minus = 0.0
        self.n = 0

    def update(self, value: float) -> ChangePointResult:
        """Update detector with new observation.

        Args:
            value: New observation.

        Returns:
            ChangePointResult.
        """
        self.n += 1

        # Update CUSUM statistics
        self._s_plus = max(0, self._s_plus + (value - self.target_mean - self.slack))
        self._s_minus = max(0, self._s_minus - (value - self.target_mean + self.slack))

        statistic = max(self._s_plus, self._s_minus)
        detected = statistic > self.threshold

        return ChangePointResult(
            detected=detected,
            statistic=statistic,
            threshold=self.threshold,
            location=self.n if detected else None,
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._s_plus = 0.0
        self._s_minus = 0.0
        self.n = 0

    @property
    def state(self) -> dict[str, float]:
        """Current detector state."""
        return {
            "s_plus": self._s_plus,
            "s_minus": self._s_minus,
            "n": self.n,
        }


class SPRT:
    """Sequential Probability Ratio Test.

    Tests between two simple hypotheses sequentially.

    Mathematical Foundation:
        Log-likelihood ratio: L_n = sum(log(f1(x_i) / f0(x_i)))
        Reject H0 if L_n > log(B)
        Accept H0 if L_n < log(A)

    Example:
        ```python
        from ununseptium.mathstats import SPRT

        sprt = SPRT(mu0=0, mu1=1, sigma=1, alpha=0.05, beta=0.1)

        for value in data:
            result = sprt.update(value)
            if result is not None:
                print(f"Decision: {result}")
                break
        ```
    """

    def __init__(
        self,
        mu0: float,
        mu1: float,
        sigma: float = 1.0,
        alpha: float = 0.05,
        beta: float = 0.1,
    ) -> None:
        """Initialize SPRT.

        Args:
            mu0: Mean under null hypothesis.
            mu1: Mean under alternative hypothesis.
            sigma: Known standard deviation.
            alpha: Type I error probability.
            beta: Type II error probability.
        """
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        # Compute boundaries
        self._log_a = np.log(beta / (1 - alpha))
        self._log_b = np.log((1 - beta) / alpha)
        self._log_ratio = 0.0
        self.n = 0

    def update(self, value: float) -> str | None:
        """Update test with new observation.

        Args:
            value: New observation.

        Returns:
            'H0' (accept null), 'H1' (reject null), or None (continue).
        """
        self.n += 1

        # Log-likelihood ratio for normal distributions
        llr = ((self.mu1 - self.mu0) / (self.sigma**2)) * (value - (self.mu0 + self.mu1) / 2)
        self._log_ratio += llr

        if self._log_ratio >= self._log_b:
            return "H1"  # Reject H0
        if self._log_ratio <= self._log_a:
            return "H0"  # Accept H0
        return None  # Continue sampling

    def reset(self) -> None:
        """Reset test state."""
        self._log_ratio = 0.0
        self.n = 0

    @property
    def state(self) -> dict[str, float]:
        """Current test state."""
        return {
            "log_ratio": self._log_ratio,
            "log_a": self._log_a,
            "log_b": self._log_b,
            "n": self.n,
        }


class ADWIN:
    """ADaptive WINdowing for concept drift detection.

    Maintains a sliding window and detects distribution changes.

    Example:
        ```python
        from ununseptium.mathstats import ADWIN

        detector = ADWIN(delta=0.002)

        for value in stream:
            detected = detector.update(value)
            if detected:
                print(f"Drift detected, new mean: {detector.mean}")
        ```
    """

    def __init__(self, delta: float = 0.002) -> None:
        """Initialize ADWIN detector.

        Args:
            delta: Confidence parameter.
        """
        self.delta = delta
        self._window: deque[float] = deque()
        self._sum = 0.0
        self._variance = 0.0
        self.n = 0

    def update(self, value: float) -> bool:
        """Update detector with new observation.

        Args:
            value: New observation.

        Returns:
            True if drift detected.
        """
        self._window.append(value)
        self._sum += value
        self.n += 1

        # Check for drift
        drift_detected = False

        while len(self._window) >= 2:
            if self._check_cut():
                # Remove oldest element
                old = self._window.popleft()
                self._sum -= old
                self.n -= 1
                drift_detected = True
            else:
                break

        return drift_detected

    def _check_cut(self) -> bool:
        """Check if window should be cut."""
        n = len(self._window)
        if n < 2:
            return False

        # Simple cut check: compare halves
        mid = n // 2
        window_list = list(self._window)

        mean1 = np.mean(window_list[:mid])
        mean2 = np.mean(window_list[mid:])
        n1, n2 = mid, n - mid

        # Hoeffding bound
        m = 1 / (1 / n1 + 1 / n2)
        epsilon = np.sqrt(np.log(4 / self.delta) / (2 * m))

        return abs(mean1 - mean2) > epsilon

    @property
    def mean(self) -> float:
        """Current window mean."""
        if self.n == 0:
            return 0.0
        return self._sum / self.n

    def reset(self) -> None:
        """Reset detector state."""
        self._window.clear()
        self._sum = 0.0
        self.n = 0


class DriftDetector:
    """Combined drift detection using multiple methods.

    Example:
        ```python
        from ununseptium.mathstats import DriftDetector

        detector = DriftDetector(methods=['cusum', 'adwin'])

        for value in stream:
            results = detector.update(value)
            if any(r['detected'] for r in results.values()):
                print("Drift detected!")
        ```
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize drift detector.

        Args:
            methods: Detection methods to use.
            **kwargs: Method-specific parameters.
        """
        if methods is None:
            methods = ["cusum", "adwin"]

        self._detectors: dict[str, CUSUM | ADWIN] = {}

        if "cusum" in methods:
            self._detectors["cusum"] = CUSUM(
                target_mean=kwargs.get("cusum_mean", 0.0),
                threshold=kwargs.get("cusum_threshold", 5.0),
                slack=kwargs.get("cusum_slack", 0.5),
            )

        if "adwin" in methods:
            self._detectors["adwin"] = ADWIN(
                delta=kwargs.get("adwin_delta", 0.002),
            )

    def update(self, value: float) -> dict[str, dict[str, Any]]:
        """Update all detectors.

        Args:
            value: New observation.

        Returns:
            Dictionary of results per detector.
        """
        results: dict[str, dict[str, Any]] = {}

        for name, detector in self._detectors.items():
            if isinstance(detector, CUSUM):
                result = detector.update(value)
                results[name] = {
                    "detected": result.detected,
                    "statistic": result.statistic,
                    "threshold": result.threshold,
                }
            elif isinstance(detector, ADWIN):
                detected = detector.update(value)
                results[name] = {
                    "detected": detected,
                    "mean": detector.mean,
                    "n": detector.n,
                }

        return results

    def reset(self) -> None:
        """Reset all detectors."""
        for detector in self._detectors.values():
            detector.reset()
