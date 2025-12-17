"""Numerical utilities for mathematical stability.

Provides utilities for numerically stable computations.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class NumericsUtils:
    """Numerical computation utilities.

    Provides stable implementations of common operations.

    Example:
        ```python
        from ununseptium.mathstats import NumericsUtils

        # Stable log-sum-exp
        values = np.array([1000.0, 1000.1, 1000.2])
        result = NumericsUtils.logsumexp(values)

        # Stable softmax
        logits = np.array([1.0, 2.0, 3.0])
        probs = NumericsUtils.softmax(logits)
        ```
    """

    @staticmethod
    def logsumexp(x: np.ndarray, axis: int | None = None) -> float | np.ndarray:
        """Compute log(sum(exp(x))) in a numerically stable way.

        Args:
            x: Input array.
            axis: Axis to reduce over.

        Returns:
            Log-sum-exp result.
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

        if axis is not None:
            return result.squeeze(axis=axis)
        return float(result.item())

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax in a numerically stable way.

        Args:
            x: Input logits.
            axis: Axis to compute over.

        Returns:
            Softmax probabilities.
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute log-softmax in a numerically stable way.

        Args:
            x: Input logits.
            axis: Axis to compute over.

        Returns:
            Log-softmax values.
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute sigmoid in a numerically stable way.

        Args:
            x: Input array.

        Returns:
            Sigmoid values.
        """
        # Use conditional for stability
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = np.zeros_like(x)
        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[neg_mask])
        result[neg_mask] = exp_x / (1 + exp_x)

        return result

    @staticmethod
    def log1p_exp(x: np.ndarray) -> np.ndarray:
        """Compute log(1 + exp(x)) stably.

        Args:
            x: Input array.

        Returns:
            log(1 + exp(x)).
        """
        # log1p(exp(x)) = x + log1p(exp(-abs(x))) for stability
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))

    @staticmethod
    def safe_log(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Compute log with floor for numerical stability.

        Args:
            x: Input array.
            eps: Minimum value floor.

        Returns:
            Log values with floored inputs.
        """
        return np.log(np.maximum(x, eps))

    @staticmethod
    def safe_divide(
        a: np.ndarray,
        b: np.ndarray,
        default: float = 0.0,
    ) -> np.ndarray:
        """Safe division handling zeros.

        Args:
            a: Numerator.
            b: Denominator.
            default: Value to use when denominator is zero.

        Returns:
            a / b with zeros handled.
        """
        result = np.full_like(a, default, dtype=float)
        non_zero = b != 0
        result[non_zero] = a[non_zero] / b[non_zero]
        return result

    @staticmethod
    def welford_update(
        count: int,
        mean: float,
        m2: float,
        new_value: float,
    ) -> tuple[int, float, float]:
        """Welford's online algorithm for mean and variance.

        Args:
            count: Current count.
            mean: Current mean.
            m2: Current sum of squared deviations.
            new_value: New observation.

        Returns:
            Tuple of (new_count, new_mean, new_m2).
        """
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        m2 += delta * delta2
        return count, mean, m2

    @staticmethod
    def welford_finalize(count: int, m2: float) -> tuple[float, float]:
        """Finalize Welford algorithm to get variance.

        Args:
            count: Sample count.
            m2: Sum of squared deviations.

        Returns:
            Tuple of (variance, sample_variance).
        """
        if count < 2:
            return 0.0, 0.0
        variance = m2 / count
        sample_variance = m2 / (count - 1)
        return variance, sample_variance

    @staticmethod
    def kahan_sum(values: np.ndarray) -> float:
        """Kahan summation for reduced floating-point error.

        Args:
            values: Values to sum.

        Returns:
            Accurate sum.
        """
        total = 0.0
        compensation = 0.0

        for v in values.flat:
            y = v - compensation
            t = total + y
            compensation = (t - total) - y
            total = t

        return total

    @staticmethod
    def numerical_gradient(
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical gradient via central differences.

        Args:
            f: Function to differentiate.
            x: Point at which to compute gradient.
            eps: Step size.

        Returns:
            Gradient array.
        """
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

        return grad

    @staticmethod
    def check_gradient(
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        eps: float = 1e-5,
        tol: float = 1e-4,
    ) -> tuple[bool, float]:
        """Check gradient implementation against numerical gradient.

        Args:
            f: Function.
            grad_f: Gradient function.
            x: Point to check.
            eps: Numerical gradient step size.
            tol: Tolerance for error.

        Returns:
            Tuple of (passed, max_error).
        """
        numerical = NumericsUtils.numerical_gradient(f, x, eps)
        analytical = grad_f(x)

        diff = np.abs(numerical - analytical)
        max_error = np.max(diff)

        return max_error < tol, float(max_error)

    @staticmethod
    def is_positive_definite(matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite.

        Args:
            matrix: Square matrix to check.

        Returns:
            True if positive definite.
        """
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def make_positive_definite(
        matrix: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Make a matrix positive definite by adding to diagonal.

        Args:
            matrix: Input matrix.
            eps: Minimum eigenvalue.

        Returns:
            Positive definite matrix.
        """
        # Symmetrize
        matrix = (matrix + matrix.T) / 2

        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eig = np.min(eigenvalues)

        if min_eig < eps:
            # Add to diagonal
            matrix = matrix + (eps - min_eig) * np.eye(len(matrix))

        return matrix
