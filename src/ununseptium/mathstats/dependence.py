"""Dependence modeling with copulas.

Provides copula fitting and dependence metrics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as sp_stats


class CopulaType(str, Enum):
    """Types of copulas."""

    GAUSSIAN = "gaussian"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"
    EMPIRICAL = "empirical"


class CopulaFitResult(BaseModel):
    """Copula fitting result.

    Attributes:
        copula_type: Type of copula.
        parameter: Copula parameter.
        correlation: Implied correlation (for Gaussian).
        kendall_tau: Kendall's tau.
        log_likelihood: Log-likelihood.
        aic: Akaike Information Criterion.
    """

    copula_type: CopulaType
    parameter: float | None = None
    correlation: float | None = None
    kendall_tau: float | None = None
    log_likelihood: float | None = None
    aic: float | None = None


class DependenceResult(BaseModel):
    """Dependence analysis result.

    Attributes:
        pearson: Pearson correlation.
        spearman: Spearman correlation.
        kendall: Kendall's tau.
        tail_lower: Lower tail dependence.
        tail_upper: Upper tail dependence.
    """

    pearson: float = Field(ge=-1.0, le=1.0)
    spearman: float = Field(ge=-1.0, le=1.0)
    kendall: float = Field(ge=-1.0, le=1.0)
    tail_lower: float = Field(ge=0.0, le=1.0)
    tail_upper: float = Field(ge=0.0, le=1.0)


class DependenceMetrics:
    """Compute dependence metrics between variables.

    Example:
        ```python
        from ununseptium.mathstats import DependenceMetrics
        import numpy as np

        metrics = DependenceMetrics()

        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5

        result = metrics.compute(x, y)
        print(f"Kendall tau: {result.kendall}")
        ```
    """

    def compute(self, x: np.ndarray, y: np.ndarray) -> DependenceResult:
        """Compute all dependence metrics.

        Args:
            x: First variable.
            y: Second variable.

        Returns:
            DependenceResult with all metrics.
        """
        # Correlation measures
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman, _ = sp_stats.spearmanr(x, y)
        kendall, _ = sp_stats.kendalltau(x, y)

        # Tail dependence (empirical)
        tail_lower = self._empirical_tail_dependence(x, y, upper=False)
        tail_upper = self._empirical_tail_dependence(x, y, upper=True)

        return DependenceResult(
            pearson=float(pearson),
            spearman=float(spearman),
            kendall=float(kendall),
            tail_lower=tail_lower,
            tail_upper=tail_upper,
        )

    def _empirical_tail_dependence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        upper: bool = True,
        quantile: float = 0.95,
    ) -> float:
        """Estimate tail dependence empirically."""
        n = len(x)

        # Compute ranks
        u = sp_stats.rankdata(x) / (n + 1)
        v = sp_stats.rankdata(y) / (n + 1)

        if upper:
            q = quantile
            joint = np.sum((u > q) & (v > q))
            marginal = np.sum(u > q)
        else:
            q = 1 - quantile
            joint = np.sum((u < q) & (v < q))
            marginal = np.sum(u < q)

        return joint / marginal if marginal > 0 else 0.0


class CopulaFitter:
    """Fit and sample from copulas.

    Example:
        ```python
        from ununseptium.mathstats import CopulaFitter
        import numpy as np

        fitter = CopulaFitter()

        # Generate correlated data
        x = np.random.randn(500)
        y = 0.7 * x + np.sqrt(1 - 0.7**2) * np.random.randn(500)

        # Fit Gaussian copula
        result = fitter.fit(x, y, copula_type=CopulaType.GAUSSIAN)
        print(f"Correlation: {result.correlation}")

        # Sample from copula
        samples = fitter.sample(100)
        ```
    """

    def __init__(self) -> None:
        """Initialize the fitter."""
        self._copula_type: CopulaType | None = None
        self._parameter: float | None = None
        self._marginal_x: Any = None
        self._marginal_y: Any = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        copula_type: CopulaType = CopulaType.GAUSSIAN,
    ) -> CopulaFitResult:
        """Fit a copula to data.

        Args:
            x: First variable.
            y: Second variable.
            copula_type: Type of copula to fit.

        Returns:
            CopulaFitResult with fitted parameters.
        """
        self._copula_type = copula_type

        # Transform to uniform via probability integral transform
        u = sp_stats.rankdata(x) / (len(x) + 1)
        v = sp_stats.rankdata(y) / (len(y) + 1)

        # Store marginal distributions (empirical)
        self._marginal_x = x
        self._marginal_y = y

        if copula_type == CopulaType.GAUSSIAN:
            return self._fit_gaussian(u, v)
        if copula_type == CopulaType.CLAYTON:
            return self._fit_clayton(u, v)
        if copula_type == CopulaType.GUMBEL:
            return self._fit_gumbel(u, v)

        # Default to empirical
        return CopulaFitResult(copula_type=CopulaType.EMPIRICAL)

    def _fit_gaussian(self, u: np.ndarray, v: np.ndarray) -> CopulaFitResult:
        """Fit Gaussian copula."""
        # Transform to normal
        z1 = sp_stats.norm.ppf(u)
        z2 = sp_stats.norm.ppf(v)

        # Correlation parameter
        rho = float(np.corrcoef(z1, z2)[0, 1])
        self._parameter = rho

        # Kendall's tau relationship: tau = (2/pi) * arcsin(rho)
        kendall = (2 / np.pi) * np.arcsin(rho)

        # Log-likelihood
        n = len(u)
        if abs(rho) < 1:
            det = 1 - rho**2
            quad = (z1**2 + z2**2 - 2 * rho * z1 * z2) / det
            ll = -0.5 * n * np.log(det) - 0.5 * np.sum(quad - z1**2 - z2**2)
        else:
            ll = -np.inf

        return CopulaFitResult(
            copula_type=CopulaType.GAUSSIAN,
            parameter=rho,
            correlation=rho,
            kendall_tau=float(kendall),
            log_likelihood=float(ll) if np.isfinite(ll) else None,
            aic=float(-2 * ll + 2) if np.isfinite(ll) else None,
        )

    def _fit_clayton(self, u: np.ndarray, v: np.ndarray) -> CopulaFitResult:
        """Fit Clayton copula."""
        # Use Kendall's tau to estimate theta
        tau, _ = sp_stats.kendalltau(u, v)

        # theta = 2 * tau / (1 - tau) for Clayton
        if tau > 0 and tau < 1:
            theta = 2 * tau / (1 - tau)
        else:
            theta = 0.5

        self._parameter = max(0.01, theta)

        return CopulaFitResult(
            copula_type=CopulaType.CLAYTON,
            parameter=self._parameter,
            kendall_tau=float(tau),
        )

    def _fit_gumbel(self, u: np.ndarray, v: np.ndarray) -> CopulaFitResult:
        """Fit Gumbel copula."""
        # Use Kendall's tau to estimate theta
        tau, _ = sp_stats.kendalltau(u, v)

        # theta = 1 / (1 - tau) for Gumbel
        if tau > 0 and tau < 1:
            theta = 1 / (1 - tau)
        else:
            theta = 1.5

        self._parameter = max(1.0, theta)

        return CopulaFitResult(
            copula_type=CopulaType.GUMBEL,
            parameter=self._parameter,
            kendall_tau=float(tau),
        )

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample from fitted copula.

        Args:
            n: Number of samples.
            seed: Random seed.

        Returns:
            Tuple of (x, y) samples.
        """
        if seed is not None:
            np.random.seed(seed)

        if self._copula_type is None:
            msg = "Copula not fitted. Call fit() first."
            raise ValueError(msg)

        if self._copula_type == CopulaType.GAUSSIAN:
            return self._sample_gaussian(n)

        # Default: independent uniform
        u = np.random.uniform(0, 1, n)
        v = np.random.uniform(0, 1, n)
        return u, v

    def _sample_gaussian(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample from Gaussian copula."""
        rho = self._parameter or 0.0

        # Generate correlated normals
        z1 = np.random.randn(n)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn(n)

        # Transform to uniform
        u = sp_stats.norm.cdf(z1)
        v = sp_stats.norm.cdf(z2)

        return u, v

    def conditional_sample(
        self,
        u_given: float,
        n: int = 1,
    ) -> np.ndarray:
        """Sample V | U = u (conditional sampling).

        Args:
            u_given: Given value of U.
            n: Number of samples.

        Returns:
            Conditional samples of V.
        """
        if self._copula_type != CopulaType.GAUSSIAN:
            # Only implemented for Gaussian
            return np.random.uniform(0, 1, n)

        rho = self._parameter or 0.0
        z1 = sp_stats.norm.ppf(u_given)

        # Conditional distribution: N(rho * z1, 1 - rho^2)
        conditional_mean = rho * z1
        conditional_std = np.sqrt(1 - rho**2)

        z2 = conditional_mean + conditional_std * np.random.randn(n)
        return sp_stats.norm.cdf(z2)
