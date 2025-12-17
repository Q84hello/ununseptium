"""Extreme Value Theory for tail risk analysis.

Provides GPD fitting and tail risk scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as sp_stats


class GPDFit(BaseModel):
    """Generalized Pareto Distribution fit result.

    Attributes:
        shape: Shape parameter (xi).
        scale: Scale parameter (sigma).
        threshold: Threshold used for POT.
        n_exceedances: Number of exceedances.
        aic: Akaike Information Criterion.
    """

    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    aic: float | None = None


class TailRiskScore(BaseModel):
    """Tail risk score result.

    Attributes:
        var: Value at Risk at specified quantile.
        es: Expected Shortfall.
        quantile: Quantile level.
        exceedance_prob: Probability of exceeding threshold.
    """

    var: float
    es: float
    quantile: float = Field(ge=0.0, le=1.0)
    exceedance_prob: float = Field(ge=0.0, le=1.0)


class EVTAnalyzer:
    """Extreme Value Theory analyzer.

    Uses Peaks Over Threshold (POT) with GPD for tail risk.

    Mathematical Foundation:
        For exceedances over threshold u, the GPD is:
        F(x) = 1 - (1 + xi*x/sigma)^(-1/xi)

        VaR_p = u + (sigma/xi)*((n/k*(1-p))^(-xi) - 1)
        ES_p = VaR_p/(1-xi) + (sigma - xi*u)/(1-xi)

    Example:
        ```python
        from ununseptium.mathstats import EVTAnalyzer
        import numpy as np

        analyzer = EVTAnalyzer()

        # Fit GPD to loss data
        losses = np.random.pareto(2, 1000)
        fit = analyzer.fit_gpd(losses, threshold_quantile=0.9)

        # Compute tail risk
        risk = analyzer.tail_risk(fit, quantile=0.99)
        print(f"VaR(99%): {risk.var}")
        ```
    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        pass

    def fit_gpd(
        self,
        data: np.ndarray,
        *,
        threshold: float | None = None,
        threshold_quantile: float = 0.9,
    ) -> GPDFit:
        """Fit Generalized Pareto Distribution.

        Args:
            data: Observation data.
            threshold: Fixed threshold (overrides quantile).
            threshold_quantile: Quantile for threshold selection.

        Returns:
            GPDFit with estimated parameters.
        """
        if threshold is None:
            threshold = float(np.quantile(data, threshold_quantile))

        # Extract exceedances
        exceedances = data[data > threshold] - threshold
        n_exceedances = len(exceedances)

        if n_exceedances < 10:
            # Insufficient data, use moment estimator
            mean_exc = np.mean(exceedances)
            var_exc = np.var(exceedances)

            # Method of moments
            shape = 0.5 * (1 - mean_exc**2 / var_exc) if var_exc > 0 else 0.0
            scale = mean_exc * (1 - shape) if shape < 1 else mean_exc

            return GPDFit(
                shape=float(shape),
                scale=float(max(scale, 1e-6)),
                threshold=threshold,
                n_exceedances=n_exceedances,
            )

        # MLE fit using scipy
        shape, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)

        # Compute AIC
        log_lik = np.sum(sp_stats.genpareto.logpdf(exceedances, shape, 0, scale))
        aic = 2 * 2 - 2 * log_lik  # 2 parameters

        return GPDFit(
            shape=float(shape),
            scale=float(scale),
            threshold=threshold,
            n_exceedances=n_exceedances,
            aic=float(aic),
        )

    def tail_risk(
        self,
        fit: GPDFit,
        quantile: float = 0.99,
        n_obs: int | None = None,
    ) -> TailRiskScore:
        """Compute tail risk measures.

        Args:
            fit: GPD fit result.
            quantile: VaR quantile level.
            n_obs: Total number of observations (for exceedance adjustment).

        Returns:
            TailRiskScore with VaR and ES.
        """
        xi = fit.shape
        sigma = fit.scale
        u = fit.threshold

        # Exceedance rate
        if n_obs is not None and n_obs > 0:
            prob_exceed = fit.n_exceedances / n_obs
        else:
            prob_exceed = 1 - quantile

        # VaR calculation
        if abs(xi) < 1e-10:
            # Exponential case (xi -> 0)
            var = u + sigma * np.log(prob_exceed / (1 - quantile))
        else:
            var = u + (sigma / xi) * ((prob_exceed / (1 - quantile)) ** (-xi) - 1)

        # Expected Shortfall
        if xi < 1:
            es = var / (1 - xi) + (sigma - xi * u) / (1 - xi)
        else:
            es = np.inf  # Undefined for xi >= 1

        return TailRiskScore(
            var=float(var),
            es=float(es) if np.isfinite(es) else float("inf"),
            quantile=quantile,
            exceedance_prob=float(prob_exceed),
        )

    def threshold_selection(
        self,
        data: np.ndarray,
        quantile_range: tuple[float, float] = (0.8, 0.98),
        n_points: int = 20,
    ) -> dict[str, Any]:
        """Analyze threshold selection for GPD.

        Args:
            data: Observation data.
            quantile_range: Range of quantiles to test.
            n_points: Number of threshold points.

        Returns:
            Dictionary with threshold analysis results.
        """
        quantiles = np.linspace(*quantile_range, n_points)
        results = []

        for q in quantiles:
            try:
                fit = self.fit_gpd(data, threshold_quantile=q)
                results.append(
                    {
                        "quantile": float(q),
                        "threshold": fit.threshold,
                        "shape": fit.shape,
                        "scale": fit.scale,
                        "n_exceedances": fit.n_exceedances,
                        "aic": fit.aic,
                    }
                )
            except (ValueError, RuntimeError):
                continue

        # Find optimal threshold (stable shape parameter)
        if len(results) >= 3:
            shapes = np.array([r["shape"] for r in results])
            # Look for stability (minimal derivative)
            shape_diffs = np.abs(np.diff(shapes))
            stable_idx = int(np.argmin(shape_diffs)) + 1
            optimal = results[stable_idx]
        else:
            optimal = results[-1] if results else None

        return {
            "analysis": results,
            "recommended": optimal,
        }

    def exceedance_probability(
        self,
        fit: GPDFit,
        value: float,
    ) -> float:
        """Compute probability of exceeding a value.

        Args:
            fit: GPD fit result.
            value: Value to compute exceedance probability for.

        Returns:
            Exceedance probability.
        """
        if value <= fit.threshold:
            return 1.0

        x = value - fit.threshold
        xi = fit.shape
        sigma = fit.scale

        if abs(xi) < 1e-10:
            # Exponential case
            return float(np.exp(-x / sigma))

        if 1 + xi * x / sigma <= 0:
            return 0.0

        return float((1 + xi * x / sigma) ** (-1 / xi))
