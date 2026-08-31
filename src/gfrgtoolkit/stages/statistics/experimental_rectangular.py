"""Legacy agent-generated rectangular lag-window covariance heuristic."""

from __future__ import annotations

from dataclasses import dataclass

import gvar as gv
import numpy as np

from ...errors import ConfigurationError
from .core import (
    CovarianceProjection,
    CovarianceProjectionEvidence,
    LongRunCovarianceEstimate,
    LongRunCovarianceEvidence,
    validate_histories,
)


@dataclass(frozen=True)
class ExperimentalRectangularLongRunCovariance:
    """Legacy lag-window heuristic retained for differential comparison."""

    window_factor: float = 3.0
    covariance_projection: CovarianceProjection = CovarianceProjection.NearestPSD

    def __post_init__(self) -> None:
        if not np.isfinite(self.window_factor) or self.window_factor <= 0.0:
            raise ConfigurationError("experimental window_factor must be positive and finite")
        if not isinstance(self.covariance_projection, CovarianceProjection):
            raise ConfigurationError("experimental covariance_projection must be a CovarianceProjection")

    def estimate(self, histories: np.ndarray) -> LongRunCovarianceEstimate:
        """Estimate correlated means with the experimental heuristic."""
        return _estimate(histories, self)


def _estimate(
    histories: np.ndarray,
    method: ExperimentalRectangularLongRunCovariance,
) -> LongRunCovarianceEstimate:
    """Return a projected rectangular lag-sum estimate."""
    values = validate_histories(histories, method_name="experimental rectangular covariance",)
    configuration_count, value_count = values.shape
    means = values.mean(axis=0)
    centered = values - means[np.newaxis, :]
    variances = np.sum(centered * centered, axis=0) / configuration_count
    safe_variances = np.where(variances > 0.0, variances, 1.0)

    integrated_times = np.full(value_count, 0.5)
    maximum_probe = min(configuration_count // 4, 1000)
    for lag in range(1, maximum_probe):
        correlations = (
            np.sum(centered[:-lag] * centered[lag:], axis=0)
            / (configuration_count * safe_variances)
        )
        active = (correlations > 0.0) & (
            lag < method.window_factor * integrated_times
        )
        integrated_times = np.where(
            active,
            integrated_times + correlations,
            integrated_times,
        )
        if not active.any():
            break

    window = max(4, int(np.ceil(method.window_factor * float(integrated_times.max()))),)
    window = min(window, configuration_count // 4)

    covariance_sum = centered.T @ centered
    for lag in range(1, window + 1):
        lagged = centered[:-lag].T @ centered[lag:]
        covariance_sum += lagged + lagged.T
    covariance = covariance_sum / (configuration_count * configuration_count)
    covariance = 0.5 * (covariance + covariance.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    projected_eigenvalues = np.maximum(eigenvalues, 0.0)
    covariance = (eigenvectors * projected_eigenvalues) @ eigenvectors.T
    spectrum_norm = float(np.linalg.norm(eigenvalues))
    adjustment_norm = float(np.linalg.norm(projected_eigenvalues - eigenvalues))
    rank_tolerance = (
        value_count
        * np.finfo(float).eps
        * float(np.max(np.abs(projected_eigenvalues)))
    )
    evidence = LongRunCovarianceEvidence(
        method=method,
        configuration_count=configuration_count,
        value_count=value_count,
        maximum_lag=window,
        iid_centering_correction=1.0,
        numerical_rank=int(
            np.count_nonzero(projected_eigenvalues > rank_tolerance)
        ),
        rank_tolerance=rank_tolerance,
        covariance=CovarianceProjectionEvidence(
            policy=method.covariance_projection,
            projected_mode_count=int(np.count_nonzero(eigenvalues < 0.0)),
            minimum_eigenvalue_before=float(eigenvalues[0]),
            maximum_eigenvalue_before=float(eigenvalues[-1]),
            relative_frobenius_adjustment=(
                adjustment_norm / spectrum_norm if spectrum_norm else 0.0
            ),
            implementation="symmetric-covariance-eigenvalue-clipping",
        ),
        estimator="experimental-rectangular-lag-sum",
        source=None,
        covariance_representation="dense-projected-covariance",
    )
    return LongRunCovarianceEstimate(
        values=np.asarray(gv.gvar(means, covariance), dtype=object),
        evidence=evidence,
    )


# Transitional import alias for early notebooks. This is not Wolff's method.
GammaMethod = ExperimentalRectangularLongRunCovariance
