"""Multivariate non-overlapping batch-means covariance estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import ConfigurationError
from .core import (
    CovarianceProjectionEvidence,
    LongRunCovarianceEstimate,
    LongRunCovarianceEvidence,
    StatisticsError,
    correlated_values_from_factor,
    factor_spectrum,
    validate_histories,
)


@dataclass(frozen=True)
class MultivariateBatchMeans:
    """Multivariate non-overlapping batch-means covariance estimator."""

    batch_size: int

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ConfigurationError("batch_size must be an integer")
        if self.batch_size < 1: raise ConfigurationError("batch_size must be positive")

    def estimate(self, histories: np.ndarray) -> LongRunCovarianceEstimate:
        """Estimate correlated means with non-overlapping batches."""
        return _estimate(histories, self)


def _estimate(
    histories: np.ndarray,
    method: MultivariateBatchMeans,
) -> LongRunCovarianceEstimate:
    values = validate_histories(histories, method_name="batch means")
    configuration_count, value_count = values.shape
    batch_count = configuration_count // method.batch_size
    if batch_count < 2:
        raise StatisticsError("batch_size must leave at least two complete batches")
    used_configuration_count = batch_count * method.batch_size
    discarded_configuration_count = configuration_count - used_configuration_count
    used_values = values[:used_configuration_count]
    batch_means = used_values.reshape(batch_count, method.batch_size, value_count,).mean(axis=1)
    means = batch_means.mean(axis=0)
    centered_batch_means = batch_means - means[np.newaxis, :]
    factor = centered_batch_means / np.sqrt(batch_count * (batch_count - 1.0))
    spectrum = factor_spectrum(factor)

    return LongRunCovarianceEstimate(
        values=correlated_values_from_factor(means, factor),
        evidence=LongRunCovarianceEvidence(
            method=method,
            configuration_count=configuration_count,
            value_count=value_count,
            maximum_lag=None,
            iid_centering_correction=1.0,
            numerical_rank=spectrum.numerical_rank,
            rank_tolerance=spectrum.rank_tolerance,
            covariance=CovarianceProjectionEvidence(
                policy=None,
                projected_mode_count=0,
                minimum_eigenvalue_before=spectrum.minimum_eigenvalue,
                maximum_eigenvalue_before=spectrum.maximum_eigenvalue,
                relative_frobenius_adjustment=0.0,
            ),
            estimator="multivariate-batch-means",
            source="https://doi.org/10.1093/biomet/asz002#equation-10",
            batch_size=method.batch_size,
            batch_count=batch_count,
            discarded_configuration_count=discarded_configuration_count,
            covariance_representation="positive-semidefinite-factor",
        ),
    )
