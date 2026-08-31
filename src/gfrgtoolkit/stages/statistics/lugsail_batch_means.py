"""Positive-leading-bias lugsail batch-means covariance estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import ConfigurationError
from .core import (
    CovarianceProjection,
    CovarianceProjectionEvidence,
    LongRunCovarianceEstimate,
    LongRunCovarianceEvidence,
    StatisticsError,
    correlated_values_from_factor,
    validate_histories,
)


@dataclass(frozen=True)
class LugsailBatchMeans:
    """Positive-leading-bias lugsail batch-means systematic variation."""

    batch_size: int
    lugsail_scale: int = 3
    lugsail_weight: float = 0.5
    covariance_projection: CovarianceProjection = CovarianceProjection.NearestPSD

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ConfigurationError("batch_size must be an integer")
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be positive")
        if isinstance(self.lugsail_scale, bool) or not isinstance(self.lugsail_scale, int):
            raise ConfigurationError("lugsail_scale must be an integer")
        if self.lugsail_scale < 2:
            raise ConfigurationError("lugsail_scale must be at least two")
        if self.batch_size % self.lugsail_scale != 0:
            raise ConfigurationError("batch_size must be divisible by lugsail_scale")
        if (
            not np.isfinite(self.lugsail_weight)
            or self.lugsail_weight <= 0.0
            or self.lugsail_weight >= 1.0
        ): raise ConfigurationError("lugsail_weight must satisfy 0 < lugsail_weight < 1")
        if self.lugsail_weight <= 1.0 / self.lugsail_scale:
            raise ConfigurationError("over-lugsail requires lugsail_weight > 1 / lugsail_scale")
        if not isinstance(self.covariance_projection, CovarianceProjection):
            raise ConfigurationError("lugsail covariance_projection must be a CovarianceProjection")

    def estimate(self, histories: np.ndarray) -> LongRunCovarianceEstimate:
        """Estimate the configured over-lugsail systematic variation."""
        return _estimate(histories, self)


def _estimate(
    histories: np.ndarray,
    method: LugsailBatchMeans,
) -> LongRunCovarianceEstimate:
    values = validate_histories(histories, method_name="lugsail batch means")
    configuration_count, value_count = values.shape
    smaller_batch_size = method.batch_size // method.lugsail_scale
    batch_count = configuration_count // method.batch_size
    if batch_count < 2:
        raise StatisticsError("batch_size must leave at least two complete batches")
    used_configuration_count = batch_count * method.batch_size
    used_values = values[:used_configuration_count]
    means = used_values.mean(axis=0)

    def factor_for_batch_size(batch_size: int) -> np.ndarray:
        count = used_configuration_count // batch_size
        batch_means = used_values.reshape(count, batch_size, value_count,).mean(axis=1)
        centered = batch_means - means[np.newaxis, :]
        return centered / np.sqrt(count * (count - 1.0))

    large_factor = factor_for_batch_size(method.batch_size)
    small_factor = factor_for_batch_size(smaller_batch_size)
    positive_scale = 1.0 / (1.0 - method.lugsail_weight)
    negative_scale = method.lugsail_weight / (1.0 - method.lugsail_weight)
    stacked = np.vstack(
        (
            np.sqrt(positive_scale) * large_factor,
            np.sqrt(negative_scale) * small_factor,
        )
    )
    signs = np.concatenate(
        (
            np.ones(large_factor.shape[0]),
            -np.ones(small_factor.shape[0]),
        )
    )
    left, singular_values, right = np.linalg.svd(stacked, full_matrices=False)
    singular_tolerance = (
        max(stacked.shape)
        * np.finfo(float).eps
        * (float(singular_values[0]) if len(singular_values) else 0.0)
    )
    supported = singular_values > singular_tolerance
    left = left[:, supported]
    singular_values = singular_values[supported]
    right = right[supported]
    signed_metric = left.T @ (signs[:, None] * left)
    reduced_covariance = (
        singular_values[:, None]
        * signed_metric
        * singular_values[None, :]
    )
    reduced_covariance = 0.5 * (
        reduced_covariance + reduced_covariance.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(reduced_covariance)
    rank_tolerance = (
        value_count
        * np.finfo(float).eps
        * (float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0.0)
    )
    negative = eigenvalues < 0.0
    projected_mode_count = int(np.count_nonzero(negative))
    projected_eigenvalues = np.maximum(eigenvalues, 0.0)
    spectrum_norm = float(np.linalg.norm(eigenvalues))
    adjustment = float(
        np.linalg.norm(projected_eigenvalues - eigenvalues) / spectrum_norm
        if spectrum_norm
        else 0.0
    )
    projected_factor = (
        np.sqrt(projected_eigenvalues)[:, None]
        * (eigenvectors.T @ right)
    )

    return LongRunCovarianceEstimate(
        values=correlated_values_from_factor(means, projected_factor),
        evidence=LongRunCovarianceEvidence(
            method=method,
            configuration_count=configuration_count,
            value_count=value_count,
            maximum_lag=None,
            iid_centering_correction=1.0,
            numerical_rank=int(
                np.count_nonzero(projected_eigenvalues > rank_tolerance)
            ),
            rank_tolerance=rank_tolerance,
            covariance=CovarianceProjectionEvidence(
                policy=method.covariance_projection,
                projected_mode_count=projected_mode_count,
                minimum_eigenvalue_before=min(
                    0.0,
                    float(eigenvalues[0]) if len(eigenvalues) else 0.0,
                ),
                maximum_eigenvalue_before=max(
                    0.0,
                    float(eigenvalues[-1]) if len(eigenvalues) else 0.0,
                ),
                relative_frobenius_adjustment=adjustment,
                implementation="symmetric-covariance-eigenvalue-clipping",
            ),
            estimator="over-lugsail-batch-means",
            source="https://doi.org/10.1093/biomet/asab049#equation-7",
            batch_size=method.batch_size,
            batch_count=batch_count,
            discarded_configuration_count=(
                configuration_count - used_configuration_count
            ),
            first_order_bias="positive-for-positive-correlation",
            covariance_representation="projected-low-rank-factor",
        ),
    )
