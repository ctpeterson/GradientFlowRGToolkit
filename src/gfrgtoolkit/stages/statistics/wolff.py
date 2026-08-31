"""Projected scalar validation using Wolff's published Gamma method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import ConfigurationError
from .core import StatisticsError


@dataclass(frozen=True)
class ProjectedWolffValidation:
    """Scalar Wolff Gamma-method validation applied to every coordinate."""

    exponential_scale: float
    maximum_lag: int
    relative_variance_tolerance: float | None = None
    projections: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.exponential_scale)
            or self.exponential_scale < 1.0
            or self.exponential_scale > 2.0
        ): raise ConfigurationError("Wolff exponential_scale must satisfy 1 <= scale <= 2")
        if isinstance(self.maximum_lag, bool) or not isinstance(self.maximum_lag, int):
            raise ConfigurationError("Wolff maximum_lag must be an integer")
        if self.maximum_lag < 1:
            raise ConfigurationError("Wolff maximum_lag must be positive")
        if self.relative_variance_tolerance is not None and (
            not np.isfinite(self.relative_variance_tolerance)
            or self.relative_variance_tolerance <= 0.0
        ):
            raise ConfigurationError(
                "Wolff relative_variance_tolerance must be positive and finite"
            )
        projections = tuple(
            tuple(float(coefficient) for coefficient in projection)
            for projection in self.projections
        )
        if any(not projection for projection in projections):
            raise ConfigurationError("Wolff projections must not be empty")
        if projections and len({len(projection) for projection in projections}) != 1:
            raise ConfigurationError("Wolff projections must have equal dimensions")
        if any(
            not np.all(np.isfinite(projection))
            or not np.any(np.asarray(projection) != 0.0)
            for projection in projections
        ):
            raise ConfigurationError(
                "Wolff projections must be finite and nonzero"
            )
        object.__setattr__(self, "projections", projections)


@dataclass(frozen=True)
class ProjectedWolffEvidence:
    source: str
    exponential_scale: float
    maximum_lag: int
    minimum_selected_window: int | None
    maximum_selected_window: int | None
    maximum_relative_variance_difference: float | None
    unresolved_coordinate_count: int
    relative_variance_tolerance: float | None
    assessed_coordinate_count: int
    declared_projection_count: int
    unresolved_declared_projection_count: int


def validate_projected_wolff(
    centered: np.ndarray,
    selected_factor: np.ndarray,
    validation: ProjectedWolffValidation,
) -> ProjectedWolffEvidence:
    """Compare coordinate variances with Wolff's scalar automatic window."""
    configuration_count, coordinate_count = centered.shape
    if validation.projections:
        projection_matrix = np.asarray(validation.projections, dtype=float)
        if projection_matrix.shape[1] != coordinate_count:
            raise StatisticsError(
                "Wolff projection dimension must match the history value count"
            )
        assessment_matrix = np.vstack(
            (np.eye(coordinate_count), projection_matrix)
        )
        centered = centered @ assessment_matrix.T
        selected_factor = selected_factor @ assessment_matrix.T
    value_count = centered.shape[1]
    selected_variances = np.sum(selected_factor * selected_factor, axis=0)
    maximum_lag = min(validation.maximum_lag, configuration_count - 1)
    gamma_zero = np.sum(centered * centered, axis=0) / configuration_count
    covariance_sums = gamma_zero.copy()
    selected_windows = np.full(value_count, -1, dtype=int)
    wolff_variances = np.zeros(value_count)
    constant = gamma_zero == 0.0
    selected_windows[constant] = 0

    for lag in range(1, maximum_lag + 1):
        lag_covariance = (
            np.sum(centered[:-lag] * centered[lag:], axis=0)
            / (configuration_count - lag)
        )
        covariance_sums += 2.0 * lag_covariance
        integrated_times = np.divide(
            covariance_sums,
            2.0 * gamma_zero,
            out=np.full(value_count, 0.5),
            where=gamma_zero > 0.0,
        )
        effective_times = np.full(value_count, np.finfo(float).tiny)
        positive_time = integrated_times > 0.5
        effective_times[positive_time] = (
            validation.exponential_scale
            / np.log(
                (2.0 * integrated_times[positive_time] + 1.0)
                / (2.0 * integrated_times[positive_time] - 1.0)
            )
        )
        window_function = (
            np.exp(-lag / effective_times)
            - effective_times / np.sqrt(lag * configuration_count)
        )
        newly_selected = (selected_windows < 0) & (window_function < 0.0)
        if np.any(newly_selected):
            selected_windows[newly_selected] = lag
            wolff_variances[newly_selected] = (
                covariance_sums[newly_selected]
                * (1.0 + (2.0 * lag + 1.0) / configuration_count)
                / configuration_count
            )
        if np.all(selected_windows >= 0): break

    unresolved_coordinate_count = int(
        np.count_nonzero(selected_windows[:coordinate_count] < 0)
    )
    unresolved_declared_projection_count = int(
        np.count_nonzero(selected_windows[coordinate_count:] < 0)
    )
    unresolved_count = (
        unresolved_coordinate_count + unresolved_declared_projection_count
    )
    resolved_windows = selected_windows[selected_windows >= 0]
    maximum_difference = None
    if not unresolved_count:
        variance_scale = np.maximum(
            np.maximum(np.abs(selected_variances), np.abs(wolff_variances)),
            np.finfo(float).tiny,
        )
        maximum_difference = float(
            np.max(np.abs(selected_variances - wolff_variances) / variance_scale)
        )
    return ProjectedWolffEvidence(
        source="https://doi.org/10.1016/S0010-4655(03)00467-3",
        exponential_scale=validation.exponential_scale,
        maximum_lag=maximum_lag,
        minimum_selected_window=(
            int(np.min(resolved_windows)) if len(resolved_windows) else None
        ),
        maximum_selected_window=(
            int(np.max(resolved_windows)) if len(resolved_windows) else None
        ),
        maximum_relative_variance_difference=maximum_difference,
        unresolved_coordinate_count=unresolved_coordinate_count,
        relative_variance_tolerance=validation.relative_variance_tolerance,
        assessed_coordinate_count=coordinate_count,
        declared_projection_count=len(validation.projections),
        unresolved_declared_projection_count=(
            unresolved_declared_projection_count
        ),
    )
