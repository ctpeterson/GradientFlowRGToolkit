"""PSD Bartlett/Newey--West long-run covariance estimation.

The covariance can be written
(1) C = F^T F.
Bartlett and ordinary batch means calculate F directly, as opposed to forming an
indefinite matrix and repairing it afterwards.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import ConfigurationError
from .core import (
    AutocorrelationResolutionEvidence,
    AutocorrelationResolutionStatus,
    BandwidthComparisonEvidence,
    CovarianceProjectionEvidence,
    LongRunCovarianceEstimate,
    LongRunCovarianceEvidence,
    StatisticsError,
    UnresolvedAutocorrelation,
    UnresolvedAutocorrelationAction,
    correlated_values_from_factor,
    factor_spectrum,
    validate_histories,
)
from .wolff import ProjectedWolffValidation, validate_projected_wolff


@dataclass(frozen=True)
class BandwidthStabilityCheck:
    """Declared comparison grid and tolerance for a Bartlett plateau check."""

    comparison_lags: tuple[int, ...]
    relative_tolerance: float

    def __post_init__(self) -> None:
        lags = tuple(self.comparison_lags)
        if not lags: raise ConfigurationError("comparison_lags must not be empty")
        if any(
            isinstance(lag, bool) or not isinstance(lag, int) or lag < 0
            for lag in lags
        ): raise ConfigurationError("comparison_lags must contain non-negative integers")
        if tuple(sorted(set(lags))) != lags:
            raise ConfigurationError("comparison_lags must be strictly increasing")
        if (not np.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0):
            raise ConfigurationError("relative_tolerance must be positive and finite")
        object.__setattr__(self, "comparison_lags", lags)


@dataclass(frozen=True)
class BartlettLongRunCovariance:
    """Bartlett/Newey--West long-run covariance with an explicit lag limit."""

    maximum_lag: int
    stability: BandwidthStabilityCheck | None = None
    wolff_validation: ProjectedWolffValidation | None = None
    on_unresolved: UnresolvedAutocorrelationAction = (UnresolvedAutocorrelationAction.Record)

    def __post_init__(self) -> None:
        if isinstance(self.maximum_lag, bool) or not isinstance(self.maximum_lag, int):
            raise ConfigurationError("maximum_lag must be an integer")
        if self.maximum_lag < 0:
            raise ConfigurationError("maximum_lag must be non-negative")
        if self.stability is not None:
            if not isinstance(self.stability, BandwidthStabilityCheck):
                raise ConfigurationError("stability must be a BandwidthStabilityCheck")
            if self.stability.comparison_lags[-1] >= self.maximum_lag:
                raise ConfigurationError("comparison lags must be smaller than maximum_lag")
        if self.wolff_validation is not None and not isinstance(
            self.wolff_validation,
            ProjectedWolffValidation,
        ): raise ConfigurationError("wolff_validation must be a ProjectedWolffValidation")
        if not isinstance(self.on_unresolved, UnresolvedAutocorrelationAction):
            raise ConfigurationError("on_unresolved must be an UnresolvedAutocorrelationAction")

    def estimate(self, histories: np.ndarray) -> LongRunCovarianceEstimate:
        """Estimate correlated means with the configured Bartlett window."""
        return _estimate(histories, self)


def _estimate(
    histories: np.ndarray,
    method: BartlettLongRunCovariance,
) -> LongRunCovarianceEstimate:
    values = validate_histories(histories, method_name="Bartlett covariance")
    configuration_count, value_count = values.shape
    if method.maximum_lag >= configuration_count:
        raise StatisticsError("maximum_lag must be smaller than the configuration count")

    means = values.mean(axis=0)
    centered = values - means[np.newaxis, :]

    def factor_at(maximum_lag: int) -> tuple[np.ndarray, float]:
        weighted_pair_count = 0.0
        for lag in range(1, maximum_lag + 1):
            weight = 1.0 - lag / (maximum_lag + 1.0)
            weighted_pair_count += weight * (configuration_count - lag)
        iid_bias_factor = (
            (configuration_count - 1.0)
            - 2.0 * weighted_pair_count / configuration_count
        ) / configuration_count
        bandwidth = maximum_lag + 1
        padded = np.pad(centered, ((bandwidth - 1, bandwidth - 1), (0, 0)),)
        cumulative = np.vstack((np.zeros((1, value_count)), np.cumsum(padded, axis=0)))
        rolling_sums = cumulative[bandwidth:] - cumulative[:-bandwidth]
        factor = rolling_sums / np.sqrt(
            bandwidth
            * configuration_count
            * configuration_count
            * iid_bias_factor
        )
        return factor, iid_bias_factor

    factor, iid_bias_factor = factor_at(method.maximum_lag)
    selected_variances = np.sum(factor * factor, axis=0)
    bandwidth_scan_lags: tuple[int, ...] = ()
    bandwidth_comparisons: list[BandwidthComparisonEvidence] = []
    maximum_relative_variance_change: float | None = None
    autocorrelation_status = AutocorrelationResolutionStatus.NotAssessed
    autocorrelation_diagnostics: list[str] = []

    if method.stability is not None:
        autocorrelation_status = AutocorrelationResolutionStatus.Resolved
        for comparison_lag in method.stability.comparison_lags:
            comparison_factor, _ = factor_at(comparison_lag)
            comparison_variances = np.sum(
                comparison_factor * comparison_factor,
                axis=0,
            )
            scale = np.maximum(
                np.maximum(
                    np.abs(selected_variances),
                    np.abs(comparison_variances),
                ),
                np.finfo(float).tiny,
            )
            bandwidth_comparisons.append(
                BandwidthComparisonEvidence(
                    comparison_lag=comparison_lag,
                    selected_lag=method.maximum_lag,
                    maximum_relative_variance_change=float(
                        np.max(
                            np.abs(
                                selected_variances - comparison_variances
                            )
                            / scale
                        )
                    ),
                )
            )
        worst_comparison = max(
            bandwidth_comparisons,
            key=lambda item: item.maximum_relative_variance_change,
        )
        maximum_relative_variance_change = (
            worst_comparison.maximum_relative_variance_change
        )
        bandwidth_scan_lags = (*method.stability.comparison_lags, method.maximum_lag,)
        if maximum_relative_variance_change > method.stability.relative_tolerance:
            autocorrelation_status = AutocorrelationResolutionStatus.Unresolved
            diagnostic = (
                "Bartlett bandwidth sensitivity "
                f"{maximum_relative_variance_change:.3g} exceeds declared "
                f"tolerance {method.stability.relative_tolerance:.3g} between "
                f"lags {worst_comparison.comparison_lag} and "
                f"{method.maximum_lag}"
            )
            autocorrelation_diagnostics.append(diagnostic)
            if method.on_unresolved is UnresolvedAutocorrelationAction.Raise:
                raise UnresolvedAutocorrelation(diagnostic)

    wolff_evidence = None
    if method.wolff_validation is not None:
        wolff_evidence = validate_projected_wolff(
            centered,
            factor,
            method.wolff_validation,
        )
        unresolved_wolff_count = (
            wolff_evidence.unresolved_coordinate_count
            + wolff_evidence.unresolved_declared_projection_count
        )
        if unresolved_wolff_count:
            autocorrelation_status = AutocorrelationResolutionStatus.Unresolved
            diagnostic = (
                "Wolff automatic window unresolved for "
                f"{wolff_evidence.unresolved_coordinate_count} coordinates and "
                f"{wolff_evidence.unresolved_declared_projection_count} "
                "declared projections at "
                f"declared lag cap {wolff_evidence.maximum_lag}"
            )
            autocorrelation_diagnostics.append(diagnostic)
            if method.on_unresolved is UnresolvedAutocorrelationAction.Raise:
                raise UnresolvedAutocorrelation(diagnostic)
        elif (
            method.wolff_validation.relative_variance_tolerance is not None
            and wolff_evidence.maximum_relative_variance_difference
            is not None
            and wolff_evidence.maximum_relative_variance_difference
            > method.wolff_validation.relative_variance_tolerance
        ):
            autocorrelation_status = AutocorrelationResolutionStatus.Unresolved
            diagnostic = (
                "Wolff variance disagreement "
                f"{wolff_evidence.maximum_relative_variance_difference:.3g} "
                "exceeds declared tolerance "
                f"{method.wolff_validation.relative_variance_tolerance:.3g}"
            )
            autocorrelation_diagnostics.append(diagnostic)
            if method.on_unresolved is UnresolvedAutocorrelationAction.Raise:
                raise UnresolvedAutocorrelation(diagnostic)
        elif autocorrelation_status is AutocorrelationResolutionStatus.NotAssessed:
            autocorrelation_status = AutocorrelationResolutionStatus.Resolved

    spectrum = factor_spectrum(factor)
    return LongRunCovarianceEstimate(
        values=correlated_values_from_factor(means, factor),
        evidence=LongRunCovarianceEvidence(
            method=method,
            configuration_count=configuration_count,
            value_count=value_count,
            maximum_lag=method.maximum_lag,
            iid_centering_correction=1.0 / iid_bias_factor,
            numerical_rank=spectrum.numerical_rank,
            rank_tolerance=spectrum.rank_tolerance,
            covariance=CovarianceProjectionEvidence(
                policy=None,
                projected_mode_count=0,
                minimum_eigenvalue_before=spectrum.minimum_eigenvalue,
                maximum_eigenvalue_before=spectrum.maximum_eigenvalue,
                relative_frobenius_adjustment=0.0,
            ),
            estimator="bartlett-newey-west",
            source="https://doi.org/10.2307/1913610#equation-5",
            autocorrelation=AutocorrelationResolutionEvidence(
                status=autocorrelation_status,
                diagnostics=tuple(autocorrelation_diagnostics),
            ),
            bandwidth_scan_lags=bandwidth_scan_lags,
            bandwidth_comparisons=tuple(bandwidth_comparisons),
            maximum_relative_variance_change=maximum_relative_variance_change,
            wolff_validation=wolff_evidence,
            covariance_representation="positive-semidefinite-factor",
        ),
    )
