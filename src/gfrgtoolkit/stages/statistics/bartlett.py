"""Modified-Bartlett multivariate spectral variance estimator

These are some notes from Curtis. Equations annotated below.

Summary
-------

The Bartlett method estimates the full covariance of the sample mean
by summing the same-configuration and lagged cross-configuration covariance while
downweighting more distant configurations. The triangular lag window yields a
positive semi-definite covariance matrix by construction, but correlations beyond
the maximum lag remain unaccounted for.

Details
-------

Every covariance matrix "C" is positive semidefinite; as such, it admits a
factorization
(1) C = F^T F.
For each aligned Monte Carlo configuration s = 0, ..., N - 1, let
Xs be a vector of "p" observed values. Its entries might, for example,
be Yang-Mills energy density measured at several flow times. Define
(2) Zs = Xs - mean(X)
and the unnormalized lag-product matrix
(3) Ak = sum_{s = 0}^{N-k-1} Zs Z_{s+k}^T.
A0 supplies the same-configuration covariance term. A1 measures whether adjacent
configurations fluctuate together, A2 does the same for configurations separated by
two units of Monte Carlo time, and so forth.

For a declared maximum lag "m", let b = m + 1. Extend Zs by zero outside the observed
history and form every overlapping length-"b" sum
(4) Rl = sum_{j=0}^{b-1} Z_{l+j}
with l = -(b - 1), ..., N - 1. The overlap between two such windows produces the
Bartlett weights exactly [1, 2]:
(5) (1 / b) sum_l Rl RRl^T
    = A0 + sum_{k=1}^m wk (Ak + Ak^T)
with wk = 1 - k / b. We estimate the covariance of the sample mean, rather than the
unscaled multivariate long-run covariance [3]. After applying a finite-sample
IID-centering correction
(6) r_Nm = [(N - 1) - (2 / N) sum_{k=1}^m wk (N - k)] / N,
it constructs a matrix ``F`` whose row for window ``l`` is
(7) F[l, :] = R_l^T / sqrt(b N^2 r_Nm).
Consequently,
(8) F^T F = 1 / (N^2 r_Nm)
            * [A0 + sum_{k=1}^m wk (Ak + Ak^T)]
            = estimated Cov(mean(X)).
This is a Gram matrix, so the estimate is positive semidefinite by construction and
needs no after-the-fact eigenvalue repair. Ordinary batch means constructs an
analogous factor from centered batch means [4].

To represent the correlated estimate without necessarily materializing a
dense covariance matrix, let "u" be a vector of independent standard-normal
latent variables and define
(A) x = mean(X) + F^T u.
It follows that
(B) Cov(x) = F^T F.
See `correlated_values_from_factor` in `statistics/core.py` for the implementation
of this latent-factor representation using `gvar` [7]. The optional projected scalar
validation uses Wolff's Gamma method [5, 6].

References
----------
[1] M. S. Bartlett, "Periodogram Analysis and Continuous Spectra,"
    Biometrika 37 (1950) 1--16.
    https://doi.org/10.1093/biomet/37.1-2.1

[2] W. K. Newey and K. D. West, "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix,"
    Econometrica 55 (1987) 703--708.
    https://doi.org/10.2307/1913610

[3] D. Vats, J. M. Flegal, and G. L. Jones, "Strong Consistency of
    Multivariate Spectral Variance Estimators in Markov Chain Monte Carlo,"
    Bernoulli 24 (2018) 1860--1909.
    https://doi.org/10.3150/16-BEJ914

[4] D. Vats, J. M. Flegal, and G. L. Jones, "Multivariate Output Analysis for
    Markov Chain Monte Carlo," Biometrika 106 (2019) 321--337.
    https://doi.org/10.1093/biomet/asz002

[5] U. Wolff, "Monte Carlo Errors with Less Errors," Computer Physics
    Communications 156 (2004) 143--153.
    https://doi.org/10.1016/S0010-4655(03)00467-3

[6] U. Wolff, "Erratum to 'Monte Carlo Errors with Less Errors'," Computer
    Physics Communications 176 (2007) 383.
    https://doi.org/10.1016/j.cpc.2006.12.001

[7] G. P. Lepage, ``gvar`` documentation, "Gaussian Random Variables."
    https://gvar.readthedocs.io/en/latest/gvar.html
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
)
from .core import (
    correlated_values_from_factor,
    factor_spectrum,
    validate_histories,
)
from .gamma import ProjectedWolffValidation, validate_projected_wolff


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

    # Eqn 2: center each configuration vector about the sample mean.
    means = values.mean(axis=0)
    centered = values - means[np.newaxis, :]

    def factor_at(maximum_lag: int) -> tuple[np.ndarray, float]:
        # Eqns 5 & 6: use the Bartlett weights in the exact
        # finite-sample correction for centering an otherwise IID history.
        weighted_pair_count = 0.0
        for lag in range(1, maximum_lag + 1):
            weight = 1.0 - lag / (maximum_lag + 1.0)
            weighted_pair_count += weight * (configuration_count - lag)
        iid_bias_factor = (
            (configuration_count - 1.0)
            - 2.0 * weighted_pair_count / configuration_count
        ) / configuration_count
        bandwidth = maximum_lag + 1

        # Eqn 4: construct every length-b rolling sum R_l. Padding by
        # b - 1 realizes the declared zero extension at both boundaries.
        padded = np.pad(centered, ((bandwidth - 1, bandwidth - 1), (0, 0)),)
        cumulative = np.vstack((np.zeros((1, value_count)), np.cumsum(padded, axis=0)))
        rolling_sums = cumulative[bandwidth:] - cumulative[:-bandwidth]

        # Eqns 3, 5, and 7: overlapping R_l vectors encode the
        # lag products A_k and their Bartlett weights without constructing
        # those matrices explicitly; normalization makes each row a row of F.
        factor = rolling_sums / np.sqrt(
            bandwidth
            * configuration_count
            * configuration_count
            * iid_bias_factor
        )
        return factor, iid_bias_factor

    factor, iid_bias_factor = factor_at(method.maximum_lag)

    # Eqn 8: the selected covariance is F^T F. Its diagonal can be
    # calculated without materializing the full covariance matrix.
    selected_variances = np.sum(factor * factor, axis=0)
    bandwidth_scan_lags: tuple[int, ...] = ()
    bandwidth_comparisons: list[BandwidthComparisonEvidence] = []
    maximum_relative_variance_change: float | None = None
    autocorrelation_status = AutocorrelationResolutionStatus.NotAssessed
    autocorrelation_diagnostics: list[str] = []

    if method.stability is not None:
        autocorrelation_status = AutocorrelationResolutionStatus.Resolved
        for comparison_lag in method.stability.comparison_lags:
            # Re-evaluate Equations (4)--(8) at each comparison bandwidth.
            comparison_factor, _ = factor_at(comparison_lag)
            comparison_variances = np.sum(comparison_factor * comparison_factor, axis=0,)
            scale = np.maximum(
                np.maximum(np.abs(selected_variances), np.abs(comparison_variances),),
                np.finfo(float).tiny,
            )
            bandwidth_comparisons.append(
                BandwidthComparisonEvidence(
                    comparison_lag=comparison_lag,
                    selected_lag=method.maximum_lag,
                    maximum_relative_variance_change=float(
                        np.max(np.abs(selected_variances - comparison_variances) / scale)
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
        # Equations (1), (8), (A), and (B): expose mean(X) and F through
        # correlated values whose covariance is exactly the Bartlett F^T F.
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
