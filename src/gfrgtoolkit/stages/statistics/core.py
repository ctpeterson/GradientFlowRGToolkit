"""Shared interface and evidence values for long-run covariance methods."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import gvar as gv
import numpy as np

if TYPE_CHECKING:
    from .gamma import ProjectedWolffEvidence


class StatisticsError(ValueError):
    """Raised when a Monte Carlo history cannot produce an estimate."""


class UnresolvedAutocorrelation(StatisticsError):
    """Raised when declared diagnostics do not resolve the correlation tail."""


class AutocorrelationResolutionStatus(Enum):
    """Whether declared diagnostics resolved the observed correlation tail."""

    NotAssessed = "not-assessed"
    Resolved = "resolved"
    Unresolved = "unresolved"


class UnresolvedAutocorrelationAction(Enum):
    """Action to take when declared diagnostics cannot resolve a tail."""

    Record = "record"
    Raise = "raise"


@runtime_checkable
class LongRunCovarianceMethod(Protocol):
    """Interface implemented by every long-run covariance method."""

    def estimate(self, histories: np.ndarray) -> LongRunCovarianceEstimate:
        """Estimate correlated means from aligned configuration histories."""


@dataclass(frozen=True)
class CovarianceProjectionEvidence:
    """Magnitude of a declared repair applied to an estimated covariance."""

    policy: Literal["nearest-positive-semidefinite"] | None
    projected_mode_count: int
    minimum_eigenvalue_before: float
    maximum_eigenvalue_before: float
    relative_frobenius_adjustment: float
    implementation: str = "not-applied"

    @property
    def applied(self) -> bool: return self.projected_mode_count > 0


@dataclass(frozen=True)
class AutocorrelationResolutionEvidence:
    """Outcome and explanations from declared autocorrelation diagnostics."""

    status: AutocorrelationResolutionStatus
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class BandwidthComparisonEvidence:
    """Variance sensitivity between one lower and the selected bandwidth."""

    comparison_lag: int
    selected_lag: int
    maximum_relative_variance_change: float


@dataclass(frozen=True)
class LongRunCovarianceEvidence:
    """Evidence for one long-run covariance estimate."""

    method: LongRunCovarianceMethod
    configuration_count: int
    value_count: int
    maximum_lag: int | None
    iid_centering_correction: float
    numerical_rank: int
    rank_tolerance: float
    covariance: CovarianceProjectionEvidence
    estimator: str
    source: str | None
    autocorrelation: AutocorrelationResolutionEvidence = (
        AutocorrelationResolutionEvidence(
            status=AutocorrelationResolutionStatus.NotAssessed,
        )
    )
    batch_size: int | None = None
    batch_count: int | None = None
    discarded_configuration_count: int = 0
    first_order_bias: str = "not-declared"
    bandwidth_scan_lags: tuple[int, ...] = ()
    bandwidth_comparisons: tuple[BandwidthComparisonEvidence, ...] = ()
    maximum_relative_variance_change: float | None = None
    wolff_validation: ProjectedWolffEvidence | None = None
    covariance_representation: str = "unspecified"

    @property
    def rank_deficient(self) -> bool: return self.numerical_rank < self.value_count


@dataclass(frozen=True)
class LongRunCovarianceEstimate:
    """Correlated means and the evidence supporting their covariance."""

    values: np.ndarray
    evidence: LongRunCovarianceEvidence


@dataclass(frozen=True)
class FactorSpectrum:
    """Numerical rank evidence for a covariance factor."""

    numerical_rank: int
    rank_tolerance: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float


def validate_histories(histories: np.ndarray, *, method_name: str) -> np.ndarray:
    """Return a finite history matrix satisfying the common method contract."""
    values = np.asarray(histories, dtype=float)
    if values.ndim != 2:
        raise StatisticsError("Monte Carlo histories must have shape (configurations, values)")
    configuration_count, value_count = values.shape
    if configuration_count < 2 or value_count < 1:
        raise StatisticsError(f"{method_name} requires at least two configurations and one value")
    if not np.all(np.isfinite(values)):
        raise StatisticsError("Monte Carlo histories contain non-finite values")
    return values


def correlated_values_from_factor(means: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Construct correlated values without forcing a large dense covariance."""
    value_count = factor.shape[1]
    if value_count <= factor.shape[0]:
        covariance = factor.T @ factor
        return np.asarray(gv.gvar(means, covariance, fast=True), dtype=object)
    latent = np.asarray(
        gv.gvar(np.zeros(factor.shape[0]), np.ones(factor.shape[0])),
        dtype=object,
    )
    return np.asarray(means + factor.T @ latent, dtype=object)


def factor_spectrum(factor: np.ndarray) -> FactorSpectrum:
    """Compute scale-aware rank evidence without forming a dense covariance."""
    singular_values = np.linalg.svd(factor, compute_uv=False)
    singular_tolerance = (
        max(factor.shape)
        * np.finfo(float).eps
        * (float(singular_values[0]) if len(singular_values) else 0.0)
    )
    numerical_rank = int(np.count_nonzero(singular_values > singular_tolerance))
    maximum_eigenvalue = (
        float(singular_values[0] ** 2) if len(singular_values) else 0.0
    )
    minimum_eigenvalue = (
        float(singular_values[-1] ** 2)
        if numerical_rank == factor.shape[1] and len(singular_values)
        else 0.0
    )
    return FactorSpectrum(
        numerical_rank=numerical_rank,
        rank_tolerance=singular_tolerance * singular_tolerance,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
    )
