"""Aggregate interface for autocorrelation-aware statistical methods."""

from .bartlett import BartlettLongRunCovariance, BandwidthStabilityCheck
from .batch_means import MultivariateBatchMeans
from .core import (
    AutocorrelationResolutionEvidence,
    AutocorrelationResolutionStatus,
    BandwidthComparisonEvidence,
    CovarianceProjectionEvidence,
    LongRunCovarianceEstimate,
    LongRunCovarianceEvidence,
    LongRunCovarianceMethod,
    StatisticsError,
    UnresolvedAutocorrelation,
    UnresolvedAutocorrelationAction,
)
from .gamma import ProjectedWolffEvidence, ProjectedWolffValidation
from .lugsail import LugsailBatchMeans

__all__ = [
    "AutocorrelationResolutionEvidence",
    "AutocorrelationResolutionStatus",
    "BandwidthComparisonEvidence",
    "BartlettLongRunCovariance",
    "BandwidthStabilityCheck",
    "CovarianceProjectionEvidence",
    "LongRunCovarianceEstimate",
    "LongRunCovarianceEvidence",
    "LongRunCovarianceMethod",
    "LugsailBatchMeans",
    "MultivariateBatchMeans",
    "ProjectedWolffEvidence",
    "ProjectedWolffValidation",
    "StatisticsError",
    "UnresolvedAutocorrelation",
    "UnresolvedAutocorrelationAction",
]
