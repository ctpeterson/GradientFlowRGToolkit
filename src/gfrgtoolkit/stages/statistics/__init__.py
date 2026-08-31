"""Aggregate interface for autocorrelation-aware statistical methods."""

from .bartlett import BartlettLongRunCovariance, BandwidthStabilityCheck
from .batch_means import MultivariateBatchMeans
from .core import (
    AutocorrelationResolutionEvidence,
    AutocorrelationResolutionStatus,
    BandwidthComparisonEvidence,
    CovarianceProjection,
    CovarianceProjectionEvidence,
    LongRunCovarianceEstimate,
    LongRunCovarianceEvidence,
    LongRunCovarianceMethod,
    StatisticsError,
    UnresolvedAutocorrelation,
    UnresolvedAutocorrelationAction,
)
from .experimental_rectangular import (
    ExperimentalRectangularLongRunCovariance,
    GammaMethod,
)
from .lugsail_batch_means import LugsailBatchMeans
from .wolff import ProjectedWolffEvidence, ProjectedWolffValidation

# Transitional evidence alias for early notebooks.
GammaMethodEvidence = LongRunCovarianceEvidence

__all__ = [
    "AutocorrelationResolutionEvidence",
    "AutocorrelationResolutionStatus",
    "BandwidthComparisonEvidence",
    "BartlettLongRunCovariance",
    "BandwidthStabilityCheck",
    "CovarianceProjection",
    "CovarianceProjectionEvidence",
    "ExperimentalRectangularLongRunCovariance",
    "GammaMethod",
    "GammaMethodEvidence",
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
