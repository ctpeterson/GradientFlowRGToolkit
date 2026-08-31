from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .errors import ConfigurationError
from .stages.statistics import LongRunCovarianceMethod
from .stages.tree_level import (
    CorrectionMethod,
    FiniteLatticeTreeLevelNormalization,
    FiniteVolumeNormalization,
)


class Correction(Enum):
    """Finite-lattice correction policies supported by processing."""

    FiniteVolume = "finite-volume"
    FiniteVolumeTreeLevel = "finite-volume-tree-level"


@dataclass(frozen=True)
class CorrectionConfiguration:
    correction: Correction | CorrectionMethod

    def __post_init__(self) -> None:
        if not isinstance(self.correction, (Correction, CorrectionMethod)):
            raise ConfigurationError("correction must be a Correction value or correction method")

    @property
    def method(self) -> CorrectionMethod:
        if self.correction is Correction.FiniteVolume:
            return FiniteVolumeNormalization()
        if self.correction is Correction.FiniteVolumeTreeLevel:
            return FiniteLatticeTreeLevelNormalization()
        return self.correction


@dataclass(frozen=True)
class StatisticsConfiguration:
    method: LongRunCovarianceMethod

    def __post_init__(self) -> None:
        if not isinstance(self.method, LongRunCovarianceMethod):
            raise ConfigurationError("statistics method must be a supported long-run covariance method")


@dataclass(frozen=True)
class EnsembleKey:
    coupling: str
    volume: str
    mass: str


@dataclass(frozen=True)
class DatasetEntry:
    key: EnsembleKey
    flow: str
    path: Path


DatasetPredicate = Callable[[DatasetEntry], bool]


def _select_every_entry(_entry: DatasetEntry) -> bool:
    return True


@dataclass(frozen=True)
class DatasetConfiguration:
    path: Path | str
    flows: tuple[str, ...]
    observables: tuple[str, ...]
    times: tuple[float, float]
    predicate: DatasetPredicate = _select_every_entry
    combine: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not self.flows or len(set(self.flows)) != len(self.flows):
            raise ConfigurationError("flows must contain unique values")
        if not self.observables or len(set(self.observables)) != len(self.observables):
            raise ConfigurationError("observables must contain unique values")
        if len(self.times) != 2:
            raise ConfigurationError("times must contain a minimum and maximum")
        minimum, maximum = self.times
        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
            raise ConfigurationError("times must satisfy finite minimum < maximum")
        if not callable(self.predicate):
            raise ConfigurationError("predicate must be callable")

        normalized_combine: dict[str, Mapping[str, float]] = {}
        for target, sources in self.combine.items():
            if target not in self.observables or not sources:
                raise ConfigurationError("combined observables must name a configured target and at least one source")
            if any(source not in self.observables for source in sources):
                raise ConfigurationError("combined-observable sources must be configured")
            if any(not np.isfinite(weight) for weight in sources.values()):
                raise ConfigurationError("combined-observable weights must be finite")
            normalized_combine[target] = MappingProxyType(dict(sources))

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "flows", tuple(self.flows))
        object.__setattr__(self, "observables", tuple(self.observables))
        object.__setattr__(self, "times", (float(minimum), float(maximum)))
        object.__setattr__(self, "combine", MappingProxyType(normalized_combine))


@dataclass(frozen=True)
class ProcessingConfiguration:
    dataset: DatasetConfiguration
    correction: CorrectionConfiguration
    averaging: StatisticsConfiguration
    verbosity: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.verbosity, bool) or not isinstance(self.verbosity, int):
            raise ConfigurationError("verbosity must be an integer")
        if self.verbosity < 0:
            raise ConfigurationError("verbosity must be non-negative")
