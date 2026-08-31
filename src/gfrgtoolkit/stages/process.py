from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import gvar as gv
import numpy as np

from ..setup import (
    DatasetConfiguration,
    DatasetEntry,
    EnsembleKey,
    ProcessingConfiguration,
)
from .statistics import (
    AutocorrelationResolutionStatus,
    LongRunCovarianceEvidence,
)
from .tree_level import (
    CorrectionEvidence,
    CorrectionRequest,
    TreeLevelCorrectionError,
)


class ProcessingError(ValueError):
    """Raised when selected measurement data violate the processing contract."""


_DATA_FILE = re.compile(
    r"^(?P<coupling>[^_]+)_"
    r"(?P<volume>l\d+l\d+l\d+t\d+)_"
    r"(?P<mass>[^_]+)_"
    r"(?P<flow>[^_]+)\.bin$"
)


@dataclass(frozen=True)
class ProcessedChannel:
    flow: str
    observable: str
    flow_times: tuple[float, ...]
    coupling: tuple[object, ...]
    beta: tuple[object, ...]
    statistics: LongRunCovarianceEvidence
    correction: CorrectionEvidence


@dataclass(frozen=True)
class ProcessedEnsemble:
    key: EnsembleKey
    channels: tuple[ProcessedChannel, ...]


@dataclass(frozen=True)
class ExcludedDatasetEntry:
    path: Path
    reason: str


@dataclass(frozen=True)
class ProcessingResult:
    ensembles: tuple[ProcessedEnsemble, ...]
    exclusions: tuple[ExcludedDatasetEntry, ...]

    @property
    def ensemble_count(self) -> int: return len(self.ensembles)

    @property
    def measurement_channels(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    f"{channel.flow}/{channel.observable}"
                    for ensemble in self.ensembles
                    for channel in ensemble.channels
                }
            )
        )

    @property
    def flow_time_point_count(self) -> int:
        return sum(
            len(channel.flow_times)
            for ensemble in self.ensembles
            for channel in ensemble.channels
        )

    @property
    def covariance_projection_count(self) -> int:
        return sum(
            ensemble.channels[0].statistics.covariance.applied
            for ensemble in self.ensembles
            if ensemble.channels
        )

    @property
    def maximum_relative_covariance_adjustment(self) -> float:
        adjustments = (
            ensemble.channels[0].statistics.covariance.relative_frobenius_adjustment
            for ensemble in self.ensembles
            if ensemble.channels
        )
        return max(adjustments, default=0.0)

    @property
    def unresolved_autocorrelation_count(self) -> int:
        return sum(
            ensemble.channels[0].statistics.autocorrelation.status
            is AutocorrelationResolutionStatus.Unresolved
            for ensemble in self.ensembles
            if ensemble.channels
        )

    @property
    def statistical_methods(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    f"{channel.statistics.estimator} "
                    f"[source: {channel.statistics.source or 'unavailable'}]"
                    for ensemble in self.ensembles
                    for channel in ensemble.channels
                }
            )
        )

    @property
    def correction_methods(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    channel.correction.report_description
                    for ensemble in self.ensembles
                    for channel in ensemble.channels
                }
            )
        )

    def __str__(self) -> str:
        channels = ", ".join(self.measurement_channels) or "none"
        covariance_estimate_count = sum(
            bool(ensemble.channels)
            for ensemble in self.ensembles
        )
        lines = [
                "Processing result",
                f"processed ensembles: {self.ensemble_count}",
                f"excluded ensembles: {len(self.exclusions)}",
                f"measurement channels: {channels}",
                f"processed flow-time points: {self.flow_time_point_count}",
                f"statistics: {', '.join(self.statistical_methods) or 'none'}",
                f"corrections: {', '.join(self.correction_methods) or 'none'}",
                "covariance projections: "
                f"{self.covariance_projection_count}/"
                f"{covariance_estimate_count} estimates",
                "maximum relative covariance adjustment: "
                f"{self.maximum_relative_covariance_adjustment:.3g}",
        ]
        if self.unresolved_autocorrelation_count:
            lines.append(
                "unresolved autocorrelation estimates: "
                f"{self.unresolved_autocorrelation_count}/"
                f"{covariance_estimate_count}"
            )
        return "\n".join(lines)


def _catalog(configuration: DatasetConfiguration) -> tuple[DatasetEntry, ...]:
    if not configuration.path.is_dir():
        raise ProcessingError(f"dataset path does not exist: {configuration.path}")

    entries = []
    for path in sorted(configuration.path.glob("*.bin"), key=lambda item: item.name):
        match = _DATA_FILE.fullmatch(path.name)
        if match is None:
            continue
        entry = DatasetEntry(
            key=EnsembleKey(
                coupling=match.group("coupling"),
                volume=match.group("volume"),
                mass=match.group("mass"),
            ),
            flow=match.group("flow"),
            path=path,
        )
        if entry.flow in configuration.flows and configuration.predicate(entry):
            entries.append(entry)
    return tuple(entries)


def _differentiate(values: np.ndarray, flow_times: np.ndarray) -> np.ndarray:
    if len(flow_times) < 5:
        raise ProcessingError("at least five flow times are required for the beta function")
    beta_values = []
    for center in range(2, len(flow_times) - 2):
        local_times = flow_times[center - 2 : center + 3]
        offsets = local_times - flow_times[center]
        system = np.vstack([offsets**power for power in range(5)])
        target = np.zeros(5)
        target[1] = 1.0
        weights = np.linalg.solve(system, target)
        derivative = sum(
            weight * value for weight, value in zip(weights, values[center - 2 : center + 3],)
        )
        beta_values.append(-flow_times[center] * derivative)
    return np.asarray(beta_values, dtype=object)


def _load_trusted_pickle(path: Path) -> dict:
    try:
        with path.open("rb") as stream: value = pickle.load(stream)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise ProcessingError(f"could not load trusted dataset entry {path.name}") from error
    if not isinstance(value, dict):
        raise ProcessingError(f"{path.name} does not contain a measurement mapping")
    return value


def _selected_histories(
    raw_data: dict,
    observable: str,
    flow_times: np.ndarray,
    selected_indices: np.ndarray,
) -> np.ndarray:
    name = "E" + observable
    if name not in raw_data:
        raise ProcessingError(f"measurement data do not contain {name}")
    histories = raw_data[name]
    if len(histories) != len(flow_times):
        raise ProcessingError(f"{name} history count does not match the flow-time grid")
    selected = [histories[int(index)] for index in selected_indices]
    configuration_counts = {len(history) for history in selected}
    if len(configuration_counts) != 1:
        raise ProcessingError(f"{name} histories have inconsistent configuration counts")
    return np.asarray(selected, dtype=float).T


def _process_entry(
    entry: DatasetEntry,
    configuration: ProcessingConfiguration,
    nc: int,
    gauge_action: str,
) -> ProcessedEnsemble:
    raw_data = _load_trusted_pickle(entry.path)
    if "flow_times" not in raw_data:
        raise ProcessingError("measurement data do not contain flow_times")
    flow_times = np.asarray(raw_data["flow_times"], dtype=float)
    if flow_times.ndim != 1 or not np.all(np.isfinite(flow_times)):
        raise ProcessingError("flow times must be a finite one-dimensional grid")
    if np.any(np.diff(flow_times) <= 0.0):
        raise ProcessingError("flow times must be strictly increasing")

    minimum, maximum = configuration.dataset.times
    selected_indices = np.flatnonzero(
        (flow_times >= minimum) & (flow_times <= maximum)
    )
    selected_times = flow_times[selected_indices]
    if len(selected_times) < 5:
        raise ProcessingError("the configured flow-time window must contain at least five points")

    histories = {
        observable: _selected_histories(
            raw_data,
            observable,
            flow_times,
            selected_indices,
        ) for observable in configuration.dataset.observables
    }

    # combine histories according to the configuration's combine rules;
    # for example, applied if one wants to calculate Symanzik observable
    # from available values of plaquette and rectangle operators
    for target, sources in configuration.dataset.combine.items():
        combined = np.zeros_like(next(iter(histories.values())))
        for source, weight in sources.items():
            combined = combined + weight * histories[source]
        histories[target] = combined

    # this is where all of the real numerical work for processing MC data
    observable_count = len(configuration.dataset.observables)
    time_count = len(selected_times)
    joint_histories = np.concatenate( # concatenate samples along value axis
        [histories[observable] for observable in configuration.dataset.observables],
        axis=1,
    )
    joint_estimate = configuration.averaging.method.estimate(joint_histories)
    joint_energy = joint_estimate.values.reshape(observable_count, time_count)

    channels = []
    for observable_index, observable in enumerate(configuration.dataset.observables):
        try:
            correction = configuration.correction.method.estimate(
                CorrectionRequest(
                    flow_times=tuple(float(time) for time in selected_times),
                    volume=entry.key.volume,
                    flow_action=entry.flow,
                    gauge_action=gauge_action,
                    energy_density_operator=observable,
                )
            )
        except TreeLevelCorrectionError as error:
            raise ProcessingError(str(error)) from error
        normalization = (
            128.0 * np.pi * np.pi / (3.0 * (nc * nc - 1.0))
            * selected_times
            * selected_times
            / (1.0 + correction.delta)
        )
        energy = joint_energy[observable_index]
        coupling = normalization * energy
        beta = _differentiate(coupling, selected_times)
        channels.append(
            ProcessedChannel(
                flow=entry.flow,
                observable=observable,
                flow_times=tuple(float(value) for value in selected_times[2:-2]),
                coupling=tuple(coupling[2:-2]),
                beta=tuple(beta),
                statistics=joint_estimate.evidence,
                correction=correction.evidence,
            )
        )
    return ProcessedEnsemble(key=entry.key, channels=tuple(channels))


def process_dataset(
    configuration: ProcessingConfiguration,
    *,
    nc: int,
    gauge_action: str,
) -> ProcessingResult:
    entries = _catalog(configuration.dataset)
    if not entries:
        raise ProcessingError("dataset selection contains no ensemble files")

    ensembles = []
    exclusions = []
    for entry in entries:
        try: ensembles.append(_process_entry(entry, configuration, nc, gauge_action))
        except ProcessingError as error:
            exclusions.append(ExcludedDatasetEntry(path=entry.path, reason=str(error)))
    if not ensembles:
        reasons = "; ".join(exclusion.reason for exclusion in exclusions)
        raise ProcessingError(f"no selected ensemble could be processed: {reasons}")
    return ProcessingResult(ensembles=tuple(ensembles), exclusions=tuple(exclusions))
