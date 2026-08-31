"""Shared result and evidence values for tree-level corrections."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, runtime_checkable

import numpy as np


class TreeLevelCorrectionError(ValueError):
    """Raised when a tree-level correction is undefined for a request."""


@dataclass(frozen=True)
class LatticeGeometry:
    """Four positive periodic lattice extents parsed from canonical syntax."""

    extents: tuple[int, int, int, int]

    @classmethod
    def parse(cls, volume: str) -> LatticeGeometry:
        match = re.fullmatch(r"l(\d+)l(\d+)l(\d+)t(\d+)", volume)
        if match is None:
            raise TreeLevelCorrectionError(
                "volume must have form l24l24l24t48"
            )
        extents = tuple(int(value) for value in match.groups())
        if any(value <= 0 for value in extents):
            raise TreeLevelCorrectionError(
                "volume must contain four positive extents"
            )
        return cls(extents=extents)


@dataclass(frozen=True)
class CorrectionEvidence:
    """Identity, authority, and applicability of an applied correction."""

    method: str
    source: str
    volume: str
    flow_action: str | None
    gauge_action: str | None
    energy_density_operator: str | None
    flow_time_units: str
    interpolation_spacing: float | None
    validity_domain: str | None = None
    numerical_tolerance: float | None = None
    implementation_notes: tuple[str, ...] = ()

    @property
    def report_description(self) -> str:
        details = [f"source: {self.source}"]
        if self.flow_action is not None:
            details.append(f"flow: {self.flow_action}")
        if self.gauge_action is not None:
            details.append(f"gauge action: {self.gauge_action}")
        if self.energy_density_operator is not None:
            details.append(f"energy: {self.energy_density_operator}")
        if self.validity_domain is not None:
            details.append(f"domain: {self.validity_domain}")
        if self.numerical_tolerance is not None:
            details.append(
                f"numerical tolerance: {self.numerical_tolerance:.3g}"
            )
        if self.implementation_notes:
            details.append(
                "implementation: " + ", ".join(self.implementation_notes)
            )
        return f"{self.method} [{'; '.join(details)}]"


@dataclass(frozen=True)
class CorrectionEstimate:
    """A correction factor minus one and the evidence supporting it."""

    delta: np.ndarray
    evidence: CorrectionEvidence


@dataclass(frozen=True)
class CorrectionRequest:
    """Scientific inputs needed by a correction method."""

    flow_times: tuple[float, ...]
    volume: str
    flow_action: str
    gauge_action: str
    energy_density_operator: str


@runtime_checkable
class CorrectionMethod(Protocol):
    """Interface implemented by finite-lattice correction methods."""

    def estimate(self, request: CorrectionRequest) -> CorrectionEstimate:
        """Return a correction and its applicability evidence."""
