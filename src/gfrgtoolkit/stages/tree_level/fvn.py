"""Continuum finite-volume normalization on a periodic four-torus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import ConfigurationError
from .core import (
    CorrectionEstimate,
    CorrectionEvidence,
    CorrectionRequest,
    LatticeGeometry,
    TreeLevelCorrectionError,
)


_METHOD = "finite-volume-normalization"
_SOURCE = "https://doi.org/10.1007/JHEP11(2012)007"


@dataclass(frozen=True)
class FiniteVolumeNormalization:
    """Continuum finite-volume normalization correction method."""

    theta_relative_tolerance: float = 1e-15

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.theta_relative_tolerance)
            or self.theta_relative_tolerance <= 0.0
        ):
            raise ConfigurationError(
                "theta_relative_tolerance must be positive and finite"
            )

    def estimate(self, request: CorrectionRequest) -> CorrectionEstimate:
        return finite_volume_correction(
            request.flow_times,
            volume=request.volume,
            theta_relative_tolerance=self.theta_relative_tolerance,
        )


def _extents(volume: str) -> np.ndarray:
    return np.asarray(LatticeGeometry.parse(volume).extents, dtype=float)


def _theta3_exp_minus(
    exponent: np.ndarray,
    *,
    relative_tolerance: float,
) -> np.ndarray:
    """Evaluate theta_3(exp(-x)) with a uniformly convergent series."""
    use_modular_form = exponent < np.pi
    reduced_exponent = np.where(
        use_modular_form,
        np.pi**2 / exponent,
        exponent,
    )
    prefactor = np.where(
        use_modular_form,
        np.sqrt(np.pi / exponent),
        1.0,
    )
    theta = np.ones_like(exponent)
    for index in range(1, 65):
        term = 2.0 * np.exp(-reduced_exponent * index * index)
        theta += term
        if np.all(term <= relative_tolerance * theta):
            return prefactor * theta
    raise TreeLevelCorrectionError(
        "Jacobi theta series did not reach its declared tolerance"
    )


def finite_volume_correction(
    flow_times,
    *,
    volume: str,
    theta_relative_tolerance: float = 1e-15,
) -> CorrectionEstimate:
    """Return the continuum finite-volume correction ``delta(t, L)``.

    The Jacobi-theta product is evaluated through its direct or modular series
    until the declared relative tolerance is reached. The algebraic zero-mode
    term is evaluated exactly for the rectangular-torus extension.
    """
    times = np.asarray(flow_times, dtype=float)
    if (
        times.ndim != 1
        or times.size == 0
        or not np.all(np.isfinite(times))
        or np.any(times <= 0.0)
    ):
        raise TreeLevelCorrectionError(
            "finite-volume normalization requires positive finite flow times"
        )
    if (
        not np.isfinite(theta_relative_tolerance)
        or theta_relative_tolerance <= 0.0
    ):
        raise TreeLevelCorrectionError(
            "theta_relative_tolerance must be positive and finite"
        )
    extents = _extents(volume)
    algebraic = np.full_like(times, -64.0 * np.pi**2 / 3.0)
    exponential = np.ones_like(times)
    for extent in extents:
        ratio = extent * extent / times
        algebraic /= np.sqrt(ratio)
        exponential *= _theta3_exp_minus(
            ratio / 8.0,
            relative_tolerance=theta_relative_tolerance,
        )
    return CorrectionEstimate(
        delta=algebraic + exponential - 1.0,
        evidence=CorrectionEvidence(
            method=_METHOD,
            source=_SOURCE,
            volume=volume,
            flow_action=None,
            gauge_action=None,
            energy_density_operator=None,
            flow_time_units="t/a^2",
            interpolation_spacing=None,
            validity_domain="positive finite flow time",
            numerical_tolerance=float(theta_relative_tolerance),
            implementation_notes=(
                "rectangular-torus product extension",
                "Jacobi theta modular-series evaluation",
            ),
        ),
    )
