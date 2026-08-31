"""Finite-volume, finite-spacing tree-level normalization (TLN).

This module implements published Eq. (6.2) of Fodor et al., JHEP 09 (2014) 018,
``https://doi.org/10.1007/JHEP09(2014)018``, for periodic gauge fields. In
lattice units ``a=1``, the normalized tree-level factor is

``C(t,L) = 128*pi**2*t**2/(3*V) + 64*pi**2*t**2/(3*V) * sum_{p != 0} T(p,t)``.

Here ``T`` is the trace of two flowed kernels, the inverse gauge-action
kernel, and the energy-density kernel. ``tree_level_delta`` returns
``C(t,L)-1`` because processing divides the measured coupling by ``1+delta``.

The implementation is restricted to Wilson flow. The dynamical gauge action
may be Wilson or tree-level Symanzik, and the energy-density operator may be
plaquette, tree-level Symanzik, or clover. Those discretizations are recorded
because they change the finite-spacing factor.

The direct momentum sum is reorganized without changing the formula: spatial
permutations and reflections are replaced by weighted momentum orbits; the
symmetric flow kernel is diagonalized once per momentum; and equal spectral
exponents are collapsed. The collapsed spectrum is evaluated directly at the
requested flow times with a fixed-order reduction. The zero momentum mode is
omitted from the lattice sum and restored analytically.

This code was originally implemented by Akhil Chauhan from the University of Illinois
Urbana-Champaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from numba import njit, prange
import numpy as np

from ...errors import ConfigurationError
from .core import (
    CorrectionEstimate,
    CorrectionEvidence,
    CorrectionRequest,
    LatticeGeometry,
    TreeLevelCorrectionError,
)


_ENERGY_KERNEL = {
    "p": 0.0,
    "s": -1.0 / 12.0,
    "c": 999.0,
}
_GAUGE_KERNEL = {
    "s": -1.0 / 12.0,
    "w": 0.0,
}
_DIRECTIONS = np.arange(4)
_METHOD = "finite-lattice-tree-level-normalization"
_SOURCE = "https://doi.org/10.1007/JHEP09(2014)018"


@dataclass(frozen=True)
class FiniteLatticeTreeLevelNormalization:
    """Finite-volume, finite-spacing tree-level correction method."""

    exponent_collapse_tolerance: float = 1e-11

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.exponent_collapse_tolerance)
            or self.exponent_collapse_tolerance <= 0.0
        ):
            raise ConfigurationError(
                "exponent_collapse_tolerance must be positive and finite"
            )

    def estimate(self, request: CorrectionRequest) -> CorrectionEstimate:
        return tree_level_correction(
            request.flow_times,
            flow_action=request.flow_action,
            observable=request.energy_density_operator,
            volume=request.volume,
            gauge_action=request.gauge_action,
            exponent_collapse_tolerance=(
                self.exponent_collapse_tolerance
            ),
        )


def _momentum_orbits(spatial_extent: int, temporal_extent: int):
    """Return nonzero momentum representatives and exact multiplicities.

    Reflections reduce each component to ``0..N_mu//2``. Permuting the equal
    spatial axes permits sorted triples ``n_x <= n_y <= n_z``. The closure
    check proves that all ``Ns**3*Nt - 1`` nonzero momenta are represented.
    """
    spatial = np.arange(spatial_extent // 2 + 1)
    temporal = np.arange(temporal_extent // 2 + 1)
    spatial_reflections = np.where(
        (spatial == 0) | (2 * spatial == spatial_extent), 1, 2,
    )
    temporal_reflections = np.where(
        (temporal == 0) | (2 * temporal == temporal_extent), 1, 2,
    )

    first, second, third = np.meshgrid(spatial, spatial, spatial, indexing="ij",)
    keep = (first <= second) & (second <= third)
    first = first[keep]
    second = second[keep]
    third = third[keep]

    first_equals_second = first == second
    second_equals_third = second == third
    permutations = np.where(
        first_equals_second & second_equals_third,
        1,
        np.where(first_equals_second | second_equals_third, 3, 6),
    )
    spatial_multiplicity = (
        permutations
        * spatial_reflections[first]
        * spatial_reflections[second]
        * spatial_reflections[third]
    )

    temporal_count = temporal.size
    representatives = np.empty(
        (first.size * temporal_count, 4),
        dtype=np.int64,
    )
    representatives[:, 0] = np.repeat(first, temporal_count)
    representatives[:, 1] = np.repeat(second, temporal_count)
    representatives[:, 2] = np.repeat(third, temporal_count)
    representatives[:, 3] = np.tile(temporal, first.size)
    multiplicities = (
        np.repeat(spatial_multiplicity, temporal_count)
        * np.tile(temporal_reflections, first.size)
    ).astype(float)

    representatives = representatives[1:]
    multiplicities = multiplicities[1:]
    expected = spatial_extent**3 * temporal_extent - 1
    if int(multiplicities.sum()) != expected:
        raise TreeLevelCorrectionError("tree-level momentum-orbit multiplicities do not close")
    return representatives, multiplicities


def _action_kernel(coefficient, half_momentum, half_squared, half_norm):
    """Evaluate the Symanzik-family kernel of published Eq. (3.5)."""
    matrix = (
        -half_momentum[:, :, None]
        * half_momentum[:, None, :]
        * (
            1.0
            - coefficient
            * (half_squared[:, :, None] + half_squared[:, None, :])
        )
    )
    matrix[:, _DIRECTIONS, _DIRECTIONS] += (
        half_norm[:, None]
        - coefficient * np.sum(half_squared**2, axis=1)[:, None]
        - coefficient * half_squared * half_norm[:, None]
    )
    return matrix


def _clover_kernel(momentum, momentum_norm, cosine_half):
    """Evaluate the clover energy-density kernel of published Eq. (3.6)."""
    matrix = -momentum[:, :, None] * momentum[:, None, :]
    matrix[:, _DIRECTIONS, _DIRECTIONS] += momentum_norm[:, None]
    return matrix * cosine_half[:, :, None] * cosine_half[:, None, :]


def _kernel(
    coefficient,
    half_momentum,
    half_squared,
    half_norm,
    momentum,
    momentum_norm,
    cosine_half,
):
    """Dispatch a Symanzik coefficient or the private clover sentinel."""
    if coefficient == 999.0:
        return _clover_kernel(momentum, momentum_norm, cosine_half)
    return _action_kernel(coefficient, half_momentum, half_squared, half_norm,)


def _spectral_terms(
    representatives,
    multiplicities,
    spatial_extent,
    temporal_extent,
    flow_kernel,
    gauge_kernel,
    energy_kernel,
    block_size=200_000,
):
    """Reduce the nonzero momentum trace to exponential coefficients.

    With ``Sf+G = V diag(lambda) V.T``, define
    ``A = V.T @ inv(Sg+G) @ V`` and ``B = V.T @ Se @ V``. The trace at one
    momentum is ``sum_ij A_ij B_ji exp[-t(lambda_i+lambda_j)]``. Orbit
    multiplicities are folded into the coefficients. Blocks bound temporary
    memory without changing the mathematical sum.
    """
    extents = np.array(
        [spatial_extent, spatial_extent, spatial_extent, temporal_extent],
        dtype=float,
    )
    coefficients = []
    exponents = []

    for lower in range(0, representatives.shape[0], block_size):
        indices = representatives[lower : lower + block_size]
        weights = multiplicities[lower : lower + block_size]
        lattice_momentum = 2.0 * np.pi * indices / extents
        half_momentum = 2.0 * np.sin(lattice_momentum / 2.0)
        half_squared = half_momentum**2
        half_norm = half_squared.sum(axis=1)
        momentum = np.sin(lattice_momentum)
        momentum_norm = (momentum**2).sum(axis=1)
        cosine_half = np.cos(lattice_momentum / 2.0)

        gauge_fixing = half_momentum[:, :, None] * half_momentum[:, None, :]
        flow_matrix = _kernel(
            flow_kernel,
            half_momentum,
            half_squared,
            half_norm,
            momentum,
            momentum_norm,
            cosine_half,
        ) + gauge_fixing
        gauge_matrix = _kernel(
            gauge_kernel,
            half_momentum,
            half_squared,
            half_norm,
            momentum,
            momentum_norm,
            cosine_half,
        ) + gauge_fixing
        energy_matrix = _kernel(
            energy_kernel,
            half_momentum,
            half_squared,
            half_norm,
            momentum,
            momentum_norm,
            cosine_half,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(flow_matrix)
        transpose = np.swapaxes(eigenvectors, 1, 2)
        inverse_action = transpose @ np.linalg.solve(gauge_matrix, eigenvectors)
        measured_energy = transpose @ (energy_matrix @ eigenvectors)
        coefficients.append(
            (
                inverse_action
                * np.swapaxes(measured_energy, 1, 2)
                * weights[:, None, None]
            ).ravel()
        )
        exponents.append((eigenvalues[:, :, None] + eigenvalues[:, None, :]).ravel())

    return np.concatenate(coefficients), np.concatenate(exponents)


def _collapse_terms(exponents, coefficients, tolerance=1e-11):
    """Merge numerically equal exponents by summing their coefficients."""
    order = np.argsort(exponents, kind="stable")
    sorted_exponents = exponents[order]
    sorted_coefficients = coefficients[order]
    starts_new = np.empty(sorted_exponents.size, dtype=bool)
    starts_new[0] = True
    np.greater(np.diff(sorted_exponents), tolerance, out=starts_new[1:],)
    starts = np.flatnonzero(starts_new)
    return (sorted_exponents[starts], np.add.reduceat(sorted_coefficients, starts),)


@njit(parallel=True, cache=False)
def _evaluate_spectrum(coefficients, exponents, flow_times):
    """Evaluate spectral terms directly with a fixed per-time reduction."""
    values = np.empty(flow_times.shape[0])
    for time_index in prange(flow_times.shape[0]):
        total = 0.0
        for term_index in range(coefficients.shape[0]):
            total += coefficients[term_index] * np.exp(
                -flow_times[time_index] * exponents[term_index]
            )
        values[time_index] = total
    return values


@lru_cache(maxsize=64)
def _tree_level_spectrum(
    spatial_extent: int,
    temporal_extent: int,
    gauge_kernel: float,
    energy_kernel: float,
    exponent_collapse_tolerance: float,
):
    """Compute and cache collapsed nonzero-momentum spectral terms."""
    representatives, multiplicities = _momentum_orbits(
        spatial_extent,
        temporal_extent,
    )
    coefficients, exponents = _spectral_terms(
        representatives,
        multiplicities,
        spatial_extent,
        temporal_extent,
        flow_kernel=0.0,
        gauge_kernel=gauge_kernel,
        energy_kernel=energy_kernel,
    )
    exponents, coefficients = _collapse_terms(
        exponents,
        coefficients,
        tolerance=exponent_collapse_tolerance,
    )
    coefficients.setflags(write=False)
    exponents.setflags(write=False)
    return coefficients, exponents


def _tree_level_normalization(
    flow_times: np.ndarray,
    *,
    spatial_extent: int,
    temporal_extent: int,
    gauge_kernel: float,
    energy_kernel: float,
    exponent_collapse_tolerance: float,
) -> np.ndarray:
    """Evaluate published Eq. (6.2) directly at requested flow times."""
    coefficients, exponents = _tree_level_spectrum(
        spatial_extent,
        temporal_extent,
        gauge_kernel,
        energy_kernel,
        exponent_collapse_tolerance,
    )
    lattice_sum = _evaluate_spectrum(
        coefficients,
        exponents,
        flow_times,
    )
    prefactor = (
        64.0
        * np.pi**2
        * flow_times * flow_times
        / (3.0 * spatial_extent**3 * temporal_extent)
    )
    return 2.0 * prefactor + prefactor * lattice_sum


def _geometry(volume: str) -> tuple[int, int]:
    """Parse an isotropic spatial lattice and its temporal extent."""
    values = LatticeGeometry.parse(volume).extents
    if len(set(values[:3])) != 1:
        raise TreeLevelCorrectionError("tree-level normalization requires equal spatial extents")
    if int(np.prod(values)) <= 1:
        raise TreeLevelCorrectionError(
            "tree-level normalization requires at least one nonzero momentum"
        )
    return values[0], values[-1]


def tree_level_delta(
    flow_times,
    *,
    observable: str,
    volume: str,
    gauge_action: str,
    exponent_collapse_tolerance: float = 1e-11,
):
    """Return the Wilson-flow tree-level correction ``C(t,L)-1``.

    ``flow_times`` are expressed in lattice units ``t/a**2``. ``observable``
    is ``p`` (plaquette), ``s`` (tree-level Symanzik), or ``c`` (clover), and
    ``gauge_action`` is ``w`` (Wilson) or ``s`` (tree-level Symanzik).
    Requests beyond ``sqrt(8t)/L_s=1/2`` reject. Spectral exponent merging is
    controlled by an explicit numerical tolerance and requested times are
    evaluated directly without interpolation.
    """
    if observable not in _ENERGY_KERNEL:
        raise TreeLevelCorrectionError(f"unsupported energy-density operator {observable!r}")
    if gauge_action not in _GAUGE_KERNEL:
        raise TreeLevelCorrectionError(f"unsupported gauge action {gauge_action!r}")
    times = np.asarray(flow_times, dtype=float)
    if (
        times.ndim != 1
        or times.size == 0
        or not np.all(np.isfinite(times))
        or np.any(times <= 0.0)
    ):
        raise TreeLevelCorrectionError("tree-level normalization requires positive finite flow times")
    if (
        not np.isfinite(exponent_collapse_tolerance)
        or exponent_collapse_tolerance <= 0.0
    ):
        raise TreeLevelCorrectionError(
            "exponent_collapse_tolerance must be positive and finite"
        )

    spatial_extent, temporal_extent = _geometry(volume)
    maximum_time = spatial_extent**2 / 32.0
    if float(times.max()) > maximum_time:
        raise TreeLevelCorrectionError(
            "tree-level normalization requires sqrt(8t)/N_s <= 1/2; "
            f"maximum flow time is {maximum_time:g}"
        )
    normalization = _tree_level_normalization(
        times,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        gauge_kernel=_GAUGE_KERNEL[gauge_action],
        energy_kernel=_ENERGY_KERNEL[observable],
        exponent_collapse_tolerance=float(exponent_collapse_tolerance),
    )
    if not np.all(np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise TreeLevelCorrectionError(
            "tree-level normalization produced a non-positive or non-finite factor"
        )
    return normalization - 1.0


def tree_level_correction(
    flow_times,
    *,
    flow_action: str,
    observable: str,
    volume: str,
    gauge_action: str,
    exponent_collapse_tolerance: float = 1e-11,
) -> CorrectionEstimate:
    """Return finite-lattice tree-level normalization and its source evidence."""
    if flow_action != "wilson":
        raise TreeLevelCorrectionError("tree-level normalization supports only Wilson flow")
    return CorrectionEstimate(
        delta=tree_level_delta(
            flow_times,
            observable=observable,
            volume=volume,
            gauge_action=gauge_action,
            exponent_collapse_tolerance=exponent_collapse_tolerance,
        ),
        evidence=CorrectionEvidence(
            method=_METHOD,
            source=_SOURCE,
            volume=volume,
            flow_action=flow_action,
            gauge_action=gauge_action,
            energy_density_operator=observable,
            flow_time_units="t/a^2",
            interpolation_spacing=None,
            validity_domain="0 < sqrt(8t)/N_s <= 1/2",
            numerical_tolerance=float(exponent_collapse_tolerance),
            implementation_notes=(
                "rectangular-time periodic momentum sum",
                "direct requested-time spectral evaluation",
                "fixed-order per-time reduction without fast math",
            ),
        ),
    )
