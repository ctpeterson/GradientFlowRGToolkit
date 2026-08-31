from __future__ import annotations

import gfrgtoolkit as betafn
from numba import get_num_threads, set_num_threads
import numpy as np
import pytest
from scipy.linalg import expm


def _request(
    flow_times,
    *,
    volume="l8l8l8t8",
    flow_action="wilson",
    gauge_action="w",
    energy_density_operator="p",
):
    return betafn.CorrectionRequest(
        flow_times=tuple(flow_times),
        volume=volume,
        flow_action=flow_action,
        gauge_action=gauge_action,
        energy_density_operator=energy_density_operator,
    )


def _published_action_kernel(coefficient, half_momentum):
    half_squared = half_momentum**2
    half_norm = np.sum(half_squared)
    matrix = -np.outer(half_momentum, half_momentum) * (
        1.0
        - coefficient
        * (half_squared[:, None] + half_squared[None, :])
    )
    matrix[np.diag_indices(4)] += (
        half_norm
        - coefficient * np.sum(half_squared**2)
        - coefficient * half_squared * half_norm
    )
    return matrix


def _published_energy_kernel(observable, momentum):
    if observable != "c":
        coefficient = {"p": 0.0, "s": -1.0 / 12.0}[observable]
        return _published_action_kernel(
            coefficient,
            2.0 * np.sin(momentum / 2.0),
        )
    sine = np.sin(momentum)
    matrix = np.eye(4) * np.sum(sine**2) - np.outer(sine, sine)
    cosine = np.cos(momentum / 2.0)
    return matrix * np.outer(cosine, cosine)


def _published_full_momentum_tln(
    time,
    *,
    spatial_extent,
    temporal_extent,
    gauge_action,
    observable,
):
    extents = np.asarray(
        [spatial_extent, spatial_extent, spatial_extent, temporal_extent]
    )
    indices = np.indices(tuple(extents)).reshape(4, -1).T
    volume = int(np.prod(extents))
    trace_sum = 0.0
    for index in indices[1:]:
        momentum = 2.0 * np.pi * index / extents
        half_momentum = 2.0 * np.sin(momentum / 2.0)
        gauge_fixing = np.outer(half_momentum, half_momentum)
        flow = _published_action_kernel(0.0, half_momentum) + gauge_fixing
        gauge = (
            _published_action_kernel(
                {"w": 0.0, "s": -1.0 / 12.0}[gauge_action],
                half_momentum,
            )
            + gauge_fixing
        )
        energy = _published_energy_kernel(observable, momentum)
        flowed = expm(-time * flow)
        trace_sum += np.trace(
            flowed @ np.linalg.solve(gauge, flowed @ energy)
        )
    prefactor = 64.0 * np.pi**2 * time**2 / (3.0 * volume)
    return 2.0 * prefactor + prefactor * trace_sum


def test_fvn_evaluates_the_complete_theta_function_outside_small_c():
    # Independent evaluation of the published Jacobi-theta formula at c=2.
    estimate = betafn.FiniteVolumeNormalization().estimate(
        _request((32.0,))
    )
    np.testing.assert_allclose(
        1.0 + estimate.delta,
        [105.2757802782865],
        rtol=1e-13,
        atol=0.0,
    )


def test_tln_rejects_flow_time_above_exact_half_scheme_boundary():
    maximum_time = 10.0**2 / 32.0

    with pytest.raises(
        betafn.TreeLevelCorrectionError,
        match=r"sqrt\(8t\)/N_s <= 1/2",
    ):
        betafn.FiniteLatticeTreeLevelNormalization().estimate(
            _request(
                (maximum_time + 0.001,),
                volume="l10l10l10t10",
            )
        )


def test_tln_matches_the_independent_wwp_full_momentum_sum_at_tiny_time():
    spatial_extent = 4
    temporal_extent = 6
    time = 0.0001
    extents = np.asarray(
        [spatial_extent, spatial_extent, spatial_extent, temporal_extent]
    )
    indices = np.indices(tuple(extents)).reshape(4, -1).T
    nonzero = np.any(indices != 0, axis=1)
    momentum = 2.0 * np.pi * indices[nonzero] / extents
    half_norm = np.sum((2.0 * np.sin(momentum / 2.0)) ** 2, axis=1)
    volume = spatial_extent**3 * temporal_extent
    expected = (
        128.0 * np.pi**2 * time**2 / (3.0 * volume)
        + 64.0
        * np.pi**2
        * time**2
        / volume
        * np.sum(np.exp(-2.0 * time * half_norm))
    )

    estimate = betafn.FiniteLatticeTreeLevelNormalization().estimate(
        _request(
            (time,),
            volume="l4l4l4t6",
        )
    )

    np.testing.assert_allclose(
        1.0 + estimate.delta,
        [expected],
        # The optimized path includes 4x4 eigensolves before its fixed-order
        # reduction; this bound remains four orders above machine epsilon.
        rtol=5e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "method",
    [
        betafn.FiniteVolumeNormalization(),
        betafn.FiniteLatticeTreeLevelNormalization(),
    ],
)
def test_correction_methods_reject_a_volume_with_surrounding_junk(method):
    with pytest.raises(
        betafn.TreeLevelCorrectionError,
        match="volume must have form",
    ):
        method.estimate(
            _request(
                (0.5,),
                volume="prefix-l8l8l8t8-suffix",
            )
        )


@pytest.mark.parametrize(
    "method",
    [
        betafn.FiniteVolumeNormalization(),
        betafn.FiniteLatticeTreeLevelNormalization(),
    ],
)
def test_correction_methods_reject_an_empty_flow_time_grid(method):
    with pytest.raises(
        betafn.TreeLevelCorrectionError,
        match="positive finite flow times",
    ):
        method.estimate(_request(()))


def test_tln_rejects_a_lattice_without_nonzero_momentum():
    with pytest.raises(
        betafn.TreeLevelCorrectionError,
        match="at least one nonzero momentum",
    ):
        betafn.FiniteLatticeTreeLevelNormalization().estimate(
            _request((0.01,), volume="l1l1l1t1")
        )


@pytest.mark.parametrize("gauge_action", ["w", "s"])
@pytest.mark.parametrize("observable", ["p", "s", "c"])
def test_tln_matches_published_full_momentum_sum_for_every_supported_kernel(
    gauge_action,
    observable,
):
    time = 0.17
    expected = _published_full_momentum_tln(
        time,
        spatial_extent=4,
        temporal_extent=6,
        gauge_action=gauge_action,
        observable=observable,
    )

    estimate = betafn.FiniteLatticeTreeLevelNormalization().estimate(
        _request(
            (time,),
            volume="l4l4l4t6",
            gauge_action=gauge_action,
            energy_density_operator=observable,
        )
    )

    np.testing.assert_allclose(
        1.0 + estimate.delta,
        [expected],
        rtol=2e-12,
        atol=2e-15,
    )


def test_tln_reduction_is_independent_of_numba_thread_count():
    original_thread_count = get_num_threads()
    alternative_thread_count = min(2, original_thread_count)
    request = _request(
        (0.03, 0.17, 0.31),
        volume="l4l4l4t6",
        gauge_action="s",
        energy_density_operator="c",
    )
    try:
        set_num_threads(1)
        serial = betafn.FiniteLatticeTreeLevelNormalization().estimate(request)
        set_num_threads(alternative_thread_count)
        parallel = betafn.FiniteLatticeTreeLevelNormalization().estimate(request)
    finally:
        set_num_threads(original_thread_count)

    np.testing.assert_array_equal(serial.delta, parallel.delta)
