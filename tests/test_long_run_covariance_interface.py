from __future__ import annotations

import pickle

import gfrgtoolkit as betafn
import gvar as gv
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov


def test_legacy_rectangular_covariance_is_not_public():
    legacy_names = (
        "ExperimentalRectangularLongRunCovariance",
        "GammaMethod",
        "GammaMethodEvidence",
    )
    assert all(not hasattr(betafn, name) for name in legacy_names)


def test_covariance_projection_policy_is_not_public():
    assert not hasattr(betafn, "CovarianceProjection")


def test_lugsail_configuration_rejects_incompatible_batch_scales():
    with pytest.raises(
        betafn.ConfigurationError,
        match="divisible",
    ):
        betafn.LugsailBatchMeans(
            batch_size=61,
            lugsail_scale=3,
            lugsail_weight=0.5,
        )


def test_bartlett_processing_preserves_cross_channel_covariance(tmp_path):
    flow_times = np.arange(1.0, 8.0)
    offsets = np.asarray([-0.14, -0.09, -0.03, 0.02, 0.06, 0.11, 0.15])
    plaquette = np.asarray(
        [
            1.0 + 0.1 * flow_time + offsets
            for flow_time in flow_times
        ]
    )
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": plaquette.tolist(),
        "Es": (2.0 * plaquette).tolist(),
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    result = betafn.RunningCoupling(nf=2).process(
        betafn.ProcessingConfiguration(
            dataset=betafn.DatasetConfiguration(
                path=tmp_path,
                flows=("wilson",),
                observables=("p", "s"),
                times=(1.0, 7.0),
            ),
            correction=betafn.CorrectionConfiguration(
                betafn.Correction.FiniteVolume
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.BartlettLongRunCovariance(maximum_lag=0)
            ),
        )
    )

    plaquette_channel, symanzik_channel = result.ensembles[0].channels
    paired = np.asarray(
        [plaquette_channel.coupling[0], symanzik_channel.coupling[0]],
        dtype=object,
    )

    np.testing.assert_allclose(gv.evalcorr(paired), [[1.0, 1.0], [1.0, 1.0]])
    assert "covariance projections: 0/1 estimates" in str(result)


def test_zero_lag_bartlett_is_unbiased_for_iid_history_covariance(tmp_path):
    flow_times = np.arange(1.0, 8.0)
    offsets = np.asarray([-3.0, -1.0, 1.0, 3.0])
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            (10.0 + 0.1 * flow_time + offsets).tolist()
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    result = betafn.RunningCoupling(nf=2).process(
        betafn.ProcessingConfiguration(
            dataset=betafn.DatasetConfiguration(
                path=tmp_path,
                flows=("wilson",),
                observables=("p",),
                times=(1.0, 7.0),
            ),
            correction=betafn.CorrectionConfiguration(
                betafn.Correction.FiniteVolume
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.BartlettLongRunCovariance(maximum_lag=0)
            ),
        )
    )

    first_retained_coupling = result.ensembles[0].channels[0].coupling[0]
    expected_energy_mean = 10.3
    expected_variance_of_mean = 5.0 / 3.0

    np.testing.assert_allclose(
        gv.sdev(first_retained_coupling) / gv.mean(first_retained_coupling),
        np.sqrt(expected_variance_of_mean) / expected_energy_mean,
        rtol=1e-12,
        atol=1e-12,
    )


def test_multivariate_batch_means_matches_independent_worked_example(tmp_path):
    flow_times = np.arange(1.0, 8.0)
    offsets = np.asarray([-3.0, -1.0, -1.0, 1.0, 1.0, 3.0, 3.0, 5.0])
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            (10.0 + 0.1 * flow_time + offsets).tolist()
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    result = betafn.RunningCoupling(nf=2).process(
        betafn.ProcessingConfiguration(
            dataset=betafn.DatasetConfiguration(
                path=tmp_path,
                flows=("wilson",),
                observables=("p",),
                times=(1.0, 7.0),
            ),
            correction=betafn.CorrectionConfiguration(
                betafn.Correction.FiniteVolume
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.MultivariateBatchMeans(batch_size=2)
            ),
        )
    )

    channel = result.ensembles[0].channels[0]
    first_retained_coupling = channel.coupling[0]

    # The four batch means have offsets [-2, 0, 2, 4]. Their unbiased sample
    # variance is 20/3, so the estimated variance of their mean is 5/3.
    np.testing.assert_allclose(
        gv.sdev(first_retained_coupling) / gv.mean(first_retained_coupling),
        np.sqrt(5.0 / 3.0) / 11.3,
        rtol=1e-12,
        atol=1e-12,
    )
    assert channel.statistics.batch_size == 2
    assert channel.statistics.batch_count == 4
    assert channel.statistics.discarded_configuration_count == 0


def test_bartlett_reports_unavoidable_high_dimensional_rank_deficiency(tmp_path):
    configuration_count = 12
    flow_times = np.arange(1.0, 21.0)
    histories = np.random.default_rng(1701).normal(
        size=(configuration_count, len(flow_times))
    )
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": histories.T.tolist(),
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    result = betafn.RunningCoupling(nf=2).process(
        betafn.ProcessingConfiguration(
            dataset=betafn.DatasetConfiguration(
                path=tmp_path,
                flows=("wilson",),
                observables=("p",),
                times=(1.0, 20.0),
            ),
            correction=betafn.CorrectionConfiguration(
                betafn.Correction.FiniteVolume
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.BartlettLongRunCovariance(maximum_lag=3)
            ),
        )
    )

    evidence = result.ensembles[0].channels[0].statistics

    assert evidence.value_count == len(flow_times)
    assert evidence.numerical_rank <= configuration_count - 1
    assert evidence.rank_deficient
    assert (
        evidence.covariance.minimum_eigenvalue_before
        >= -evidence.rank_tolerance
    )
    assert not evidence.covariance.applied


def test_over_lugsail_batch_means_offsets_positive_correlation_bias(tmp_path):
    configuration_count = 6000
    burn_in = 1000
    autoregression = 0.9
    generator = np.random.default_rng(1701)
    history = np.zeros(configuration_count + burn_in)
    innovations = generator.normal(size=len(history))
    for index in range(1, len(history)):
        history[index] = (
            autoregression * history[index - 1] + innovations[index]
        )
    history = history[burn_in:]

    flow_times = np.arange(1.0, 8.0)
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            (10.0 + 0.1 * flow_time + history).tolist()
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    def estimate_variance(method):
        result = betafn.RunningCoupling(nf=2).process(
            betafn.ProcessingConfiguration(
                dataset=betafn.DatasetConfiguration(
                    path=tmp_path,
                    flows=("wilson",),
                    observables=("p",),
                    times=(1.0, 7.0),
                ),
                correction=betafn.CorrectionConfiguration(
                    betafn.Correction.FiniteVolume
                ),
                averaging=betafn.StatisticsConfiguration(method),
            )
        )
        channel = result.ensembles[0].channels[0]
        coupling = channel.coupling[0]
        energy_mean = 10.3 + history.mean()
        normalization = gv.mean(coupling) / energy_mean
        return gv.var(coupling) / normalization**2, channel.statistics

    ordinary_variance, _ = estimate_variance(
        betafn.MultivariateBatchMeans(batch_size=60)
    )
    lugsail_variance, evidence = estimate_variance(
        betafn.LugsailBatchMeans(
            batch_size=60,
            lugsail_scale=3,
            lugsail_weight=0.5,
        )
    )
    exact_asymptotic_variance = (
        1.0
        / (1.0 - autoregression) ** 2
        / configuration_count
    )

    assert ordinary_variance < 0.9 * exact_asymptotic_variance
    assert lugsail_variance > ordinary_variance
    np.testing.assert_allclose(
        lugsail_variance,
        exact_asymptotic_variance,
        rtol=0.05,
    )
    assert evidence.first_order_bias == "positive-for-positive-correlation"
    assert (
        evidence.source
        == "https://doi.org/10.1093/biomet/asab049#equation-7"
    )


def test_lugsail_nearest_psd_policy_projects_negative_modes():
    histories = np.random.default_rng(0).normal(size=(12, 2))

    estimate = betafn.LugsailBatchMeans(batch_size=6).estimate(histories)
    evidence = estimate.evidence.covariance

    assert evidence.minimum_eigenvalue_before < 0.0
    assert evidence.projected_mode_count > 0
    assert evidence.applied
    assert np.linalg.eigvalsh(gv.evalcov(estimate.values))[0] >= -1e-15


def test_lugsail_preserves_a_low_rank_factor_in_high_dimension():
    histories = np.random.default_rng(81).normal(size=(60, 200))

    estimate = betafn.LugsailBatchMeans(batch_size=6).estimate(histories)

    assert estimate.evidence.covariance_representation == "projected-low-rank-factor"
    assert estimate.evidence.numerical_rank <= 39
    assert len(estimate.values) == 200


def test_bartlett_records_unresolved_bandwidth_without_blocking_analysis(tmp_path):
    configuration_count = 500
    generator = np.random.default_rng(23)
    history = np.zeros(configuration_count + 500)
    innovations = generator.normal(size=len(history))
    for index in range(1, len(history)):
        history[index] = 0.95 * history[index - 1] + innovations[index]
    history = history[500:]

    flow_times = np.arange(1.0, 8.0)
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            (10.0 + flow_time + history).tolist()
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    result = betafn.RunningCoupling(nf=2).process(
        betafn.ProcessingConfiguration(
            dataset=betafn.DatasetConfiguration(
                path=tmp_path,
                flows=("wilson",),
                observables=("p",),
                times=(1.0, 7.0),
            ),
            correction=betafn.CorrectionConfiguration(
                betafn.Correction.FiniteVolume
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.BartlettLongRunCovariance(
                    maximum_lag=4,
                    stability=betafn.BandwidthStabilityCheck(
                        comparison_lags=(0, 2),
                        relative_tolerance=0.05,
                    ),
                )
            ),
        )
    )

    evidence = result.ensembles[0].channels[0].statistics
    assert (
        evidence.autocorrelation.status
        is betafn.AutocorrelationResolutionStatus.Unresolved
    )
    assert "bandwidth sensitivity" in evidence.autocorrelation.diagnostics[0]
    assert "unresolved autocorrelation estimates: 1/1" in str(result)


def test_bartlett_evaluates_every_declared_bandwidth_comparison():
    generator = np.random.default_rng(481)
    innovations = generator.normal(size=(600, 2))
    histories = np.empty_like(innovations)
    histories[0] = innovations[0]
    for index in range(1, len(histories)):
        histories[index] = 0.8 * histories[index - 1] + innovations[index]

    estimate = betafn.BartlettLongRunCovariance(
        maximum_lag=12,
        stability=betafn.BandwidthStabilityCheck(
            comparison_lags=(2, 4, 8),
            relative_tolerance=1.0,
        ),
    ).estimate(histories)

    comparisons = estimate.evidence.bandwidth_comparisons
    assert tuple(item.comparison_lag for item in comparisons) == (2, 4, 8)
    assert all(item.selected_lag == 12 for item in comparisons)
    assert estimate.evidence.maximum_relative_variance_change == max(
        item.maximum_relative_variance_change for item in comparisons
    )


def test_bartlett_can_strictly_reject_unresolved_bandwidth(tmp_path):
    configuration_count = 500
    generator = np.random.default_rng(23)
    history = np.zeros(configuration_count + 500)
    innovations = generator.normal(size=len(history))
    for index in range(1, len(history)):
        history[index] = 0.95 * history[index - 1] + innovations[index]
    history = history[500:]

    flow_times = np.arange(1.0, 8.0)
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            (10.0 + flow_time + history).tolist()
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    with pytest.raises(
        betafn.UnresolvedAutocorrelation,
        match="bandwidth sensitivity",
    ):
        betafn.RunningCoupling(nf=2).process(
            betafn.ProcessingConfiguration(
                dataset=betafn.DatasetConfiguration(
                    path=tmp_path,
                    flows=("wilson",),
                    observables=("p",),
                    times=(1.0, 7.0),
                ),
                correction=betafn.CorrectionConfiguration(
                    betafn.Correction.FiniteVolume
                ),
                averaging=betafn.StatisticsConfiguration(
                    betafn.BartlettLongRunCovariance(
                        maximum_lag=4,
                        stability=betafn.BandwidthStabilityCheck(
                            comparison_lags=(0, 2),
                            relative_tolerance=0.05,
                        ),
                        on_unresolved=(
                            betafn.UnresolvedAutocorrelationAction.Raise
                        ),
                    )
                ),
            )
        )


def test_projected_wolff_records_unresolved_coordinates_without_blocking(tmp_path):
    configuration_count = 500
    generator = np.random.default_rng(23)
    history = np.zeros(configuration_count + 500)
    innovations = generator.normal(size=len(history))
    for index in range(1, len(history)):
        history[index] = 0.95 * history[index - 1] + innovations[index]
    history = history[500:]

    flow_times = np.arange(1.0, 8.0)
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            (10.0 + flow_time + history).tolist()
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    result = betafn.RunningCoupling(nf=2).process(
        betafn.ProcessingConfiguration(
            dataset=betafn.DatasetConfiguration(
                path=tmp_path,
                flows=("wilson",),
                observables=("p",),
                times=(1.0, 7.0),
            ),
            correction=betafn.CorrectionConfiguration(
                betafn.Correction.FiniteVolume
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.BartlettLongRunCovariance(
                    maximum_lag=4,
                    wolff_validation=betafn.ProjectedWolffValidation(
                        exponential_scale=1.5,
                        maximum_lag=1,
                    ),
                )
            ),
        )
    )

    evidence = result.ensembles[0].channels[0].statistics
    assert (
        evidence.autocorrelation.status
        is betafn.AutocorrelationResolutionStatus.Unresolved
    )
    assert evidence.wolff_validation is not None
    assert evidence.wolff_validation.unresolved_coordinate_count == 7


def test_projected_wolff_disagreement_marks_bartlett_unresolved():
    histories = np.random.default_rng(915).normal(size=(500, 3))

    estimate = betafn.BartlettLongRunCovariance(
        maximum_lag=0,
        wolff_validation=betafn.ProjectedWolffValidation(
            exponential_scale=1.5,
            maximum_lag=50,
            relative_variance_tolerance=1e-12,
        ),
    ).estimate(histories)

    evidence = estimate.evidence
    assert (
        evidence.autocorrelation.status
        is betafn.AutocorrelationResolutionStatus.Unresolved
    )
    assert evidence.wolff_validation is not None
    assert evidence.wolff_validation.unresolved_coordinate_count == 0
    assert "Wolff variance disagreement" in evidence.autocorrelation.diagnostics[0]


def test_projected_wolff_assesses_declared_linear_combinations():
    histories = np.random.default_rng(1103).normal(size=(800, 2))

    estimate = betafn.BartlettLongRunCovariance(
        maximum_lag=4,
        wolff_validation=betafn.ProjectedWolffValidation(
            exponential_scale=1.5,
            maximum_lag=100,
            projections=((1.0, -1.0),),
        ),
    ).estimate(histories)

    evidence = estimate.evidence.wolff_validation
    assert evidence is not None
    assert evidence.assessed_coordinate_count == 2
    assert evidence.declared_projection_count == 1
    assert evidence.unresolved_declared_projection_count == 0


def test_bartlett_recovers_known_var1_long_run_covariance(tmp_path):
    configuration_count = 70000
    burn_in = 1000
    transition = np.asarray([[0.7, 0.1], [0.0, 0.5]])
    innovation_covariance = np.asarray([[1.0, 0.35], [0.35, 0.7]])
    generator = np.random.default_rng(90210)
    state = generator.multivariate_normal(
        np.zeros(2),
        solve_discrete_lyapunov(transition, innovation_covariance),
    )
    history = np.empty((configuration_count, 2))
    for index in range(configuration_count + burn_in):
        state = (
            transition @ state
            + generator.multivariate_normal(
                np.zeros(2),
                innovation_covariance,
            )
        )
        if index >= burn_in:
            history[index - burn_in] = state

    flow_times = np.arange(1.0, 8.0)
    plaquette = np.asarray(
        [10.0 + 0.1 * flow_time + history[:, 0] for flow_time in flow_times]
    )
    symanzik = np.asarray(
        [20.0 + 0.1 * flow_time + history[:, 1] for flow_time in flow_times]
    )
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": plaquette.tolist(),
        "Es": symanzik.tolist(),
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    result = betafn.RunningCoupling(nf=2).process(
        betafn.ProcessingConfiguration(
            dataset=betafn.DatasetConfiguration(
                path=tmp_path,
                flows=("wilson",),
                observables=("p", "s"),
                times=(1.0, 7.0),
            ),
            correction=betafn.CorrectionConfiguration(
                betafn.Correction.FiniteVolume
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.BartlettLongRunCovariance(
                    maximum_lag=100,
                    wolff_validation=betafn.ProjectedWolffValidation(
                        exponential_scale=1.5,
                        maximum_lag=1000,
                    ),
                )
            ),
        )
    )

    channels = result.ensembles[0].channels
    coupling_pair = np.asarray(
        [channels[0].coupling[0], channels[1].coupling[0]],
        dtype=object,
    )
    energy_means = np.asarray([plaquette[2].mean(), symanzik[2].mean()])
    normalizations = gv.mean(coupling_pair) / energy_means
    estimated_energy_covariance = (
        gv.evalcov(coupling_pair)
        / np.outer(normalizations, normalizations)
    )
    transfer = np.linalg.inv(np.eye(2) - transition)
    exact_energy_covariance = (
        transfer @ innovation_covariance @ transfer.T
        / configuration_count
    )

    np.testing.assert_allclose(
        estimated_energy_covariance,
        exact_energy_covariance,
        rtol=0.05,
        atol=0.0,
    )
    assert (
        channels[0].statistics.source
        == "https://doi.org/10.2307/1913610#equation-5"
    )
    assert (
        channels[0].statistics.wolff_validation.source
        == "https://doi.org/10.1016/S0010-4655(03)00467-3"
    )
    assert (
        channels[0]
        .statistics
        .wolff_validation
        .maximum_relative_variance_difference
        < 0.15
    )
