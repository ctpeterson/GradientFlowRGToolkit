from __future__ import annotations

import pickle

import gfrgtoolkit as betafn
import gvar as gv
import numpy as np
import pytest


def test_running_coupling_processes_and_summarizes_one_ensemble(tmp_path):
    flow_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    offsets = [-0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07]
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            [1.0 + 0.1 * flow_time + offset for offset in offsets]
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l8l8l8t16_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    dataset = betafn.DatasetConfiguration(
        path=tmp_path,
        flows=("wilson",),
        observables=("p",),
        times=(1.0, 7.0),
        predicate=lambda entry: True,
    )
    processing = betafn.ProcessingConfiguration(
        dataset=dataset,
        correction=betafn.CorrectionConfiguration(
            betafn.Correction.FiniteVolume
        ),
        averaging=betafn.StatisticsConfiguration(
            betafn.BartlettLongRunCovariance(maximum_lag=0)
        ),
    )

    result = betafn.RunningCoupling(nf=2).process(processing)
    correction = result.ensembles[0].channels[0].correction

    assert str(result) == (
        "Processing result\n"
        "processed ensembles: 1\n"
        "excluded ensembles: 0\n"
        "measurement channels: wilson/p\n"
        "processed flow-time points: 3\n"
        "statistics: bartlett-newey-west "
        "[source: https://doi.org/10.2307/1913610#equation-5]\n"
        "corrections: finite-volume-normalization "
        "[source: https://doi.org/10.1007/JHEP11(2012)007; "
        "domain: positive finite flow time; numerical tolerance: 1e-15; "
        "implementation: rectangular-torus product extension, "
        "Jacobi theta modular-series evaluation]\n"
        "covariance projections: 0/1 estimates\n"
        "maximum relative covariance adjustment: 0"
    )
    assert correction.method == "finite-volume-normalization"
    assert correction.source == "https://doi.org/10.1007/JHEP11(2012)007"
    assert "Jacobi theta modular-series evaluation" in str(result)


def test_processing_accepts_a_correction_method_at_the_correction_seam(tmp_path):
    flow_times = np.arange(1.0, 8.0)
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            np.linspace(1.0, 1.1, 8).tolist()
            for _flow_time in flow_times
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
                betafn.FiniteVolumeNormalization()
            ),
            averaging=betafn.StatisticsConfiguration(
                betafn.BartlettLongRunCovariance(maximum_lag=0)
            ),
        )
    )

    assert (
        result.ensembles[0].channels[0].correction.method
        == "finite-volume-normalization"
    )


def test_tree_level_processing_matches_the_frozen_reference_values(tmp_path):
    flow_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    offsets = [-0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07]
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            [1.0 + 0.1 * flow_time + offset for offset in offsets]
            for flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l16l16l16t32_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    processing = betafn.ProcessingConfiguration(
        dataset=betafn.DatasetConfiguration(
            path=tmp_path,
            flows=("wilson",),
            observables=("p",),
            times=(1.0, 7.0),
            predicate=lambda entry: True,
        ),
        correction=betafn.CorrectionConfiguration(
            betafn.Correction.FiniteVolumeTreeLevel
        ),
        averaging=betafn.StatisticsConfiguration(
            betafn.BartlettLongRunCovariance(maximum_lag=0)
        ),
    )

    result = betafn.RunningCoupling(nf=2).process(processing)
    channel = result.ensembles[0].channels[0]

    np.testing.assert_allclose(
        np.concatenate((gv.mean(channel.coupling), gv.mean(channel.beta))),
        [
            609.3643539545803,
            1185.0178028167434,
            2002.7145417371535,
            -1393.5183614302118,
            -2771.845324302385,
            -4725.917026508791,
        ],
        rtol=1e-12,
        atol=1e-12,
    )


def test_fvn_and_tln_converge_at_fixed_scheme_as_t_over_a2_grows(tmp_path):
    scheme_ratio = 0.30
    extents = (8, 16, 24, 32)
    offsets = np.linspace(-0.07, 0.07, 8)
    for extent in extents:
        target_time = (scheme_ratio * extent) ** 2 / 8.0
        flow_times = target_time + np.linspace(-0.02, 0.02, 5)
        raw_data = {
            "flow_times": [str(value) for value in flow_times],
            "Ep": [
                (1.0 + offsets).tolist()
                for _flow_time in flow_times
            ],
        }
        ensemble_file = tmp_path / (
            f"7p00_l{extent}l{extent}l{extent}t{extent}_0p00_wilson.bin"
        )
        with ensemble_file.open("wb") as stream:
            pickle.dump(raw_data, stream)

    def process(correction):
        return betafn.RunningCoupling(nf=2, gauge_action="w").process(
            betafn.ProcessingConfiguration(
                dataset=betafn.DatasetConfiguration(
                    path=tmp_path,
                    flows=("wilson",),
                    observables=("p",),
                    times=(0.1, 12.0),
                ),
                correction=betafn.CorrectionConfiguration(correction),
                averaging=betafn.StatisticsConfiguration(
                    betafn.BartlettLongRunCovariance(maximum_lag=0)
                ),
            )
        )

    fvn_result = process(betafn.Correction.FiniteVolume)
    tln_result = process(betafn.Correction.FiniteVolumeTreeLevel)
    fvn_couplings = {
        ensemble.key.volume: float(gv.mean(ensemble.channels[0].coupling[0]))
        for ensemble in fvn_result.ensembles
    }
    tln_couplings = {
        ensemble.key.volume: float(gv.mean(ensemble.channels[0].coupling[0]))
        for ensemble in tln_result.ensembles
    }
    relative_differences = np.asarray(
        [
            abs(
                tln_couplings[f"l{extent}l{extent}l{extent}t{extent}"]
                / fvn_couplings[f"l{extent}l{extent}l{extent}t{extent}"]
                - 1.0
            )
            for extent in extents
        ]
    )

    assert np.all(np.diff(relative_differences) < 0.0)
    assert relative_differences[-1] < 0.012
    assert (
        tln_result.ensembles[-1].channels[0].correction.source
        == "https://doi.org/10.1007/JHEP09(2014)018"
    )
    assert "10.1007/JHEP09(2014)018" in str(tln_result)
    assert "direct requested-time spectral evaluation" in str(tln_result)


def test_tln_rejects_a_non_wilson_flow_instead_of_mislabeling_it(tmp_path):
    flow_times = np.arange(1.0, 8.0)
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": [
            np.linspace(1.0, 1.1, 8).tolist()
            for _flow_time in flow_times
        ],
    }
    ensemble_file = tmp_path / "7p00_l16l16l16t32_0p00_zeuthen.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    with pytest.raises(
        betafn.ProcessingError,
        match="supports only Wilson flow",
    ):
        betafn.RunningCoupling(nf=2).process(
            betafn.ProcessingConfiguration(
                dataset=betafn.DatasetConfiguration(
                    path=tmp_path,
                    flows=("zeuthen",),
                    observables=("p",),
                    times=(1.0, 7.0),
                ),
                correction=betafn.CorrectionConfiguration(
                    betafn.Correction.FiniteVolumeTreeLevel
                ),
                averaging=betafn.StatisticsConfiguration(
                    betafn.BartlettLongRunCovariance(maximum_lag=0)
                ),
            )
        )


def test_processing_records_covariance_projection_evidence(tmp_path):
    configuration_count = 12
    flow_times = np.arange(1.0, 21.0)
    histories = np.random.default_rng(0).normal(
        size=(configuration_count, len(flow_times))
    )
    raw_data = {
        "flow_times": [str(value) for value in flow_times],
        "Ep": histories.T.tolist(),
    }
    ensemble_file = tmp_path / "7p00_l16l16l16t32_0p00_wilson.bin"
    with ensemble_file.open("wb") as stream:
        pickle.dump(raw_data, stream)

    method = betafn.ExperimentalRectangularLongRunCovariance(window_factor=3.0)
    processing = betafn.ProcessingConfiguration(
        dataset=betafn.DatasetConfiguration(
            path=tmp_path,
            flows=("wilson",),
            observables=("p",),
            times=(1.0, 20.0),
        ),
        correction=betafn.CorrectionConfiguration(
            betafn.Correction.FiniteVolume
        ),
        averaging=betafn.StatisticsConfiguration(method),
    )

    result = betafn.RunningCoupling(nf=2).process(processing)
    evidence = result.ensembles[0].channels[0].statistics

    assert evidence.method == method
    assert evidence.configuration_count == configuration_count
    assert evidence.value_count == len(flow_times)
    assert evidence.maximum_lag == 3
    assert evidence.covariance.policy is betafn.CovarianceProjection.NearestPSD
    assert evidence.covariance.projected_mode_count > 0
    assert evidence.covariance.relative_frobenius_adjustment > 0.0
    assert result.covariance_projection_count == 1
    assert (
        result.maximum_relative_covariance_adjustment
        == evidence.covariance.relative_frobenius_adjustment
    )
