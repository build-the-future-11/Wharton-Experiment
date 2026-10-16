import json
from dataclasses import asdict

import numpy as np
import pytest

from successors.frem_s6.experiment import (
    economics_from_records,
    ensemble_records,
    paired_date_interval,
    planned_cells,
    run_cell,
    validate_protocol,
)
from successors.frem_s6.model import MixtureForecast, ModelConfig


def tiny_integrity_data():
    # A numerical integration fixture, never used as a financial result.
    rng = np.random.default_rng(81)
    x = rng.normal(0, .02, (20, 4, 2))
    y = rng.normal(0, .01, 20)
    data = {"currency": "USD", "fit_cutoff": 13}
    for name, sl, dates in (("train", slice(0, 12), np.arange(12)),
                            ("calibration", slice(12, 16), np.arange(14, 18)),
                            ("evaluation", slice(16, 20), np.array([20., 20., 23., 23.]))):
        n = len(x[sl])
        data[name] = {"windows": x[sl], "y": y[sl], "origin_times": dates,
                      "maturity_times": dates + 1, "available_times": dates + 1,
                      "assets": np.array((["a", "b"] * 2) if name == "evaluation" else ["a"] * n),
                      "simple_returns": np.expm1(y[sl])}
    data["train"]["future_windows"] = x[:12] * .5
    data["train"]["future_available"] = np.tile(np.arange(12)[:, None] + 1, (1, 4))
    return data


def test_end_to_end_fit_delayed_scores_ledgers_and_checkpoint(tmp_path):
    rows, receipt = run_cell(tiny_integrity_data(), ModelConfig(2, epochs=3, state_dim=2, latent_dim=2, rff_dim=2),
                             tmp_path / "cell")
    assert receipt["status"] == "COMPLETE_DEVELOPMENT_CELL"
    assert receipt["evaluation_dates"] == 2
    assert len(rows) == 16
    economics = json.loads((tmp_path / "cell/economics.json").read_text())
    for ledgers in economics.values():
        assert ledgers["cash"]["terminal_wealth"] == 100000
        for ledger in ledgers.values():
            last = ledger["periods"][-1]
            # Independent terminal accounting reconstruction.
            nav = last["cash"] + sum(last["holdings"].values()) - last["liquidation_fee"]
            assert nav == pytest.approx(ledger["terminal_wealth"])
            assert ledger["fees"] >= 0
    for r in rows:
        f = MixtureForecast(r["mean"], r["scale"], r["df"], r["weights"],
                            tuple(str(i) for i in range(len(r["mean"][0]))))
        np.testing.assert_allclose(f.nll(r["y"]), r["nll"], atol=1e-14)
    combined = ensemble_records([rows, rows, rows])
    np.testing.assert_allclose([r["date_nll"] for r in combined], [r["date_nll"] for r in rows], atol=1e-14)
    pooled_economics = economics_from_records(combined, "USD")
    assert pooled_economics["10"]["memory"]["terminal_wealth"] == pytest.approx(
        economics["10"]["memory"]["terminal_wealth"])
    with pytest.raises(FileExistsError):
        run_cell(tiny_integrity_data(), ModelConfig(2), tmp_path / "cell")


def test_complete_predeclared_matrix_and_no_selected_seed():
    matrix = planned_cells(2)
    assert len(matrix) == 24
    assert len({name for name, _ in matrix}) == 24
    assert sorted({config.seed for _, config in matrix}) == [11, 29, 47]
    assert sum(config.context == "zero" for _, config in matrix) == 6
    assert sum(config.jepa_weight == 0 for _, config in matrix) == 3
    assert sum(config.quantization_step == 0 and config.ripple_delay == 0 for _, config in matrix) == 3
    with pytest.raises(ValueError):
        ensemble_records([[], [{}]])


def test_intervals_keep_insufficient_dates_and_constant_effect():
    assert paired_date_interval([.1] * 10, block_dates=3)["status"] == "INSUFFICIENT_DATES"
    result = paired_date_interval([.1] * 24, block_dates=6, repetitions=100)
    np.testing.assert_allclose(result["interval95"], [.1, .1])


def test_protocol_binds_budget_dates_precision_and_spent_inventory():
    data = tiny_integrity_data()
    data["evaluation"]["origin_times"] = np.arange(30., 54.)
    p = {"cell_configurations": {name: asdict(c) for name, c in planned_cells(2)},
         "fold_dates": {part: np.unique(data[part]["origin_times"]).tolist()
                        for part in ("train", "calibration", "evaluation")},
         "precision_analysis": {"required_evaluation_dates": 24, "block_dates": [3, 6],
                                "development_effect_analysis": "test-only declared analysis"},
         "spent_date_inventory": {"source": "integrity fixture", "dates": [1, 2]},
         "primary_endpoint": "mean_date_level_mixture_nll", "economic_policy": "shared_fixed_80_20_signal"}
    validate_protocol(p, data, with_xgboost=False)
    for bad in (["30"], [True], [float("nan")], [30.]):
        p["spent_date_inventory"]["dates"] = bad
        with pytest.raises(ValueError):
            validate_protocol(p, data, with_xgboost=False)
