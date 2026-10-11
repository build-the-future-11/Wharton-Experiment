"""Authored but unrun checks for the explicit forecast-to-portfolio bridge.

All values and instrument/source names here are synthetic test fixtures. They
are not data acquisitions, model results, portfolio advice or executed studies.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import frem.decision as decision_module
from frem.cli import main
from frem.decision import (
    LOG_RETURN_SPACE,
    SIMPLE_RETURN_SPACE,
    BoundedScenarioConfig,
    ReturnContract,
    bounded_forecast_scenarios,
    decide_from_forecasts,
)
from frem.mandate import Instrument, Mandate, Provenance
from frem.portfolio import AllocationResult

AS_OF = "2026-10-10T12:00:00Z"
IDS = ["SYNTHETIC_A", "SYNTHETIC_B"]


@pytest.fixture
def contract():
    return ReturnContract(
        LOG_RETURN_SPACE,
        "USD",
        IDS,
        AS_OF,
        AS_OF,
        5,
        "2026-10-12T13:30:00Z",
        "2026-10-19T13:30:00Z",
        True,
        "Artificial shared clock for prospective unit checks",
        "Synthetic fixture only",
    )


@pytest.fixture
def forecasts():
    return pd.DataFrame(
        {
            "asset": IDS,
            "asset_id": [0, 1],
            "time": [AS_OF, AS_OF],
            "prediction": [0.0, 0.01],
            "scale": [0.05, 0.04],
            "student_df": [5.0, 7.0],
        }
    )


@pytest.fixture
def scenario_config():
    return BoundedScenarioConfig(
        64,
        11,
        -0.20,
        0.20,
        "independent",
        "Explicit independence for a synthetic unit fixture",
        AS_OF,
        "test_decision.py",
    )


@pytest.fixture
def mandate_and_master():
    source = Provenance(
        "fixture",
        "Synthetic evidence",
        "test_decision.py",
        "user_document",
        "2026-10-10T00:00:00Z",
        "2026-10-10T00:00:00Z",
    )
    policy = replace(source, source_id="policy", kind="team_policy")
    mandate = Mandate(
        "synthetic",
        "Synthetic client",
        "USD",
        status="reviewed_research",
        sources={"fixture": source, "policy": policy},
        participant_rules_source_id="fixture",
        universe_source_id="fixture",
        policy_source_id="policy",
        allowed_markets=["US"],
        allowed_asset_classes=["equity"],
        eligible_asset_ids=IDS,
        min_cash_weight=0,
        max_cash_weight=1,
        max_asset_weight=1,
        max_turnover=2,
        max_trade_adv_fraction=0.1,
        max_position_adv_days=5,
        max_foreign_currency_weight=1,
    )
    master = [
        Instrument(
            asset_id=x,
            symbol=x,
            market="US",
            sector="synthetic",
            asset_class="equity",
            currency="USD",
            available_at="2026-10-10T00:00:00Z",
            metadata_asof="2026-10-10T00:00:00Z",
            approval_source_id="fixture",
            approved=True,
            tradable=True,
            price_base=1,
            average_daily_value_base=100_000,
            exposure_verified=True,
            classification_source_id="fixture",
        )
        for x in IDS
    ]
    return mandate, master


def bundle(forecasts, contract, config):
    return bounded_forecast_scenarios(
        forecasts, contract, config, as_of=AS_OF, asset_ids=IDS, base_currency="USD"
    )


def test_realized_outcome_columns_do_not_change_decision_scenarios(forecasts, contract, scenario_config):
    reference = bundle(forecasts, contract, scenario_config)
    changed = forecasts.assign(actual=[-1000, 1000], label_time=[None, "2099-01-01"])
    alternative = bundle(changed, contract, scenario_config)
    np.testing.assert_array_equal(reference.simple_returns, alternative.simple_returns)


def test_forecast_row_order_cannot_change_asset_alignment(forecasts, contract, scenario_config):
    first = bundle(forecasts, contract, scenario_config)
    second = bundle(forecasts.iloc[::-1], contract, scenario_config)
    np.testing.assert_array_equal(first.simple_returns, second.simple_returns)
    assert [row["asset"] for row in second.forecasts] == IDS


def test_bounds_and_same_seed_are_reproducible(forecasts, contract, scenario_config):
    first = bundle(forecasts, contract, scenario_config)
    second = bundle(forecasts, contract, scenario_config)
    np.testing.assert_array_equal(first.simple_returns, second.simple_returns)
    assert np.all(first.simple_returns >= np.expm1(-0.20))
    assert np.all(first.simple_returns <= np.expm1(0.20))
    assert first.assumptions["covariance_convention"].startswith("weighted finite-distribution")


def test_single_positive_scenario_weight_has_zero_population_variance(forecasts, contract, scenario_config):
    configured = replace(
        scenario_config,
        n_scenarios=2,
        probabilities=[1.0, 0.0],
        weighting_rationale="Degenerate probability fixture, not a market belief",
    )
    result = bundle(forecasts, contract, configured)
    np.testing.assert_allclose(result.expected_simple_returns, result.simple_returns[0], atol=1e-14)
    np.testing.assert_allclose(result.simple_return_covariance, 0, atol=1e-14)


def test_perfect_copula_correlation_preserves_identical_marginals(forecasts, contract, scenario_config):
    same = forecasts.assign(prediction=0.0, scale=0.05, student_df=5.0)
    configured = replace(
        scenario_config,
        dependence="gaussian_copula",
        correlation=[[1, 1], [1, 1]],
        correlation_available_at=AS_OF,
        correlation_source="Synthetic singular correlation fixture",
    )
    result = bundle(same, contract, configured)
    np.testing.assert_allclose(result.simple_returns[:, 0], result.simple_returns[:, 1], atol=1e-12)


def test_missing_dependence_is_not_implicitly_independent(forecasts, contract, scenario_config):
    with pytest.raises(ValueError, match="explicitly"):
        bundle(forecasts, contract, replace(scenario_config, dependence=""))


def test_future_correlation_is_rejected(forecasts, contract, scenario_config):
    configured = replace(
        scenario_config,
        dependence="gaussian_copula",
        correlation=np.eye(2),
        correlation_available_at="2026-10-11T00:00:00Z",
        correlation_source="Unavailable fixture",
    )
    with pytest.raises(ValueError, match="unavailable"):
        bundle(forecasts, contract, configured)


def test_simple_return_claim_cannot_enter_log_forecast_bridge(forecasts, contract, scenario_config):
    with pytest.raises(ValueError, match="return_space"):
        bundle(forecasts, replace(contract, return_space=SIMPLE_RETURN_SPACE), scenario_config)


def test_forecast_clocks_must_match(forecasts, contract, scenario_config):
    changed = forecasts.copy()
    changed.loc[1, "time"] = "2026-10-10T11:59:00Z"
    with pytest.raises(ValueError, match="decision timestamp"):
        bundle(changed, contract, scenario_config)


def test_float_timestamp_does_not_lose_nanosecond_precision(contract):
    with pytest.raises(ValueError, match="integer UTC"):
        replace(contract, forecast_time=1.0).validate(
            as_of=AS_OF, asset_ids=IDS, base_currency="USD", expected_return_space=LOG_RETURN_SPACE
        )


def test_local_integer_asset_id_cannot_substitute_stable_identifier(forecasts, contract, scenario_config):
    with pytest.raises(ValueError, match="columns"):
        bundle(forecasts.drop(columns="asset"), contract, scenario_config)


def test_pending_mandate_stops_before_scenario_generation(
    forecasts, contract, scenario_config, mandate_and_master, monkeypatch
):
    mandate, master = mandate_and_master
    mandate.status = "requires_inputs"

    def forbidden(*args, **kwargs):
        raise AssertionError("No scenario generation for an incomplete mandate")

    monkeypatch.setattr(decision_module, "bounded_forecast_scenarios", forbidden)
    result = decide_from_forecasts(
        forecasts,
        master,
        mandate,
        contract=contract,
        scenario_config=scenario_config,
        as_of=AS_OF,
        nav=1000,
        current_weights=[0, 0],
    )
    assert result["status"] == "blocked"
    assert result["allocation"] is None


def test_bridge_passes_derived_simple_scenarios_to_same_constraint_optimizer(
    forecasts, contract, scenario_config, mandate_and_master, monkeypatch
):
    mandate, master = mandate_and_master
    captured = {}

    def optimizer(mean, covariance, instruments, received_mandate, **kwargs):
        captured.update(
            mean=mean, covariance=covariance, instruments=instruments, mandate=received_mandate, kwargs=kwargs
        )
        return AllocationResult("fallback_cash", {x: 0.0 for x in IDS}, 1.0, "normal", {"synthetic": True})

    monkeypatch.setattr(decision_module, "optimize_portfolio", optimizer)
    result = decide_from_forecasts(
        forecasts,
        master,
        mandate,
        contract=contract,
        scenario_config=scenario_config,
        as_of=AS_OF,
        nav=1000,
        current_weights=[0, 0],
        reserve_cash=100,
    )
    assert result["status"] == "fallback_cash"
    assert captured["mandate"] is mandate
    assert captured["kwargs"]["reserve_cash"] == 100
    np.testing.assert_array_equal(captured["kwargs"]["scenarios"], result["scenarios"]["returns"])
    assert result["scenario_sha256"]
    assert result["research_only"] is True


@pytest.mark.parametrize("command", ["allocate", "decide"])
def test_cli_requires_explicit_execute_before_reading_request(command, tmp_path):
    destination = tmp_path / "no_execution.json"
    assert main([command, "--request", str(tmp_path / "absent.json"), "--output", str(destination)]) == 2
    assert not destination.exists()
