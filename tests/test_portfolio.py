"""Prospective verification tests, now covered by runtime verification.

All instrument identities, prices, sources and allocations in this file are
explicit synthetic fixtures. None describe a verified live/client portfolio.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import frem.portfolio as portfolio_module
from frem.mandate import (
    CashFlow,
    Instrument,
    Mandate,
    Provenance,
    RuleRecord,
    annuity_due_reserve,
    conditional_2031_analysis,
    discounted_cashflow_reserve,
    project_contributions,
    required_reserve_by_path,
    with_market_policy_cap,
)
from frem.portfolio import (
    AllocationResult,
    PortfolioConfig,
    optimize_portfolio,
    round_allocation_to_lots,
    validate_allocation,
)
from frem.risk import (
    RegimeInputs,
    base_currency_simple_returns,
    bond_total_return_from_shock,
    choose_regime,
    gaussian_log_to_simple_moments,
    validate_covariance,
    weighted_cvar,
)

AS_OF = "2026-10-10T20:00:00Z"


@pytest.fixture
def mandate():
    fixture_source = Provenance(
        "fixture",
        "Synthetic test-only input",
        "test_portfolio.py",
        "user_document",
        "2026-10-10T00:00:00Z",
        "2026-10-10T00:00:00Z",
        scope="Artificial fixture, no actual organizer/client source",
    )
    policy_source = replace(fixture_source, source_id="policy", kind="team_policy")
    return Mandate(
        mandate_id="synthetic-test-only",
        client_name="Synthetic test fixture",
        base_currency="USD",
        status="reviewed_research",
        sources={"fixture": fixture_source, "policy": policy_source},
        participant_rules_source_id="fixture",
        universe_source_id="fixture",
        policy_source_id="policy",
        allowed_markets=["US", "TW"],
        allowed_asset_classes=["equity", "bond", "equity_etf"],
        eligible_asset_ids=["TEST_US", "TEST_TW"],
        min_cash_weight=0,
        max_cash_weight=1,
        max_asset_weight=1,
        max_turnover=2,
        max_trade_adv_fraction=0.10,
        max_position_adv_days=5,
        max_foreign_currency_weight=1,
    )


@pytest.fixture
def instruments():
    return [
        Instrument(
            asset_id=asset_id,
            symbol=asset_id,
            market=market,
            sector="technology",
            asset_class="equity",
            currency="USD",
            available_at="2026-10-10T00:00:00Z",
            metadata_asof="2026-10-10T00:00:00Z",
            approval_source_id="fixture",
            approved=True,
            tradable=True,
            price_base=1.0,
            average_daily_value_base=1_000_000.0,
            lot_size=1,
            exposure_verified=True,
            classification_source_id="fixture",
        )
        for asset_id, market in (("TEST_US", "US"), ("TEST_TW", "TW"))
    ]


def failed_solver(fun, seed, **kwargs):
    return SimpleNamespace(success=False, x=np.asarray(seed), message="deliberate test failure", nit=0)


def test_annuity_due_immediate_payment_is_not_discounted():
    assert annuity_due_reserve(100, 2, 1.0) == pytest.approx(150.0)
    assert annuity_due_reserve(50_000, 10, 0) == pytest.approx(500_000.0)


def test_invalid_regime_input_cannot_hide_behind_early_cash_threshold():
    regime, reasons = choose_regime(
        RegimeInputs(recession_probability=0.99, annualized_volatility=float("nan"))
    )
    assert regime == "no_action"
    assert reasons


def test_string_quality_flag_cannot_claim_valid_data():
    regime, reasons = choose_regime(RegimeInputs(data_valid="False"))
    assert regime == "no_action"
    assert "Boolean" in reasons[0]


def test_missing_forecast_uses_explicit_risk_only_fallback(mandate, instruments, monkeypatch):
    monkeypatch.setattr(portfolio_module, "minimize", failed_solver)
    result = optimize_portfolio(
        None,
        np.eye(2) * 0.01,
        instruments,
        mandate,
        as_of=AS_OF,
        nav=1_000.0,
        current_weights=[0, 0],
        decision_context=RegimeInputs(forecast_available=False),
    )
    assert result.status == "fallback_cash"
    assert result.diagnostics["expected_simple_return_net_cost"] is None


def test_reserve_accepts_negative_yield_and_rejects_minus_one():
    assert annuity_due_reserve(100, 2, -0.5) == pytest.approx(300.0)
    with pytest.raises(ValueError):
        annuity_due_reserve(100, 2, -1.0)


def test_immediate_discount_curve_payment_requires_one():
    due = [CashFlow("2033-01-01", 50_000, "operating")]
    with pytest.raises(ValueError, match="immediate"):
        discounted_cashflow_reserve(due, "2033-01-01", [0.98])


def test_contribution_timing_has_six_and_five_growth_intervals():
    contributions = [
        CashFlow("2027-01-01", 300_000, "contribution"),
        CashFlow("2028-01-01", 150_000, "contribution"),
    ]
    # Extreme arithmetic fixture tests the exponent convention, not a forecast.
    rates = {year: 1.0 for year in range(2027, 2033)}
    assert project_contributions(contributions, rates, 2033) == pytest.approx(24_000_000.0)


def test_2031_interval_uses_two_forward_years_only():
    result = conditional_2031_analysis(1_000, [[0, 0], [-0.1, 0.1]], 500, 100, probabilities=[0.5, 0.5])
    assert result["projected_portfolio_values"] == pytest.approx([1_000, 990])
    assert result["facility_capacity"] == pytest.approx([400, 390])
    assert result["projection_years"] == [2031, 2032]
    with pytest.raises(ValueError, match="scenarios,2"):
        conditional_2031_analysis(1_000, [[0, 0, 0]], 500, 100)


def test_backward_reserve_includes_payment_before_growth():
    required = required_reserve_by_path([100, 100], [[1.0], [0.0], [-0.5]])
    assert required == pytest.approx([150, 200, 300])


def test_weighted_cvar_uses_fractional_quantile_atom():
    # Worst 5% contains a 2% loss atom and 3% zero-loss mass.
    assert weighted_cvar([0, 100], 0.95, [0.98, 0.02]) == pytest.approx(40)


def test_fx_return_uses_multiplicative_composition():
    assert float(base_currency_simple_returns(0.10, -0.20)) == pytest.approx(-0.12)


def test_bond_scenario_distinguishes_yield_changes_from_price_returns():
    assert bond_total_return_from_shock(5.0, 30.0, 0.01) == pytest.approx(-0.0485)


def test_covariance_rejects_non_psd_matrix():
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance([[1, 2], [2, 1]])


def test_gaussian_log_conversion_is_explicit():
    means, covariance = gaussian_log_to_simple_moments([0.0], [[0.0]])
    assert means == pytest.approx([0.0])
    assert covariance == pytest.approx(np.zeros((1, 1)))


def test_recession_policy_does_not_assume_recession():
    assert choose_regime(RegimeInputs())[0] == "normal"
    assert choose_regime(RegimeInputs(recession_probability=0.90))[0] == "cash"
    assert choose_regime(RegimeInputs(calibration_valid=False))[0] == "no_action"


def test_missing_official_participant_material_blocks_allocation(mandate, instruments):
    pending = replace(mandate, participant_rules_source_id=None)
    result = optimize_portfolio(
        [0.01, 0.01], np.eye(2), instruments, pending, as_of=AS_OF, nav=1_000, current_weights=[0, 0]
    )
    assert result.status == "blocked"
    assert result.weights is None


def test_missing_holdings_does_not_silently_assume_all_cash(mandate, instruments):
    result = optimize_portfolio([0.01, 0.01], np.eye(2), instruments, mandate, as_of=AS_OF, nav=1_000)
    assert result.status == "blocked"
    assert "current holdings" in result.diagnostics["blocking_reasons"][0]


def test_taiwan_10_percent_policy_remains_distinct_from_law(mandate):
    source = mandate.sources["policy"]
    capped = with_market_policy_cap(mandate, "TW", 0.10, source)
    assert capped.market_weight_caps["TW"] == 0.10
    assert capped.rules[-1].rule_type == "market_weight_cap"
    assert "not a foreign-investment law" in capped.rules[-1].interpretation
    assert not mandate.market_weight_caps


def test_daily_price_band_cannot_be_encoded_as_allocation_rule(mandate):
    wrong = replace(
        mandate,
        rules=[RuleRecord("wrong", "daily_price_band", "TW", 0.1, "fixture", "Mislabelled price band", True)],
    )
    assert any("daily price band" in issue for issue in wrong.validate(AS_OF))


def test_unsupported_mandatory_constraint_blocks_instead_of_being_ignored(mandate):
    wrong = replace(
        mandate,
        rules=[
            RuleRecord(
                "min-names",
                "minimum_distinct_holdings",
                "portfolio",
                5,
                "fixture",
                "A hypothetical participant requirement",
                True,
            )
        ],
    )
    assert any("unsupported mandatory" in issue for issue in wrong.validate(AS_OF))


def test_feasible_cash_fallback_obeys_country_cap(monkeypatch, mandate, instruments):
    monkeypatch.setattr(portfolio_module, "minimize", failed_solver)
    mandate = replace(mandate, market_weight_caps={"TW": 0.1})
    result = optimize_portfolio(
        [0.01, 0.10], np.eye(2), instruments, mandate, as_of=AS_OF, nav=1_000, current_weights=[0.4, 0.2]
    )
    assert result.status == "fallback_cash"
    assert result.weights["TEST_TW"] == pytest.approx(0)
    assert result.cash_weight == pytest.approx(1)
    assert result.diagnostics["final_check"]["feasible"]


def test_fallback_declares_infeasible_when_turnover_blocks_mandatory_exit(monkeypatch, mandate, instruments):
    monkeypatch.setattr(portfolio_module, "minimize", failed_solver)
    mandate = replace(mandate, max_turnover=0.0, market_weight_caps={"TW": 0.1})
    result = optimize_portfolio(
        [0.01, 0.10], np.eye(2), instruments, mandate, as_of=AS_OF, nav=1_000, current_weights=[0.4, 0.2]
    )
    assert result.status == "infeasible"
    assert result.weights is None
    assert result.cash_weight is None


def test_solver_success_cannot_override_constraint_check(monkeypatch, mandate, instruments):
    def fake_success(fun, seed, **kwargs):
        return SimpleNamespace(success=True, x=np.array([0.0, 0.8]), message="fake success", nit=1)

    monkeypatch.setattr(portfolio_module, "minimize", fake_success)
    mandate = replace(mandate, max_cash_weight=0.5, market_weight_caps={"TW": 0.1})
    result = optimize_portfolio(
        [0.01, 0.1], np.eye(2), instruments, mandate, as_of=AS_OF, nav=1_000, current_weights=[0.6, 0.05]
    )
    assert result.status == "fallback_hold"
    assert result.weights["TEST_TW"] == pytest.approx(0.05)
    assert result.diagnostics["solver_attempts"][0]["postcheck_feasible"] is False


def test_transaction_costs_cannot_spend_required_cash_reserve(mandate, instruments):
    result = validate_allocation(
        [0.8, 0],
        instruments,
        mandate,
        as_of=AS_OF,
        nav=100,
        current_weights=[0, 0],
        reserve_cash=20,
        config=PortfolioConfig(cost_bps=100),
    )
    assert not result["feasible"]
    assert "cash_reserve_amount" in result["violations"]
    assert result["cash_amount"] < 20


def test_future_metadata_excludes_instrument(mandate, instruments):
    future = replace(instruments[1], available_at="2026-10-11T00:00:00Z")
    result = validate_allocation(
        [0.1, 0.1], [instruments[0], future], mandate, as_of=AS_OF, nav=100, current_weights=[0, 0]
    )
    assert not result["feasible"]
    assert "security_cap:TEST_TW" in result["violations"]


def test_etf_fractional_exposures_count_against_country_cap(mandate, instruments):
    fund = replace(instruments[0], asset_class="equity_etf", market_exposures={"US": 0.7, "TW": 0.3})
    constrained = replace(mandate, market_weight_caps={"TW": 0.10})
    result = validate_allocation(
        [0.5, 0], [fund, instruments[1]], constrained, as_of=AS_OF, nav=100, current_weights=[0, 0]
    )
    assert not result["feasible"]
    assert result["violations"]["cap:market:TW"] == pytest.approx(0.05)


def test_lot_rounding_rechecks_original_group_floors(mandate, instruments):
    instruments = [replace(instruments[0], lot_size=100), instruments[1]]
    mandate = replace(mandate, market_weight_floors={"US": 0.4})
    continuous = AllocationResult("optimized", {"TEST_US": 0.5, "TEST_TW": 0}, 0.5, "normal", {})
    rounded = round_allocation_to_lots(
        continuous,
        instruments,
        mandate,
        as_of=AS_OF,
        nav=105,
        current_quantities=[0, 0],
        config=PortfolioConfig(cost_bps=0),
    )
    assert rounded["status"] == "infeasible"
    assert rounded["quantities"] is None


def test_false_string_is_not_approved_metadata(instruments):
    data = instruments[0].__dict__.copy()
    data["approved"] = "False"
    with pytest.raises(ValueError, match="Boolean"):
        Instrument.from_mapping(data)
