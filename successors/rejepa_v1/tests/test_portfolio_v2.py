import numpy as np
import pytest

from rejepa.international import aggregate_exposure, currency_adjusted_return, diversification_delta
from rejepa.portfolio import (
    ConstrainedInstrument,
    ScenarioAllocationPolicy,
    minimum_variance_weights,
    optimize_scenario_cvar,
)


def _instruments() -> list[ConstrainedInstrument]:
    return [
        ConstrainedInstrument("US", "Tech", "US", "USD", True, True, 0.25),
        ConstrainedInstrument("JP", "Industrial", "JP", "JPY", True, True, 0.20),
        ConstrainedInstrument("TW", "Tech", "TW", "TWD", True, True, 0.20),
    ]


def test_scenario_cvar_optimizer_obeys_caps() -> None:
    scenarios = np.array([[0.02, -0.01, -0.03], [-0.05, 0.01, -0.02], [0.01, 0.02, 0.01]])
    result = optimize_scenario_cvar(
        np.array([0.05, 0.04, 0.06]),
        scenarios,
        np.array([0.1, 0.2, 0.3]),
        np.zeros(3),
        _instruments(),
        ScenarioAllocationPolicy(min_cash=0.1, max_non_base_currency=0.3),
        data_age_days=1,
    )
    assert result["status"] == "optimized_research_only"
    assert result["solver"] == "scipy.optimize.linprog-highs"
    assert result["cash"] >= 0.1 - 1e-6
    assert result["weights"][1:].sum() <= 0.3 + 1e-6


def test_stale_data_falls_back_to_cash() -> None:
    result = optimize_scenario_cvar(
        np.zeros(3),
        np.zeros((3, 3)),
        np.zeros(3),
        np.zeros(3),
        _instruments(),
        ScenarioAllocationPolicy(),
        data_age_days=30,
    )
    assert result["status"] == "fallback_cash"
    assert result["cash"] == 1.0


def test_international_off_policy_is_valid() -> None:
    policy = ScenarioAllocationPolicy(max_non_base_currency=0.0)
    assert policy.max_non_base_currency == 0.0


def test_scenario_policy_and_current_wealth_fail_closed() -> None:
    with pytest.raises(ValueError):
        ScenarioAllocationPolicy(max_sector=1.1)
    with pytest.raises(ValueError, match="total wealth"):
        optimize_scenario_cvar(
            np.zeros(3),
            np.zeros((3, 3)),
            np.zeros(3),
            np.ones(3),
            _instruments(),
            ScenarioAllocationPolicy(),
            data_age_days=1,
        )


def test_minimum_variance_and_currency_math() -> None:
    weights = minimum_variance_weights(np.diag([0.01, 0.04]))
    assert weights.sum() == pytest.approx(1.0)
    adjusted = currency_adjusted_return(np.array([0.10]), np.array([-0.05]))
    assert adjusted[0] == pytest.approx(0.045)


def test_international_exposure_and_diversification() -> None:
    exposure = aggregate_exposure(np.array([0.5, 0.3, 0.2]), ["US", "JP", "US"], eligible=[True] * 3)
    assert exposure == {"US": 0.7, "JP": 0.3}
    result = diversification_delta(np.array([0.1, -0.1, 0.1, -0.1]), np.array([-0.1, 0.1, -0.1, 0.1]), 0.5)
    assert result["mixed_volatility"] < result["domestic_volatility"]
