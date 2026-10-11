import numpy as np

from reng.funding import (
    LiabilitySchedule,
    funding_frontier,
    resample_monthly_vectors,
    simulate_funding_path,
    sleeve_weights,
)
from reng.portfolio import (
    AllocationPolicy,
    exact_transaction_fee,
    execute_rebalance,
    fixed_60_35_5_reference,
    international_simple_return,
    minimum_variance_weights,
    optimize_allocation,
    project_to_feasible_set,
)


def test_annuity_due_first_payment_is_immediate() -> None:
    schedule = LiabilitySchedule()
    assert schedule.present_value(0.0) == 500_000.0
    assert schedule.present_value(0.05) > sum(50_000 / 1.05**j for j in range(1, 11))


def test_fees_on_buys_and_sells_cannot_create_money() -> None:
    previous = np.array([0.5, 0.0, 0.5])
    target = np.array([0.0, 0.5, 0.5])
    rates = np.array([0.001, 0.001, 0.0])
    fee = exact_transaction_fee(1000.0, target, previous, rates)
    receipt = execute_rebalance(1000.0, target, previous, rates)
    assert fee > 0
    assert np.isclose(receipt["holdings"].sum() + fee, 1000.0)


def test_optimizer_respects_long_only_caps_and_turnover() -> None:
    rng = np.random.default_rng(11)
    scenarios = {
        "model": rng.normal(0.001, 0.01, (300, 4)),
        "empirical": rng.normal(0, 0.012, (300, 4)),
    }
    previous = np.array([0.1, 0.1, 0.1, 0.7])
    result = optimize_allocation(
        np.array([0.01, 0.005, -0.001, 0.0]),
        scenarios,
        previous,
        np.ones(4, dtype=bool),
        np.eye(4) * 1e-4,
    )
    assert result.status in {"OPTIMAL", "HOLD_PRETRADE"}
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.weights[:-1].max() <= 0.2 + 1e-7
    assert np.abs(result.weights - previous).sum() <= 0.4 + 1e-7


def test_infeasible_portfolio_is_not_relaxed_to_cash() -> None:
    scenarios = {"model": np.zeros((20, 3))}
    result = optimize_allocation(
        np.zeros(3),
        scenarios,
        np.array([0.5, 0.5, 0.0]),
        np.array([False, False, True]),
        np.eye(3),
        policy=AllocationPolicy(turnover_cap=0.1),
    )
    assert result.status == "INFEASIBLE"
    assert result.weights is None


def test_international_return_formula() -> None:
    result = international_simple_return(np.array([0.1]), np.array([1.0]), np.array([1.2]))
    assert np.allclose(result, [0.32])


def test_long_horizon_scenarios_and_frontier() -> None:
    rng = np.random.default_rng(11)
    history = rng.normal(0.004, 0.02, (240, 2))
    paths = resample_monthly_vectors(history, 192, 30, block_months=12)
    frontier = funding_frontier(
        paths, np.array([0.5, 0.5]), np.array([0.0, 10_000.0]), facility_month=48
    )
    assert len(frontier) == 2
    assert 0 <= frontier[0].success_frequency <= 1
    failed = simulate_funding_path(np.full(192, -0.9), starting_wealth=1.0)
    assert not failed.success
    assert failed.payment_failures > 0


def test_vectorized_frontier_matches_scalar_accounting() -> None:
    rng = np.random.default_rng(47)
    scenarios = rng.normal(0.002, 0.01, (8, 96, 2))
    weights = np.array([0.4, 0.6])
    contribution = 25_000.0
    terminal_reserve = 10_000.0
    portfolio = scenarios @ weights
    scalar = [
        simulate_funding_path(
            path,
            facility_month=24,
            facility_contribution=contribution,
            terminal_reserve=terminal_reserve,
        )
        for path in portfolio
    ]
    summary = funding_frontier(
        scenarios,
        weights,
        np.array([contribution]),
        facility_month=24,
        terminal_reserve=terminal_reserve,
    )[0]
    assert summary.success_frequency == np.mean([result.success for result in scalar])
    assert np.isclose(
        summary.expected_shortfall, np.mean([result.total_shortfall for result in scalar])
    )
    assert np.isclose(summary.worst_shortfall, max(result.worst_shortfall for result in scalar))
    assert np.isclose(
        summary.terminal_wealth_median, np.median([result.ending_wealth for result in scalar])
    )


def test_sleeves_do_not_double_count_liabilities() -> None:
    weights = sleeve_weights(1_000_000, 50_000, 450_000, 25_000)
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights["GUARD"] == 0.075


def test_portfolio_baselines_share_the_feasible_set() -> None:
    categories = np.array(["us_equity", "international_equity", "usd_fixed_income", "cash"])
    eligible = np.ones(4, dtype=bool)
    reference = fixed_60_35_5_reference(categories, eligible)
    projected = project_to_feasible_set(reference, eligible)
    minimum = minimum_variance_weights(np.eye(4), eligible)
    assert np.isclose(projected.sum(), 1.0)
    assert projected[:-1].max() <= 0.2 + 1e-8
    assert np.isclose(minimum.sum(), 1.0)
