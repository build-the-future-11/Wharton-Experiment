"""Standard-library exact-arithmetic checks; no fitting or market observations."""

import copy
import unittest
from decimal import ROUND_UP, localcontext
from fractions import Fraction

from successors.laura_planning_v1.core import (
    FlexibilityRule,
    allocate_residual,
    annuity_due_reserve,
    bounded_facility_commitment,
    conditional_2031_range,
    discounted_reserve,
    goal_summary,
    illustrative_2031_range,
    money_record,
    number,
    planning_scenario_grid,
    project_contributions,
    required_constant_growth,
    required_reserve_by_path,
    strategic_sleeves,
)


class LauraArithmeticTests(unittest.TestCase):
    def test_annuity_due_includes_immediate_payment_and_ten_total(self):
        self.assertEqual(annuity_due_reserve(0), 500_000)
        rate = Fraction(1, 25)
        reserve = annuity_due_reserve(rate)
        self.assertEqual(reserve, sum(Fraction(50_000) / (1 + rate) ** k for k in range(10)))
        self.assertEqual(annuity_due_reserve(".04", payment_count=1), 50_000)

    def test_beginning_year_contribution_timing_matches_closed_form_exactly(self):
        rate = Fraction(1, 25)
        value = project_contributions({year: rate for year in range(2027, 2033)})
        self.assertEqual(value, 300_000 * (1 + rate) ** 6 + 150_000 * (1 + rate) ** 5)
        self.assertEqual(project_contributions({year: 0 for year in range(2027, 2033)}), 450_000)

    def test_deposit_timing_creates_sequence_risk_and_total_loss_is_retained(self):
        first = {year: 0 for year in range(2027, 2033)}
        second = first.copy()
        first.update({2027: "-.3", 2028: ".3"})
        second.update({2027: ".3", 2028: "-.3"})
        self.assertNotEqual(project_contributions(first), project_contributions(second))
        loss = {year: 0 for year in range(2027, 2033)}
        loss[2027] = -1
        self.assertEqual(project_contributions(loss), 150_000)
        loss[2032] = -1
        self.assertEqual(project_contributions(loss), 0)

    def test_dated_reserve_requires_matching_curve_and_immediate_par_value(self):
        factors = {year: Fraction(1, 1) / Fraction(26, 25) ** (year - 2033)
                   for year in range(2033, 2043)}
        options = {"known_at": "2026-10-13", "as_of": "2026-10-13", "source": "explicit rate sensitivity"}
        self.assertEqual(discounted_reserve(factors, **options), annuity_due_reserve(".04"))
        factors[2033] = Fraction(99, 100)
        with self.assertRaisesRegex(ValueError, "immediate"):
            discounted_reserve(factors, **options)
        factors.pop(2033)
        with self.assertRaisesRegex(ValueError, "every liability"):
            discounted_reserve(factors, **options)

    def test_future_reserve_yield_cannot_be_an_observed_market_quote(self):
        factors = {year: 1 for year in range(2033, 2043)}
        with self.assertRaisesRegex(ValueError, "future reserve yields"):
            discounted_reserve(factors, known_at="2026-10-13", as_of="2026-10-13",
                               source="future curve", kind="dated_market_quote")
        with self.assertRaisesRegex(ValueError, "not known"):
            discounted_reserve(factors, known_at="2033-01-01", as_of="2026-10-13",
                               source="future curve")

    def test_backward_reserve_recursion_funds_every_payment_exactly(self):
        payments = (50_000,) * 10
        returns = tuple(Fraction(i - 4, 100) for i in range(9))
        balance = required_reserve_by_path(payments, returns)
        for index, payment in enumerate(payments):
            balance -= payment
            self.assertGreaterEqual(balance, 0)
            if index < len(returns):
                balance *= 1 + returns[index]
        self.assertEqual(balance, 0)
        with self.assertRaisesRegex(ValueError, "K-1"):
            required_reserve_by_path(payments, returns + (0,))

    def test_flexibility_is_retained_from_residual_and_conserves_every_dollar(self):
        result = allocate_residual(600_000, 500_000, flexibility=FlexibilityRule(".2"))
        self.assertEqual(result.retained_flexibility, 20_000)
        self.assertEqual(result.facility_capacity, 80_000)
        self.assertEqual(result.allocated_operating_reserve + result.retained_flexibility
                         + result.facility_capacity, result.wealth)
        # The immediate 2033 operating payment is already in the $500,000 reserve.
        self.assertEqual(result.residual_after_operating, 100_000)

    def test_shortfalls_never_turn_into_negative_or_borrowed_facility_capacity(self):
        rule = FlexibilityRule(".2", minimum_cash=25_000)
        insolvent = allocate_residual(450_000, 500_000, flexibility=rule)
        self.assertEqual(insolvent.operating_shortfall, 50_000)
        self.assertEqual(insolvent.flexibility_shortfall, 25_000)
        self.assertEqual(insolvent.facility_capacity, 0)
        thin = allocate_residual(510_000, 500_000, flexibility=rule)
        self.assertEqual(thin.retained_flexibility, 10_000)
        self.assertEqual(thin.flexibility_shortfall, 15_000)
        self.assertEqual(thin.facility_capacity, 0)

    def test_flexibility_edge_policies_and_monotonicity(self):
        self.assertEqual(allocate_residual(600_000, 500_000,
                                          flexibility=FlexibilityRule(0)).facility_capacity, 100_000)
        self.assertEqual(allocate_residual(600_000, 500_000,
                                          flexibility=FlexibilityRule(1)).facility_capacity, 0)
        before = allocate_residual(600_000, 500_000, flexibility=FlexibilityRule(".2"))
        after = allocate_residual(600_000, 510_000, flexibility=FlexibilityRule(".2"))
        self.assertLessEqual(after.facility_capacity, before.facility_capacity)

    def test_scenario_joint_reserve_risk_and_no_invented_probabilities(self):
        result = goal_summary({"down": 450_000, "up": 650_000}, {"down": 500_000, "up": 450_000},
                              flexibility=FlexibilityRule(".2"))
        self.assertEqual(result["facility_capacity_range"][0]["usd"], "0.00")
        self.assertEqual(result["facility_capacity_range"][1]["usd"], "160000.00")
        self.assertIsNone(result["funding_probability"])
        self.assertFalse(result["range_is_calibrated_confidence_interval"])
        self.assertFalse(result["all_listed_scenarios_fund_operating_reserve"])
        with self.assertRaisesRegex(ValueError, "same nonempty named grid"):
            goal_summary({"down": 450_000}, {"up": 500_000}, flexibility=FlexibilityRule(".2"))

    def test_2031_update_uses_only_observed_balance_and_two_remaining_years(self):
        paths = {"flat": {2031: 0, 2032: 0}, "growth": {2031: ".1", 2032: ".1"}}
        reserve = {name: 500_000 for name in paths}
        result = conditional_2031_range(
            600_000, paths, reserve, flexibility=FlexibilityRule(".2"),
            balance_observed_at="2031-01-01", balance_available_at="2031-01-02",
            assumptions_known_at="2031-01-02", as_of="2031-01-03", source="documented 2031 update")
        self.assertEqual(result["balance_kind"], "observed_beginning_2031")
        self.assertEqual(result["scenarios"]["growth"]["wealth"]["usd"], "726000.00")
        self.assertEqual(result["scenarios"]["growth"]["facility_capacity"]["usd"], "180800.00")
        wrong = copy.deepcopy(paths)
        wrong["flat"][2030] = 99
        with self.assertRaisesRegex(ValueError, "exactly 2031 and 2032"):
            illustrative_2031_range(600_000, wrong, reserve, flexibility=FlexibilityRule(".2"),
                                    assumptions_known_at="2026-10-13", as_of="2026-10-13",
                                    source="explicit future sensitivity")

    def test_2026_planning_cannot_label_future_balance_as_observed(self):
        paths, reserve = {"flat": {2031: 0, 2032: 0}}, {"flat": 500_000}
        with self.assertRaisesRegex(ValueError, "available beginning-2031"):
            conditional_2031_range(600_000, paths, reserve, flexibility=FlexibilityRule(".2"),
                                   balance_observed_at="2031-01-01", balance_available_at="2031-01-01",
                                   assumptions_known_at="2026-10-13", as_of="2026-10-13", source="not observed")
        result = illustrative_2031_range(600_000, paths, reserve, flexibility=FlexibilityRule(".2"),
                                         assumptions_known_at="2026-10-13", as_of="2026-10-13",
                                         source="explicit future sensitivity")
        self.assertEqual(result["balance_kind"], "hypothetical_future_balance")

    def test_reject_nonfinite_boolean_and_inconsistent_assumptions(self):
        for value in (True, float("inf"), float("nan"), "NaN", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                number(value)
        for value in (-1, "-1.1"):
            with self.assertRaises(ValueError):
                annuity_due_reserve(value)
        for fraction in ("-0.1", "1.1"):
            with self.assertRaises(ValueError):
                FlexibilityRule(fraction)
        with self.assertRaisesRegex(ValueError, "return years"):
            project_contributions({year: 0 for year in range(2027, 2034)})

    def test_money_display_does_not_change_exact_accounting(self):
        self.assertEqual(money_record(Fraction(1, 3)), {"exact": "1/3", "usd": "0.33"})
        self.assertEqual(money_record(Fraction(1, 200))["usd"], "0.00")
        self.assertEqual(money_record(Fraction(3, 200))["usd"], "0.02")

    def test_money_display_rounding_is_independent_of_ambient_decimal_context(self):
        with localcontext() as context:
            context.rounding = ROUND_UP
            context.prec = 6
            self.assertEqual(money_record(Fraction(1, 200)), {"exact": "1/200", "usd": "0.00"})
            self.assertEqual(money_record(Fraction(3, 200))["usd"], "0.02")
            self.assertEqual(money_record(Fraction(1, 3))["usd"], "0.33")
            self.assertEqual(context.rounding, ROUND_UP)

    def test_all_eighteen_requested_sensitivity_cells_are_preserved(self):
        grid = planning_scenario_grid(flexibility=FlexibilityRule(".20"),
                                      known_at="2026-10-13", as_of="2026-10-13", source="declared sensitivity")
        self.assertEqual(len(grid["rows"]), 18)
        self.assertEqual(len({(row["annual_net_growth_assumption"], row["annual_reserve_discount_assumption"])
                              for row in grid["rows"]}), 18)
        self.assertIsNone(grid["funding_probability"])
        self.assertFalse(grid["uses_wins_balance_or_pnl"])
        self.assertFalse(grid["reserve_rates_are_locked_yields"])
        self.assertEqual(grid["implementation_costs"]["exact"], "0/1")
        self.assertIn("zero-cost planning assumption", grid["implementation_cost_assumption"])
        zero = next(row for row in grid["rows"] if row["annual_net_growth_assumption"] == "0"
                    and row["annual_reserve_discount_assumption"] == "0")
        self.assertEqual(zero["wealth"]["usd"], "450000.00")
        self.assertEqual(zero["signed_surplus_before_flexibility"]["usd"], "-50000.00")
        self.assertEqual(zero["facility_capacity"]["usd"], "0.00")

    def test_strategic_sleeves_never_take_from_the_operating_reserve(self):
        underfunded = strategic_sleeves(450_000, 500_000)
        self.assertEqual(underfunded["equity_target"]["usd"], "0.00")
        self.assertEqual(underfunded["short_usd_liquidity_target"]["usd"], "0.00")
        self.assertEqual(underfunded["operating_shortfall"]["usd"], "50000.00")
        funded = strategic_sleeves(600_000, 500_000)
        self.assertEqual(funded["equity_target"]["usd"], "60000.00")
        self.assertEqual(funded["short_usd_liquidity_target"]["usd"], "40000.00")
        capped = strategic_sleeves(1_000_000, 100_000)
        self.assertEqual(capped["equity_target"]["usd"], "400000.00")
        self.assertEqual(capped["short_usd_liquidity_target"]["usd"], "500000.00")
        self.assertEqual(sum(Fraction(capped[key]["exact"]) for key in
                             ("operating_reserve_allocation", "equity_target", "short_usd_liquidity_target")),
                         1_000_000)

    def test_rebalancing_cannot_spend_cash_supporting_communicated_floor(self):
        # 60k equities lose 40%; 40k pledged-surplus cash remains. A naive
        # 60/40 rebalance buys 9.6k equities from protected cash and breaks it.
        value = strategic_sleeves(576_000, 500_000, secured_surplus_floor=40_000)
        self.assertEqual(value["equity_target"]["usd"], "36000.00")
        self.assertEqual(value["short_usd_liquidity_target"]["usd"], "40000.00")
        self.assertEqual(value["unencumbered_liquidity_target"]["usd"], "0.00")
        self.assertEqual(value["secured_surplus_floor_shortfall"]["usd"], "0.00")
        loss = bounded_facility_commitment(40_000, 40_000, 80_000)
        self.assertEqual(loss["realized_capped_gift"]["usd"], "32000.00")

    def test_secured_surplus_shortfall_is_preserved_without_borrowing(self):
        value = strategic_sleeves(520_000, 500_000, secured_surplus_floor=40_000)
        self.assertEqual(value["equity_target"]["usd"], "0.00")
        self.assertEqual(value["allocated_secured_surplus_floor"]["usd"], "20000.00")
        self.assertEqual(value["secured_surplus_floor_shortfall"]["usd"], "20000.00")
        with self.assertRaises(ValueError):
            strategic_sleeves(600_000, 500_000, secured_surplus_floor=-1)

    def test_required_return_bracket_contains_target_and_is_not_prediction(self):
        zero = required_constant_growth(450_000)
        self.assertEqual(zero["status"], "BRACKETED")
        self.assertEqual(Fraction(zero["lower_rate"]), 0)
        self.assertEqual(Fraction(zero["upper_rate"]), 0)
        needed = required_constant_growth(500_000)
        self.assertEqual(needed["status"], "BRACKETED")
        self.assertLessEqual(Fraction(needed["lower_wealth"]["exact"]), 500_000)
        self.assertGreaterEqual(Fraction(needed["upper_wealth"]["exact"]), 500_000)
        self.assertLessEqual(Fraction(needed["upper_rate"]) - Fraction(needed["lower_rate"]), Fraction(1, 100_000_000))

    def test_implementation_costs_reduce_residual_with_exact_conservation(self):
        result = allocate_residual(600_000, 500_000, flexibility=FlexibilityRule(".2"),
                                   implementation_costs=10_000)
        self.assertEqual(result.allocated_implementation_costs, 10_000)
        self.assertEqual(result.wealth_after_implementation_costs, 590_000)
        self.assertEqual(result.retained_flexibility, 18_000)
        self.assertEqual(result.facility_capacity, 72_000)
        self.assertEqual(result.allocated_implementation_costs + result.allocated_operating_reserve
                         + result.retained_flexibility + result.facility_capacity, result.wealth)
        # Costs cannot leave a $500,000 reserve claimed funded out of only $490,000.
        short = allocate_residual(500_000, 500_000, flexibility=FlexibilityRule(".2"),
                                  implementation_costs=10_000)
        self.assertEqual(short.allocated_operating_reserve, 490_000)
        self.assertEqual(short.operating_shortfall, 10_000)
        self.assertEqual(short.signed_surplus_before_flexibility, -10_000)
        self.assertEqual(short.facility_capacity, 0)

    def test_implementation_cost_insufficiency_is_preserved_and_never_borrowed(self):
        result = allocate_residual(1000, 50_000, flexibility=FlexibilityRule(".2"),
                                   implementation_costs=1500)
        self.assertEqual(result.allocated_implementation_costs, 1000)
        self.assertEqual(result.implementation_cost_shortfall, 500)
        self.assertEqual(result.signed_surplus_before_flexibility, -50_500)
        self.assertEqual(result.allocated_operating_reserve, 0)
        self.assertEqual(result.operating_shortfall, 50_000)
        self.assertEqual(result.facility_capacity, 0)
        self.assertEqual(result.allocated_implementation_costs + result.allocated_operating_reserve
                         + result.retained_flexibility + result.facility_capacity, 1000)

    def test_strategic_costs_use_post_cost_equity_cap_and_conserve_wealth(self):
        result = strategic_sleeves(1_000_000, 100_000, implementation_costs=100_000)
        self.assertEqual(result["wealth_after_implementation_costs"]["usd"], "900000.00")
        self.assertEqual(result["equity_target"]["usd"], "360000.00")
        self.assertEqual(result["short_usd_liquidity_target"]["usd"], "440000.00")
        self.assertEqual(sum(Fraction(result[key]["exact"]) for key in
                             ("allocated_implementation_costs", "operating_reserve_allocation",
                              "equity_target", "short_usd_liquidity_target")), 1_000_000)
        short = strategic_sleeves(100, 500, implementation_costs=200)
        self.assertEqual(short["implementation_cost_shortfall"]["usd"], "100.00")
        self.assertEqual(short["equity_target"]["usd"], "0.00")

    def test_scenario_cost_grid_propagates_and_rejects_missing_cost_scenarios(self):
        result = goal_summary({"down": 100, "up": 600_000}, {"down": 50, "up": 500_000},
                              flexibility=FlexibilityRule(".2"), implementation_costs={"down": 200, "up": 10_000})
        self.assertEqual(result["worst_implementation_cost_shortfall"]["usd"], "100.00")
        self.assertEqual(result["facility_capacity_range"][1]["usd"], "72000.00")
        self.assertFalse(result["all_listed_scenarios_cover_costs_and_operating_reserve"])
        with self.assertRaisesRegex(ValueError, "costs must match"):
            goal_summary({"down": 100, "up": 600_000}, {"down": 50, "up": 500_000},
                         flexibility=FlexibilityRule(".2"), implementation_costs={"up": 10_000})
        grid = planning_scenario_grid(flexibility=FlexibilityRule(".2"), implementation_costs=10_000,
                                      known_at="2026-10-13", as_of="2026-10-13", source="cost sensitivity")
        self.assertEqual(len(grid["rows"]), 18)
        self.assertTrue(all(row["required_implementation_costs"]["usd"] == "10000.00" for row in grid["rows"]))

    def test_conditional_and_illustrative_costs_propagate_and_expired_future_label_rejected(self):
        paths, reserve = {"flat": {2031: 0, 2032: 0}}, {"flat": 500_000}
        actual = conditional_2031_range(600_000, paths, reserve, flexibility=FlexibilityRule(".2"),
                                        implementation_costs=10_000, balance_observed_at="2031-01-01",
                                        balance_available_at="2031-01-01", assumptions_known_at="2031-01-01",
                                        as_of="2031-01-01", source="observed-date conditional plan")
        self.assertEqual(actual["scenarios"]["flat"]["facility_capacity"]["usd"], "72000.00")
        illustrative = illustrative_2031_range(600_000, paths, reserve, flexibility=FlexibilityRule(".2"),
                                               implementation_costs=10_000, assumptions_known_at="2026-10-13",
                                               as_of="2026-10-13", source="future sensitivity")
        self.assertEqual(illustrative["scenarios"]["flat"]["facility_capacity"]["usd"], "72000.00")
        for decision in ("2031-01-01", "2032-01-01"):
            with self.assertRaisesRegex(ValueError, "before 2031-01-01"):
                illustrative_2031_range(600_000, paths, reserve, flexibility=FlexibilityRule(".2"),
                                         assumptions_known_at="2026-10-13", as_of=decision, source="future sensitivity")

    def test_negative_costs_are_rejected_across_public_funding_paths(self):
        rule = FlexibilityRule(".2")
        calls = (
            lambda: allocate_residual(600_000, 500_000, flexibility=rule, implementation_costs=-1),
            lambda: strategic_sleeves(600_000, 500_000, implementation_costs=-1),
            lambda: goal_summary({"a": 600_000}, {"a": 500_000}, flexibility=rule, implementation_costs=-1),
            lambda: planning_scenario_grid(flexibility=rule, implementation_costs=-1,
                                             known_at="2026-10-13", as_of="2026-10-13", source="sensitivity"),
        )
        for call in calls:
            with self.assertRaisesRegex(ValueError, "implementation_costs must be nonnegative"):
                call()

    def test_communicated_gift_cap_retains_upside_and_respects_secured_floor(self):
        for surplus, expected_gift, expected_remaining in ((100_000, 80_000, 20_000),
                                                           (200_000, 80_000, 120_000),
                                                           (40_000, 32_000, 8_000)):
            result = bounded_facility_commitment(surplus, 40_000, 80_000)
            self.assertEqual(Fraction(result["promised_facility_floor"]["exact"]), 32_000)
            self.assertEqual(Fraction(result["realized_capped_gift"]["exact"]), expected_gift)
            self.assertEqual(Fraction(result["remaining_flexibility"]["exact"]), expected_remaining)
            self.assertFalse(result["secured_surplus_floor_breached"])
            self.assertFalse(result["promised_facility_floor_breached"])
            self.assertIsNone(result["funding_probability"])
            self.assertFalse(result["is_unconditional_guarantee"])

    def test_secured_floor_breach_is_not_hidden_by_gift_clipping(self):
        result = bounded_facility_commitment(20_000, 40_000, 80_000)
        self.assertEqual(result["status"], "SECURED_SURPLUS_FLOOR_BREACHED")
        self.assertEqual(result["realized_capped_gift"]["usd"], "16000.00")
        self.assertTrue(result["promised_facility_floor_breached"])
        deficit = bounded_facility_commitment(-5000, 40_000, 80_000)
        self.assertEqual(deficit["surplus_deficit"]["usd"], "5000.00")
        self.assertEqual(deficit["realized_capped_gift"]["usd"], "0.00")
        self.assertEqual(deficit["remaining_flexibility"]["usd"], "0.00")
        with self.assertRaisesRegex(ValueError, "cap cannot be below"):
            bounded_facility_commitment(100_000, 40_000, 30_000)


if __name__ == "__main__":
    unittest.main()
