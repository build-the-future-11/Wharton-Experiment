"""Exact conditional Laura funding arithmetic, independent of an ML checkpoint.

The cash-flow formulas follow src/frem/mandate.py. This isolated bridge adds
explicit observation/assumption dates, scenario-specific reserve costs, and a
residual flexibility rule. It has no market-data, brokerage, NumPy or SciPy
dependency. Scenario ranges are extrema of named assumptions, never calibrated
confidence intervals. No WInS account value is accepted into Laura's cash flows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from types import MappingProxyType
from typing import Any


def number(value: Any, name: str = "number") -> Fraction:
    """Interpret decimal text/JSON numerals exactly; reject Boolean/nonfinite input."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal, Fraction)):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    return result


def _nonnegative(value: Any, name: str) -> Fraction:
    result = number(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _year(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 9999:
        raise ValueError("year must be an integer from 1 through 9999")
    return value


def _date(value: str | date) -> date:
    if isinstance(value, datetime):
        raise ValueError("use explicit ISO calendar dates, not datetimes")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("invalid ISO calendar date") from exc
    raise ValueError("date must be an ISO calendar date")


def _known(known_at: str | date, as_of: str | date, source: str) -> tuple[date, date]:
    known, decision = _date(known_at), _date(as_of)
    if known > decision:
        raise ValueError("input was not known by the decision date")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("an explicit source or assumption description is required")
    return known, decision


def money_record(value: Any) -> dict[str, str]:
    """Serializable exact ratio plus rounded USD display; display never drives math."""
    exact = number(value)
    with localcontext() as context:
        context.prec = max(60, len(str(abs(exact.numerator))) + len(str(exact.denominator)) + 5)
        context.rounding = ROUND_HALF_EVEN
        decimal = Decimal(exact.numerator) / Decimal(exact.denominator)
        display = format(decimal.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN), "f")
    return {"exact": f"{exact.numerator}/{exact.denominator}", "usd": display}


LAURA_CONTRIBUTIONS = MappingProxyType({2027: Fraction(300_000), 2028: Fraction(150_000)})
LAURA_LIABILITIES = MappingProxyType({year: Fraction(50_000) for year in range(2033, 2043)})


def annuity_due_reserve(annual_rate: Any, *, annual_payment: Any = 50_000,
                        payment_count: int = 10) -> Fraction:
    """Payment at t=0 is included without discount; no end-2042 growth is needed."""
    rate = number(annual_rate, "annual_rate")
    payment = _nonnegative(annual_payment, "annual_payment")
    if rate <= -1 or type(payment_count) is not int or payment_count < 1:
        raise ValueError("rate must exceed -1 and payment_count must be a positive integer")
    return sum((payment / (1 + rate) ** offset for offset in range(payment_count)), Fraction())


def discounted_reserve(discount_factors: Mapping[int, Any], *,
                       valuation_year: int = 2033, liabilities: Mapping[int, Any] | None = None,
                       known_at: str | date, as_of: str | date, source: str,
                       kind: str = "scenario_assumption") -> Fraction:
    """Price dated nominal liabilities from complete explicit discount factors.

    A scenario may assume a future 2033 pricing curve. A dated_market_quote
    requires its valuation date and publication date to be no later than as_of.
    An arithmetic price is not proof of executable cash-flow matching assets.
    """
    known, decision = _known(known_at, as_of, source)
    year = _year(valuation_year)
    if kind not in {"scenario_assumption", "dated_market_quote"}:
        raise ValueError("unknown reserve input kind")
    if kind == "dated_market_quote" and not date(year, 1, 1) <= known <= decision:
        raise ValueError("future reserve yields cannot be represented as observed market quotes")
    schedule = LAURA_LIABILITIES if liabilities is None else liabilities
    if not isinstance(schedule, Mapping) or not schedule or not isinstance(discount_factors, Mapping):
        raise ValueError("nonempty liability and discount mappings required")
    if set(discount_factors) != set(schedule):
        raise ValueError("discount factors must match every liability year exactly")
    cost = Fraction()
    for payment_year, amount in schedule.items():
        if _year(payment_year) < year:
            raise ValueError("reserve cannot include already-past liabilities")
        payment = _nonnegative(amount, "liability")
        factor = number(discount_factors[payment_year], "discount_factor")
        if factor <= 0 or (payment_year == year and factor != 1):
            raise ValueError("discount factors must be positive; immediate factor must equal one")
        cost += payment * factor
    return cost


def project_contributions(annual_returns: Mapping[int, Any], *, target_year: int = 2033,
                          contributions: Mapping[int, Any] | None = None) -> Fraction:
    """Beginning-year deposits earn that year's return; all deposits precede target.

    Exact year-key matching rejects accidentally supplied future returns or
    silent missing years. Returns are assumptions or separately dated inputs;
    this arithmetic function itself does not label them as observed.
    """
    target = _year(target_year)
    schedule = LAURA_CONTRIBUTIONS if contributions is None else contributions
    if not isinstance(schedule, Mapping) or not schedule or not isinstance(annual_returns, Mapping):
        raise ValueError("nonempty contributions and explicit annual returns required")
    parsed = {_year(year): _nonnegative(amount, "contribution") for year, amount in schedule.items()}
    if max(parsed) >= target:
        raise ValueError("all contributions must precede the pre-payment target year")
    required = set(range(min(parsed), target))
    if set(annual_returns) != required or any(type(year) is not int for year in annual_returns):
        raise ValueError("return years must exactly match contribution-to-target years")
    returns = {year: number(value, "annual return") for year, value in annual_returns.items()}
    if any(value < -1 for value in returns.values()):
        raise ValueError("simple return cannot be below total loss")
    wealth = Fraction()
    for year in range(min(parsed), target):
        wealth = (wealth + parsed.get(year, Fraction())) * (1 + returns[year])
    return wealth


def required_reserve_by_path(payments: tuple[Any, ...], between_payment_returns: tuple[Any, ...]) -> Fraction:
    """Backward reserve recursion; K payments require exactly K-1 return assumptions."""
    if not payments or len(between_payment_returns) != len(payments) - 1:
        raise ValueError("K payments require exactly K-1 between-payment returns")
    amounts = tuple(_nonnegative(value, "payment") for value in payments)
    rates = tuple(number(value, "between-payment return") for value in between_payment_returns)
    if any(value <= -1 for value in rates):
        raise ValueError("reserve growth factors must be strictly positive")
    reserve = amounts[-1]
    for index in range(len(amounts) - 2, -1, -1):
        reserve = amounts[index] + reserve / (1 + rates[index])
    return reserve


@dataclass(frozen=True)
class FlexibilityRule:
    """Keep an explicit fraction of residual plus a floor, capped by real surplus.

    The rule is a team/client policy choice, not a cash-flow fact supplied by the
    case. The floor is not an additional required contingency endowment.
    """

    residual_fraction: Any
    minimum_cash: Any = 0

    def __post_init__(self) -> None:
        fraction = number(self.residual_fraction, "residual_fraction")
        floor = _nonnegative(self.minimum_cash, "minimum_cash")
        if not 0 <= fraction <= 1:
            raise ValueError("residual_fraction must be between zero and one")
        object.__setattr__(self, "residual_fraction", fraction)
        object.__setattr__(self, "minimum_cash", floor)


@dataclass(frozen=True)
class GoalAllocation:
    wealth: Fraction
    required_implementation_costs: Fraction
    allocated_implementation_costs: Fraction
    implementation_cost_shortfall: Fraction
    wealth_after_implementation_costs: Fraction
    required_operating_reserve: Fraction
    allocated_operating_reserve: Fraction
    operating_shortfall: Fraction
    signed_surplus_before_flexibility: Fraction
    residual_after_operating: Fraction
    requested_flexibility: Fraction
    retained_flexibility: Fraction
    flexibility_shortfall: Fraction
    facility_capacity: Fraction

    def to_record(self) -> dict[str, dict[str, str]]:
        return {name: money_record(getattr(self, name)) for name in self.__dataclass_fields__}


def allocate_residual(wealth: Any, reserve: Any, *, flexibility: FlexibilityRule,
                      implementation_costs: Any = 0) -> GoalAllocation:
    """Fund essential implementation costs, operating reserve, then residual goals.

    Wealth = allocated costs + actual operating allocation + retained flexibility
    + facility capacity exactly. Required-but-unfunded costs and reserve are
    separate shortfalls; costs cannot silently leave a reserve labeled funded.
    Reserve includes payment one; do not subtract that payment again here.
    implementation_costs are incremental essential costs at this decision, not
    expenses already deducted in a supplied net-return path or reserve price.
    """
    available = _nonnegative(wealth, "wealth")
    required = _nonnegative(reserve, "operating reserve")
    costs = _nonnegative(implementation_costs, "implementation_costs")
    if not isinstance(flexibility, FlexibilityRule):
        raise ValueError("an explicit FlexibilityRule is required")
    paid_costs = min(available, costs)
    net_wealth = available - paid_costs
    allocated = min(net_wealth, required)
    residual = net_wealth - allocated
    request = max(flexibility.minimum_cash, flexibility.residual_fraction * residual)
    retained = min(residual, request)
    return GoalAllocation(available, costs, paid_costs, costs - paid_costs, net_wealth,
                          required, allocated, required - allocated,
                          available - costs - required, residual, request, retained,
                          request - retained, residual - retained)


def bounded_facility_commitment(terminal_surplus: Any, secured_surplus_floor: Any,
                                communicated_cap: Any, residual_fraction: Any = ".2") -> dict[str, Any]:
    """A conditional gift floor/cap rule, with reserve and costs already removed.

    The caller attests that the secured surplus floor is protected independently
    of operating liabilities and implementation costs. This function does not
    authenticate that protection. Negative terminal surplus remains a deficit.
    Realized gift is capped at the communicated amount; all upside beyond the
    cap is retained. No scenario probabilities or asset returns are generated.
    """
    surplus = number(terminal_surplus, "terminal_surplus")
    secured = _nonnegative(secured_surplus_floor, "secured_surplus_floor")
    cap = _nonnegative(communicated_cap, "communicated_cap")
    retained_fraction = number(residual_fraction, "residual_fraction")
    if not 0 <= retained_fraction <= 1:
        raise ValueError("residual_fraction must be between zero and one")
    gift_fraction = 1 - retained_fraction
    promised_floor = gift_fraction * secured
    if cap < promised_floor:
        raise ValueError("communicated cap cannot be below the conditional promised floor")
    positive_surplus = max(surplus, Fraction())
    gift = min(gift_fraction * positive_surplus, cap)
    floor_breached = surplus < secured
    return {
        "status": "SECURED_SURPLUS_FLOOR_BREACHED" if floor_breached else "WITHIN_ATTESTED_FLOOR_AND_CAP",
        "terminal_surplus_after_reserve_and_costs": money_record(surplus),
        "surplus_deficit": money_record(max(-surplus, Fraction())),
        "secured_surplus_floor": money_record(secured),
        "promised_facility_floor": money_record(promised_floor),
        "communicated_facility_cap": money_record(cap),
        "realized_capped_gift": money_record(gift),
        "remaining_flexibility": money_record(positive_surplus - gift),
        "secured_surplus_floor_breached": floor_breached,
        "promised_facility_floor_breached": gift < promised_floor,
        "floor_basis": "Caller-attested protected surplus after operating reserve and implementation costs; unauthenticated",
        "funding_probability": None,
        "is_unconditional_guarantee": False,
    }


def strategic_sleeves(wealth: Any, required_reserve: Any, *,
                      surplus_equity_fraction: Any = ".60", overall_equity_cap: Any = ".40",
                      implementation_costs: Any = 0, secured_surplus_floor: Any = 0) -> dict[str, Any]:
    """Proposed pre-2033 target dollars after protecting the operating reserve.

    Essential incremental reserve implementation costs come first. Equity =
    min(60% of positive post-reserve residual, 40% of post-cost wealth) under the
    default team proposal. Remaining residual is short USD liquidity. The reserve
    is accounted separately and never borrowed to reach the equity target. This is
    an allocation rule conditional on an independently defensible reserve price,
    not an executable order, cash-flow match certification or model prediction.
    A communicated facility floor may encumber part of surplus liquidity from
    2031 onward; rebalancing cannot move that caller-attested amount into equity.
    """
    available = _nonnegative(wealth, "wealth")
    required = _nonnegative(required_reserve, "required_reserve")
    costs = _nonnegative(implementation_costs, "implementation_costs")
    floor = _nonnegative(secured_surplus_floor, "secured_surplus_floor")
    risk_share, cap = number(surplus_equity_fraction), number(overall_equity_cap)
    if not 0 <= risk_share <= 1 or not 0 <= cap <= 1:
        raise ValueError("strategic sleeve fractions must be between zero and one")
    paid_costs = min(available, costs)
    net_wealth = available - paid_costs
    allocated = min(net_wealth, required)
    surplus = net_wealth - allocated
    protected_surplus = min(surplus, floor)
    equity = min(risk_share * surplus, cap * net_wealth, surplus - protected_surplus)
    return {
        "policy_status": "EXPLICIT_TEAM_PROPOSAL_CONDITIONAL_ON_RESERVE_PRICING",
        "wealth": money_record(available), "required_reserve": money_record(required),
        "required_implementation_costs": money_record(costs),
        "allocated_implementation_costs": money_record(paid_costs),
        "implementation_cost_shortfall": money_record(costs - paid_costs),
        "wealth_after_implementation_costs": money_record(net_wealth),
        "operating_reserve_allocation": money_record(allocated),
        "operating_shortfall": money_record(required - allocated),
        "equity_target": money_record(equity), "short_usd_liquidity_target": money_record(surplus - equity),
        "required_secured_surplus_floor": money_record(floor),
        "allocated_secured_surplus_floor": money_record(protected_surplus),
        "secured_surplus_floor_shortfall": money_record(floor - protected_surplus),
        "unencumbered_liquidity_target": money_record(surplus - equity - protected_surplus),
        "secured_floor_basis": "Caller-attested net protected surplus; unauthenticated; included in liquidity target, not additional wealth",
        "surplus_equity_fraction": str(risk_share), "overall_equity_cap": str(cap),
        "equity_cap_basis": "post-cost net wealth",
        "uses_wins_balance_or_pnl": False,
    }


def planning_scenario_grid(*, flexibility: FlexibilityRule,
                           growth_rates: tuple[Any, ...] = ("-.05", "0", ".02", ".04", ".06", ".08"),
                           reserve_rates: tuple[Any, ...] = ("0", ".02", ".04"),
                           known_at: str | date, as_of: str | date, source: str,
                           implementation_costs: Any = 0) -> dict[str, Any]:
    """Full, unselected cross-product of net growth and reserve-rate sensitivities."""
    known, decision = _known(known_at, as_of, source)
    costs = _nonnegative(implementation_costs, "implementation_costs")
    growth = tuple(number(rate, "growth assumption") for rate in growth_rates)
    discount = tuple(number(rate, "reserve assumption") for rate in reserve_rates)
    if (not growth or not discount or len(set(growth)) != len(growth)
            or len(set(discount)) != len(discount)):
        raise ValueError("nonempty distinct growth and reserve assumptions required")
    rows = []
    for rate in growth:
        wealth = project_contributions({year: rate for year in range(2027, 2033)})
        for reserve_rate in discount:
            reserve = annuity_due_reserve(reserve_rate)
            allocation = allocate_residual(wealth, reserve, flexibility=flexibility, implementation_costs=costs)
            rows.append({"annual_net_growth_assumption": str(rate),
                         "annual_reserve_discount_assumption": str(reserve_rate),
                         **allocation.to_record()})
    return {"status": "HYPOTHETICAL_SENSITIVITY_GRID_NOT_FORECAST", "as_of": decision.isoformat(),
            "assumptions_known_at": known.isoformat(), "source": source, "rows": rows,
            "funding_probability": None, "uses_wins_balance_or_pnl": False,
            "reserve_rates_are_locked_yields": False,
            "implementation_costs": money_record(costs),
            "implementation_cost_assumption": ("Explicit zero-cost planning assumption; not a fee quote"
                                               if costs == 0 else
                                               "Explicit incremental implementation-cost assumption; not a fee quote"),
            "target": "2033-01-01_before_first_payment"}


def required_constant_growth(target_wealth: Any, *, lower_rate: Any = "-1", upper_rate: Any = "1",
                             tolerance: Any = "0.00000001", max_iterations: int = 128) -> dict[str, Any]:
    """Bracket the constant annual net return needed by the exact contribution path.

    This solves an arithmetic requirement, not an expected return. Returned
    lower/upper bounds enclose the monotone root when status is BRACKETED.
    """
    target = _nonnegative(target_wealth, "target_wealth")
    lower, upper = number(lower_rate), number(upper_rate)
    epsilon = number(tolerance)
    if lower < -1 or lower >= upper or epsilon <= 0 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("invalid root bracket or precision")

    def value(rate):
        return project_contributions({year: rate for year in range(2027, 2033)})

    if not value(lower) <= target <= value(upper):
        return {"status": "TARGET_OUTSIDE_RETURN_BRACKET", "target_wealth": money_record(target),
                "lower_rate": str(lower), "upper_rate": str(upper)}
    for _ in range(max_iterations):
        if value(lower) == target:
            upper = lower
            break
        if value(upper) == target:
            lower = upper
            break
        if upper - lower <= epsilon:
            break
        midpoint = (lower + upper) / 2
        if value(midpoint) <= target:
            lower = midpoint
        else:
            upper = midpoint
    return {"status": "BRACKETED" if upper - lower <= epsilon else "PRECISION_NOT_REACHED",
            "target_wealth": money_record(target), "lower_rate": str(lower), "upper_rate": str(upper),
            "lower_wealth": money_record(value(lower)), "upper_wealth": money_record(value(upper)),
            "interpretation": "Required constant annual net return; not a forecast or probability"}


def goal_summary(wealth_by_scenario: Mapping[str, Any], reserve_by_scenario: Mapping[str, Any], *,
                 flexibility: FlexibilityRule, implementation_costs: Any = 0) -> dict[str, Any]:
    """Model-policy bridge for aligned terminal wealth/reserve scenarios.

    A model must separately justify how its forecasts become multiyear scenarios.
    This function neither extrapolates an S6 one-period forecast nor invents
    probabilities. Scenario-specific reserve prices preserve joint rate risk.
    implementation_costs may be one nonnegative amount or an identically keyed
    scenario mapping, for incremental essential costs not already in net wealth.
    """
    if (not isinstance(wealth_by_scenario, Mapping) or not wealth_by_scenario
            or not isinstance(reserve_by_scenario, Mapping)
            or set(wealth_by_scenario) != set(reserve_by_scenario)
            or any(not isinstance(name, str) or not name.strip() for name in wealth_by_scenario)):
        raise ValueError("wealth/reserve scenarios must have the same nonempty named grid")
    if isinstance(implementation_costs, Mapping):
        if set(implementation_costs) != set(wealth_by_scenario):
            raise ValueError("implementation costs must match the scenario grid exactly")
        costs = {name: _nonnegative(value, "implementation_costs")
                 for name, value in implementation_costs.items()}
    else:
        shared = _nonnegative(implementation_costs, "implementation_costs")
        costs = {name: shared for name in wealth_by_scenario}
    rows = {name: allocate_residual(wealth, reserve_by_scenario[name], flexibility=flexibility,
                                    implementation_costs=costs[name])
            for name, wealth in wealth_by_scenario.items()}
    capacities = [row.facility_capacity for row in rows.values()]
    return {
        "status": "CONDITIONAL_SCENARIO_ARITHMETIC",
        "scenarios": {name: row.to_record() for name, row in rows.items()},
        "facility_capacity_range": [money_record(min(capacities)), money_record(max(capacities))],
        "worst_implementation_cost_shortfall": money_record(max(row.implementation_cost_shortfall
                                                                  for row in rows.values())),
        "worst_operating_shortfall": money_record(max(row.operating_shortfall for row in rows.values())),
        "worst_flexibility_shortfall": money_record(max(row.flexibility_shortfall for row in rows.values())),
        "all_listed_scenarios_fund_operating_reserve": all(row.operating_shortfall == 0 for row in rows.values()),
        "all_listed_scenarios_cover_costs_and_operating_reserve": all(
            row.implementation_cost_shortfall == 0 and row.operating_shortfall == 0 for row in rows.values()),
        "implementation_cost_assumption": ("Explicit zero-cost planning assumption; not a fee quote"
                                           if all(value == 0 for value in costs.values()) else
                                           "Explicit incremental implementation-cost assumptions; not fee quotes"),
        "funding_probability": None,
        "range_is_calibrated_confidence_interval": False,
        "range_scope": "Extrema across the supplied named scenarios only; not an unconditional pledge",
    }


def _project_2031(balance: Any, annual_return_scenarios: Mapping[str, Mapping[int, Any]]) -> dict[str, Fraction]:
    starting = _nonnegative(balance, "beginning-2031 balance")
    if not isinstance(annual_return_scenarios, Mapping) or not annual_return_scenarios:
        raise ValueError("named 2031/2032 annual-return scenarios required")
    values = {}
    for name, path in annual_return_scenarios.items():
        if (not isinstance(path, Mapping) or set(path) != {2031, 2032}
                or any(type(year) is not int for year in path)):
            raise ValueError("each conditional path must contain exactly 2031 and 2032")
        rates = [number(path[year], f"{year} return") for year in (2031, 2032)]
        if any(rate < -1 for rate in rates):
            raise ValueError("return below total loss")
        values[name] = starting * (1 + rates[0]) * (1 + rates[1])
    return values


def conditional_2031_range(balance_2031: Any,
                           annual_return_scenarios: Mapping[str, Mapping[int, Any]],
                           reserve_by_scenario: Mapping[str, Any], *, flexibility: FlexibilityRule,
                           balance_observed_at: str | date, balance_available_at: str | date,
                           assumptions_known_at: str | date, as_of: str | date, source: str,
                           implementation_costs: Any = 0) -> dict[str, Any]:
    """Condition on an actually observed beginning-2031 balance, never future NAV.

    All scenario inputs, including reserve assumptions, must be declared by
    assumptions_known_at. These dates are attestations, not source authentication.
    """
    _, decision = _known(assumptions_known_at, as_of, source)
    observed, released = _date(balance_observed_at), _date(balance_available_at)
    if observed != date(2031, 1, 1) or not observed <= released <= decision or decision.year != 2031:
        raise ValueError("require an available beginning-2031 balance at a 2031 decision")
    result = goal_summary(_project_2031(balance_2031, annual_return_scenarios), reserve_by_scenario,
                          flexibility=flexibility, implementation_costs=implementation_costs)
    return {**result, "balance_kind": "observed_beginning_2031", "as_of": decision.isoformat(),
            "balance_observed_at": observed.isoformat(), "balance_available_at": released.isoformat(),
            "assumptions_known_at": _date(assumptions_known_at).isoformat(), "source": source,
            "starting_balance": money_record(balance_2031), "target": "2033-01-01_before_first_payment"}


def illustrative_2031_range(hypothetical_balance_2031: Any,
                            annual_return_scenarios: Mapping[str, Mapping[int, Any]],
                            reserve_by_scenario: Mapping[str, Any], *, flexibility: FlexibilityRule,
                            assumptions_known_at: str | date, as_of: str | date, source: str,
                            implementation_costs: Any = 0) -> dict[str, Any]:
    """A planning sensitivity usable in 2026, explicitly not an observed 2031 update."""
    _, decision = _known(assumptions_known_at, as_of, source)
    if decision >= date(2031, 1, 1):
        raise ValueError("future-balance sensitivity requires an as_of date before 2031-01-01")
    result = goal_summary(_project_2031(hypothetical_balance_2031, annual_return_scenarios),
                          reserve_by_scenario, flexibility=flexibility, implementation_costs=implementation_costs)
    return {**result, "balance_kind": "hypothetical_future_balance", "as_of": decision.isoformat(),
            "assumptions_known_at": _date(assumptions_known_at).isoformat(), "source": source,
            "starting_balance": money_record(hypothetical_balance_2031),
            "target": "2033-01-01_before_first_payment"}
