# Laura planning v1

An executable, standard-library portfolio-planning layer for Laura's supplied competition case. It implements exact cash-flow and allocation arithmetic independently of an ML checkpoint. All returned amounts retain a rational value alongside a half-even, cents-rounded display. No model prediction, scenario probability, bond quote, eligibility permission or actual trade is manufactured.

The case contributes $300,000 at the beginning of 2027 and $150,000 at the beginning of 2028. The ten $50,000 operating payments begin in 2033 and end in 2042. The first payment is already included in the 2033 reserve. WInS gains and balances are not inputs to this long-term contribution path.

```sh
rtk proxy .venv/bin/python -m successors.laura_planning_v1.check --output /tmp/laura-new-integrity
rtk proxy .venv/bin/python -m successors.laura_planning_v1.run --output /tmp/laura-new-arithmetic
```

Each output directory must be new. The integrity command uses a separate process, a 60-second parent timeout and sampled 128 MiB peak-memory ceiling. The arithmetic command reproduces all 18 declared growth/reserve cells, twelve dated reserve-price sensitivities, three required-return brackets, nine cost/flexibility combinations, and the conditional 2031 communication example. These are mathematical calculations from assumptions, not backtests or a fitted return distribution.

`core.py` exposes:

- `project_contributions` for explicit annual simple-return paths, preserving complete loss and the timing of each deposit.
- `annuity_due_reserve`, `discounted_reserve` and `required_reserve_by_path` for immediate and subsequent payment timing. A complete loss between future payments is infeasible under the backward positive-growth recursion and raises an error; callers must retain this infeasible stress, not remove it from probability calculations.
- `allocate_residual` and `strategic_sleeves` for incremental implementation costs, the operating allocation, shortfalls, and discretionary uses. The equity cap uses wealth after costs. Once a facility minimum is communicated, `secured_surplus_floor` additionally prevents rebalancing its supporting assets into equity and exposes any shortfall. Arithmetic cannot authenticate the supplied reserve price or its asset coverage.
- `goal_summary` for aligned, scenario-specific wealth, reserve and cost inputs. `conditional_2031_range` accepts a dated beginning-2031 balance available at a 2031 decision; it is not an arbitrary midyear current-NAV API. `illustrative_2031_range` is explicitly hypothetical and refuses dates from 2031 onward.
- `bounded_facility_commitment` for an at-most-80%-of-surplus gift, secured cash-flow floor, communicated cap, and retained upside. Protection is caller-attested, and violations remain visible. The function does not support a separate fixed-dollar flexibility floor.

Keep reserve valuation date, currency, source, cost basis and quote/assumption type with every result. A numerical `Fraction` alone cannot prevent a caller from mixing a 2027 NAV with a 2033 reserve cost. The generator retains these fields. When reserve securities are already owned, assess their cash-flow coverage and the cost of remaining gaps; do not deduct the full reserve twice or spend its mark-to-market appreciation.

For S6 integration, `goal_summary` consumes aligned terminal wealth/reserve/cost scenarios after a separately justified multiyear conversion. It does not extrapolate a one-period Student-t forecast to 2042 or certify financial causality. The S6 model and this planning code are distinct, additive components of the accepted dual prediction/planning concept. No S6 market-trained checkpoint exists in this delivery.

See `MATHEMATICS.md` for conditional proofs and counterexamples, and `research/laura_plan_20261013/INVESTMENT_POLICY_PROPOSAL.md` for the concrete policy. Existing frozen S6 and predecessor files remain unchanged.
