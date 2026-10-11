# Portfolio, risk and funding specification

## Information boundary

The model forecasts and the portfolio optimizer are separate components. The optimizer consumes an aligned expected-return vector, covariance, optional scenario matrix, a dated instrument master, current holdings, NAV and a reviewed mandate. No function fetches data, infers the client's actual portfolio, or submits a trade.

All returns at this interface are **simple returns in one base currency over one common executable horizon**. Model output in log-return space must be converted explicitly. For a jointly Gaussian log-return vector, `gaussian_log_to_simple_moments` applies the lognormal moment formulas. A Student-t log-return distribution has no finite positive exponential moment: exponentiating its location and calling that an expected arithmetic return is invalid. Use an explicitly bounded, calibrated empirical forecast distribution or an independently justified finite-moment model instead.

The metadata gate checks exact IDs, dated universe membership, approvals, instrument exposure, listing/delisting status, availability times, metadata staleness, base-currency prices and ADV. Unknown eligibility is not inferred from a familiar ticker. Country, sector and FX exposure maps support fractional look-through treatment for approved funds.

## Cost-aware self-financing accounting

Let `w` denote post-cost risky weights, with residual cash `c=1-sum(w)`. Let `w0` denote current positions as fractions of pre-trade NAV `V`. The retained fraction of NAV, `alpha`, solves

\[
\alpha +\sum_i k_i^+\max(\alpha w_i-w_{0i},0)
 +\sum_i k_i^-\max(w_{0i}-\alpha w_i,0)=1.
\]

The signed hypothetical trade value is `V*(alpha*w-w0)`, fees are `(1-alpha)*V`, and cash after trading is `alpha*c*V`. Thus a nominal 20% cash target does not automatically preserve a $20 cash requirement in a $100 account when fees have consumed capital. The complete check uses dollar cash after estimated costs.

The scalar equation is bracketed on `[0,1]`. Finite nonnegative per-side costs below 1,000 basis points are required; rates are assumptions unless a dated schedule is supplied. Buy and sell costs can be set independently by instrument. The default 10 basis points is an illustrative research setting, not a broker quote or verified competition cost.

## Objective and hard restrictions

The normal policy minimizes

\[
-\mathbb E[R_{net}(w)]
+\lambda\alpha^2w^\top\Sigma w
+\eta\operatorname{CVaR}_{\beta}(-R_{net}(w))
+\tau\sum_i|\alpha w_i-w_{0i}|.
\]

Scenario returns include residual cash and the same immediate trading-cost deduction. CVaR is disabled by default until an aligned scenario matrix is supplied; setting a positive penalty without scenarios blocks the request. A finite weighted distribution uses the exact worst tail mass, including a fractional atom at the VaR boundary. The scenario-loss treatment is based on [Rockafellar and Uryasev's CVaR formulation](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf).

Hard restrictions include long-only unlevered holdings, minimum/maximum cash, exact eligible-universe membership, maximum individual weights, market/sector/asset-class floors and caps, economic FX exposure, turnover, absolute reserve cash, trading participation in ADV, position size relative to ADV, and optional foreign buy headroom. An untradable held instrument is frozen in units/value at the decision point. If it is also ineligible to remain held, the request can be infeasible; the program cannot resolve a real trading halt by pretending to sell it.

Turnover is **gross risky dollars traded divided by pre-trade NAV**, not half the sum of risky weight changes. Selling a 100% holding and buying another 100% holding is approximately 200% gross turnover. Market caps operate on post-cost portfolio weights. Per-trade participation limits compare dollar changes with the instrument's synchronized ADV.

## Solver and fallback behavior

The implementation uses [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) with multiple deterministic starting candidates. It records every solver message, iteration count and constraint check. A solver success flag alone never certifies the allocation: the returned candidate must pass the independent common constraint checker.

If the main solve fails, the minimum-variance/risk-only policy uses the same constraints. Cash and unchanged holdings are then checked as deterministic fallback candidates. Cash is not automatically feasible: maximum cash, minimum asset-class weights, liquidation limits, non-tradability and costs can forbid it. If no candidate passes, output is `infeasible` with no weights. A numerical failure does not prove that the mathematical feasible set is empty; the diagnostic means that this implementation did not certify a feasible candidate.

Returned statuses are `optimized`, `fallback_minimum_variance`, `fallback_cash`, `fallback_hold`, `blocked`, `no_action`, and `infeasible`. The latter three contain no target weights. Local numerical solutions are not claimed to be globally optimal, especially with nonsmooth costs and tail-risk terms.

`round_allocation_to_lots` floors target quantities to supported lots, retains non-tradable holdings, recomputes costs and cash, and applies the same checker. Rounding can violate a lower exposure bound even when the continuous solution passes. Such a proposal is rejected with quantities withheld; no restriction is silently loosened.

## Recession and forecast failure

`choose_regime` is a transparent policy gate. It is not a recession predictor. If no recession probability is supplied, none is inferred. Dated supplied probabilities, observed drawdown and volatility may trigger `defensive` or `cash` preferences. Failed data-quality or calibration gates cause `no_action`. Missing forecasts trigger a constrained risk-only fallback, rather than a fabricated expected return.

Default thresholds in `RegimePolicy` are unvalidated research settings and must be fixed before a protected comparison. Defensive mode raises risk aversion and adds a soft cash preference. Cash mode maximizes permitted cash subject to every hard mandate restriction. Neither mode bypasses reserve obligations or the approved universe. A predicted recession can fail to occur, and a recession need not imply all asset prices decline simultaneously.

## Stochastic and stress calculations

`estimate_covariance` shrinks a synchronized sample covariance toward its diagonal and applies a recorded eigenvalue floor. Missing values are rejected unless complete-row deletion is explicitly requested. The function uses the supplied return horizon without silently annualizing overlapping observations.

`stochastic_return_scenarios` implements GBM-style log increments with an optional common-scale Student-t shock and optional shared Poisson jumps. A declared log-return cap is recorded along with the clipping count. The heavy-tail branch is a finite-step, bounded stress approximation, not an exact continuous-time diffusion or a theorem of market efficiency. It has no claim to calibrated probabilities or no-arbitrage pricing. The arithmetic mean and covariance of its generated returns need not equal the input drift/covariance after exponentiation, clipping or jumps. It runs only when explicitly called in a later authorized experiment.

`StressScenario` provides scenario-dependent equity, market, sector, FX and bond shocks. Named examples to predefine in an experiment include recession/disinflation, inflation with rate increases, spread widening, currency shock, a Taiwan supply-chain disruption, and an upside/no-recession outcome. No such scenario was generated or assigned a probability in this release.

For a bond, the local approximation is

\[
R_{bond}\approx carry-D\Delta y-D_s\Delta s
+\tfrac12 C(\Delta y+\Delta s)^2.
\]

This requires duration, convexity, spread duration and carry over a consistent horizon. Large shocks, default, embedded options and changing duration may require explicit repricing. Base-currency return is composed multiplicatively: `(1+local_return)*(1+fx_return)-1`. A missing non-base-currency FX assumption raises an error.

## Funding is a separate decision problem

The Laura contribution/projection and annuity-due functions are implemented in `mandate.py`. They accept explicit assumptions and preserve operating shortfalls. The 2031 facility range is conditional on the observed 2031 portfolio, not on today's favorable simulation path. Cash-flow matching and scenario reserve analysis support different questions; a present-value formula alone does not establish high-certainty funding.

## Public interfaces

```python
Mandate.from_json(path)
Mandate.validate(as_of, require_allocation=True)

optimize_portfolio(
    expected_returns, covariance, instruments, mandate,
    as_of=decision_time, nav=current_nav, current_weights=current_weights,
    scenarios=scenario_returns, scenario_probabilities=probabilities,
    cash_return=cash_horizon_return, reserve_cash=required_cash,
    decision_context=RegimeInputs(...), config=PortfolioConfig(...),
)

validate_allocation(weights, instruments, mandate, as_of=decision_time,
                    nav=current_nav, current_weights=current_weights)

conditional_2031_analysis(balance_2031, return_paths_2031_2032,
                          reserve_amount, flexibility_buffer,
                          interval_mass=0.90, probabilities=probabilities)
```

All arrays align with the instrument sequence. Pass an explicit zero current-weight vector for an initial all-cash portfolio. Missing current weights are blocked. Probability vectors must already sum to one; the code does not silently repair an arbitrary weighting error.

## Verification boundary

`tests/test_portfolio.py` contains tests for reserve timing, FX, CVaR tail atoms, covariance rejection, missing mandates, future metadata, costs, fractional fund exposures, false string approvals, rounding and feasible/infeasible fallbacks. These tests now pass in the retained 163-test FREM engineering suite. That runtime compatibility evidence is not market calibration, portfolio efficacy, or mandate certification.
