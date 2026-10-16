# Checked mathematical contracts for the restart

Date: 2026-10-15. Scope: accounting and probabilistic identities, not a new forecasting model or a proof of investment advantage. Reference implementation: `successors/frem_audit_v1/contracts.py`; adversarial examples: `tests/restart/test_restart_audit.py`.

## 1. Horizon and availability

Let an observed total simple return over a declared interval be r_j >= -1. For continuous, nonoverlapping intervals exactly covering [a,b], the compounded return is

R[a,b] = product_j(1 + r_j) - 1.

Proof: each interval multiplies the preceding wealth by 1+r_j, so induction gives the product. Adding simple returns is incorrect: +10% then -10% gives -1%, not zero. A terminal -100% loss remains a total loss. The implementation rejects gaps, overlaps, incomplete coverage and returns whose recorded availability is at or after the decision. Strict earlier-than availability is the chosen convention; it does not assert that historical source timestamps exist where they do not.

For log returns X_j with finite second moments and a stationary covariance sequence gamma_k, expansion of Var(sum X_j) gives

Var(sum_{j=1}^H X_j) = H gamma_0 + 2 sum_{k=1}^{H-1}(H-k) gamma_k.

There are H-k pairs at lag k; the factor two counts both covariance directions. Square-root time scaling requires the cross terms to vanish and the relevant marginal variances to agree. The function checks a finite complete lag list, elementary covariance bounds and nonnegative resulting variance. These checks do not establish stationarity or positive semidefiniteness of the full implied covariance matrix. A new estimator must establish those conditions. Variable calendar horizons must be represented explicitly, not rounded to five sessions.

## 2. Density score is not expected portfolio return

If L is a log return, expected simple return is E[exp(L)]-1, when this expectation exists. It is generally not exp(E[L])-1. For equally likely simple returns -50% and +50%, the mean simple return is zero but exp(E[L])-1 = sqrt(0.75)-1, approximately -13.40%.

A nondegenerate unbounded Student-t log-return distribution with finite degrees of freedom has density asymptotic to C x^(-nu-1) in its positive tail, with C>0. Hence its exponential moment contains an integral bounded below by a positive constant times integral_A^infinity exp(x) x^(-nu-1) dx, which diverges. Finite log-return variance when nu>2 does not repair this. A positive-weight mixture containing such a component also has an infinite exponential moment. The degenerate zero-scale case is excluded from this statement.

Therefore a Student-t density forecaster may remain useful for log score, but it cannot supply a finite expected simple return by either naive exponentiation method. S8 explicitly used a log-moment ranking surrogate; this audit does not retroactively invalidate that declared surrogate or reopen its result. A future economic model requires a separately specified finite-moment return model or finite scenarios, with its tail assumptions and calibration evaluated.

## 3. Joint scenarios and portfolio risk

For finite joint simple-return scenarios r_s, nonnegative probabilities p_s summing to one, and each asset return at least -1, define

mu = sum_s p_s r_s,

Sigma = sum_s p_s (r_s-mu)(r_s-mu)^T.

For any vector w, w^T Sigma w = sum_s p_s [w^T(r_s-mu)]^2 >= 0. Thus the exact covariance is positive semidefinite. The implementation uses double precision, checks finite results and allows 1e-12 probability normalization tolerance; mathematical equalities are ideal-arithmetic statements. Joint rows preserve cross-asset dependence, whereas independent marginal scenarios would change it. An equal mix of perfectly offsetting assets has zero scenario variance even though both individual variances are positive.

This proves correct moments for the supplied distribution. It does not prove the scenarios describe future markets, their probabilities are calibrated, or their finite tails safely represent actual tail risk. No market scenario distribution was fitted in this restart.

## 4. Exact proportional-cost accounting

Let risky dollar holdings be h_i>=0, cash c_0>=0, NAV V=c_0+sum h_i, proportional fee c in [0,1), and target post-fee risky weights w_i>=0 with sum w_i<=1. Post-fee NAV q solves

F(q) = q + c sum_i |q w_i-h_i| - V = 0.

F(0)=c sum h_i-V<=0 and F(V)>=0. For q_2>q_1, the reverse triangle inequality gives

F(q_2)-F(q_1) >= (q_2-q_1)(1-c sum w_i)>0.

Continuity and strict monotonicity give a unique root in [0,V]. The zero-NAV case has root zero. A bisection bracket shrinks by 2^-100 after 100 steps in exact arithmetic; the implementation reaches the limits of double precision earlier and is checked against numerical tolerances. Target holdings are q w_i, fees are c sum |q w_i-h_i|, and remaining cash is q(1-sum w_i). The reference accepts only tiny numerical weight/cash tolerances and is not a target-selection algorithm.

For full entry from cash and full liquidation after simple asset return r, terminal wealth divided by initial wealth is (1+r)(1-c)/(1+c). Breaking even against zero-yield cash requires r >= 2c/(1-c). At ten basis points per side this is approximately 0.2002%, before any cash opportunity cost. Different fee, spread, tax or financing conventions require a different equation.

## 5. Saved-ledger attribution

For each interval, post-trade risky holding is (NAV_start-fee)*weight. Realized asset contribution is ending holding minus that post-trade holding. Therefore

NAV_end - NAV_start = sum asset contributions - trade fee - terminal liquidation fee.

Summing over a continuous path telescopes to terminal wealth minus initial wealth. The auditor checks continuity, fee arithmetic, long-only cash accounting, contribution totals, ending NAV, drawdown, explicit asset order and the published summary. It handles the two retained liquidation schemas only in its audit view. When an old ledger lacks weights, it reconstructs them from saved contributions and holdings; that is an internal consistency check, not independent verification against raw market prices. Origin-year totals carry wealth across years and do not restart with $100,000 annually.

## 6. What cannot be proved

No admissible forecast-free long-only policy can guarantee both beating cash and beating risky benchmarks on every possible return path. Any positive risky allocation loses to zero-yield cash when all risky assets lose 100%; all-cash allocation loses to a rising risky benchmark. Correct accounting, positive semidefinite risk and test coverage cannot remove this counterexample. Performance claims therefore require a prespecified empirical test with uncertainty and a terminal negative/inconclusive option.

Forty bounded tests exercise these contracts and saved-ledger readers. They contain constructed counterexamples solely to check mathematics and rejection behavior; they are not fabricated market observations or experimental evidence of alpha.
