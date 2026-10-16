# FREM S10 model specification

Design revision 1, 15 October 2026. Status: specified proposal for review, not implemented or scientifically frozen. The parameters below are engineering choices made before S10 outcomes, not values established optimal by predecessor results.

## 1 Question and contribution

Existing FREM experiments show that small density improvements can coexist with poor allocation, and that contextual routing may add little over global mixing. S10 asks: does delayed contextual stacking of direct joint holding-period scenarios improve distribution quality over global stacking, and does the same distribution support better net decisions under exact carried-holdings accounting?

The candidate mechanism is local mixture-level estimation with shrinkage to a global mixture, using only stored forecasts whose outcomes have matured. Local similarity controls which past forecasting situations influence today's mixture. It does not retrieve future outcomes, score the current target, or optimize each expert independently.

The empirical contribution would be a controlled diagnosis of distribution mixing and decision translation. Context-dependent stacking and prescriptive optimization have prior art. Neither the mechanism nor a favorable result is assumed novel. A failure against equal/global mixing falsifies the proposed practical contextual increment in the tested scope.

## 2 Domain and target contract

Revision 1 is restricted to three synchronized positive-price ETFs, ordered SPY, TLT, GLD, in USD. Horizon H is four consecutive common-session weekly intervals. An interval begins at the first common session open of an ISO week. Decisions occur at the existing research convention of 12:00 UTC before that open; outcomes become available at midnight UTC after the ending open date. These are assumed availability/fill conventions, not verified historical vendor timestamps or executable broker orders.

For historical origin j, asset a and endpoint t(j+4):

    g[j,a] = open_price[t(j+4),a] / open_price[t(j),a]
    d[j,a] = sum(cash_dividend[a] with t(j) < ex_date <= t(j+4)) / open_price[t(j),a]
    r[j,a] = g[j,a] + d[j,a] - 1

Thus g is a dimensionless price gross return, d is nonnegative dividend cash per starting risky dollar, and r is the total simple return with dividends kept as cash. Do not compound dividend-reinvested weekly returns to construct this target. Price and dividend components remain separate in checkpoints and decisions.

Revision 1 assumes zero cash yield, immediate ex-date cash accrual, no taxes, no split within the admitted source window, no borrowing or shorting, and ideal target-weight open fills. These are restrictive retrospective conventions. Real execution requires payment-date receivables, observed cash rates, tradability and an order-sizing/fill model. Defaulted equities, terminal zero prices, FX and changing historical constituents require a different admitted adapter. Reject unsupported events; never delete a terminal loss and continue.

All record types carry source identity, observation time, availability time, origin, endpoint and units. Features and training labels must have availability strictly less than the relevant decision timestamp. A file hash does not authorize reading a protected value.

## 3 Inputs and fixed transforms

For each asset, use the last 13 fully available weekly total log returns log(g_week+d_week). The four features are the most recent return, sum of four, sum of thirteen, and population standard deviation of thirteen. The input x is asset-major R^12. Missing any required interval invalidates that origin; dates are excluded only by timestamp/availability rules fixed before outcomes. No forward fill or calendar intersection that hides source defects.

Use a single burn-in transform fitted on valid feature/target records with four-interval labels available before 2008-01-01. Require at least 64 records. Freeze feature mean and population scale; replace feature scales below 1e-12 with 1. For scoring, fix the three asset return scales b[a] = max(population SD of burn-in r[:,a], 1e-4). Standardized score vectors are z[a]=r[a]/b[a]. These positive fixed transforms preserve the population proper-score interpretation; they do not certify calibration.

The burn-in transform uses only pre-2008 matured information. Prequential forecast records begin on/after 2008-01-01. No later refitting of the transform is allowed in the main design. This avoids silently rescaling historical scores after evaluation starts, but may become stale; that limitation remains explicit.

Holdings, fees and risk limits are policy inputs, not forecast features. Labels never enter neighbor selection. The calendar date is used for age and maturity, not as a retrospective regime label.

## 4 Scenario library and experts

At weekly origin t, the library consists of the most recent min(520, n_available) four-interval exemplars with outcome availability strictly before t. Require at least 64. Each atom is the paired vector (g_i,d_i) in R^3 x R^3 and its raw context x_i. Scenario weights sum to one and are nonnegative. All experts share exactly the same atoms and information.

Three experts:

| Expert | Atom probabilities |
|---|---|
| U, unconditional | 1/n |
| R, recent | proportional to 2^(-age_i/52), age in seven-day elapsed weeks since origin |
| C, context | uniform on the 32 nearest x_i by squared Euclidean distance in the fixed standardized 12-dimensional space |

Break distance ties by origin timestamp then stable record identity. No distance or outcome clipping is used. The nearest-neighbor expert falls back to U when its 32nd-neighbor distance exceeds the 95th percentile, using linear interpolation, of library leave-one-out 32nd-neighbor distances. With fewer than 33 eligible reference neighbors the fallback applies. This is a deterministic extrapolation flag, not a confidence probability or a guarantee that a nonflagged query is in distribution.

All comparisons retain the fallback flags and the fraction of predictions using them. The output of each expert is an empirical probability law on joint (g,d), not a continuous probability density. Finite support gives finite moments for admitted finite values. It cannot include shocks absent from that support.

## 5 Proper score and mixture optimization

Use the energy score with Euclidean norm and exponent one on the fixed standardized total returns:

    ES(p,y) = sum_i p_i ||z_i-z_y||
              - 0.5 sum_i sum_k p_i p_k ||z_i-z_k||.

Lower is better. Include diagonal and off-diagonal terms exactly; this scores the discrete predictive law, so do not replace its pair term with an unbiased Monte Carlo estimator. The distribution has a well-defined score for observations outside its atoms.

For the three expert distributions issued at past origin j, after y_j matures store:

    A_j[e] = sum_i p_e,j[i] ||z_i-z_yj||
    B_j[e,f] = sum_i sum_k p_e,j[i] p_f,j[k] ||z_i-z_k||.

For mixture alpha on the three-dimensional simplex, its exact score is alpha' A_j - 0.5 alpha' B_j alpha. The cross-expert terms are essential; averaging standalone expert scores would miss mixture complementarity. Store the source forecast hash with A_j and B_j. Do not regenerate old forecasts using the updated library.

At t, retain the last 104 matured prequential records j with origin at/after 2008-01-01. The record consists of x_j, issued expert identities/probabilities, maturity time, A_j and B_j. Weekly labels overlap, so 104 records are not 104 independent observations.

Global mixture:

    alpha_G = argmin_(alpha>=0, sum alpha=1)
              mean_j ES_j(alpha) + 0.05 ||alpha-u||^2,
    u = (1/3,1/3,1/3).

Use u until at least 26 records have matured. The score is dimensionless after the fixed b transform. The fixed regularization coefficient 0.05 has no claimed optimality.

Local mixture:

1. Select the 32 nearest eligible matured record contexts to x_t; if fewer than 32, use alpha_G.
2. On those records, assign weights a_j proportional to 2^(-age_j/52). Normalize to sum one.
3. Compute n_eff = 1/sum a_j^2. This measures weight concentration only, not effective independent sample size.
4. If fewer than 33 eligible record contexts exist, n_eff < 16, or the 32nd-neighbor distance exceeds the analogous leave-one-out 95th-percentile record-context distance, use alpha_G.
5. Otherwise minimize sum_j a_j ES_j(alpha) + 0.05 ||alpha-alpha_G||^2 on the same simplex, yielding alpha_L.
6. Set beta = min(0.5, n_eff/(n_eff+64)) and alpha_S10 = (1-beta)alpha_G + beta alpha_L.

The final scenario probabilities are p_S10[i] = sum_e alpha_S10[e] p_e,t[i]. A fallback changes the distribution toward global information; it never automatically orders cash or creates a confidence claim.

Every expected return is inside the convex hull of the library outcomes. Shared atoms do not make the experts independent information sources. This model can reweight observed situations but cannot extrapolate a new return mechanism outside its support. Whether that restricted capacity is adequate is part of the experiment, not an assumed advantage.

The Euclidean distance kernel is conditionally negative semidefinite. On the simplex tangent space, the energy-score quadratic is convex. Positive L2 regularization makes this problem strictly convex on that space. Implement a deterministic active-face solve over the seven nonempty subsets of the three experts, solving each equality-constrained quadratic, rejecting infeasible candidates and selecting the lowest objective. Check simplex feasibility and KKT residuals. Do not present an unconstrained solve with clipped negative weights as the optimum.

Numerical requirements: float64; probabilities sum to one within 1e-12; symmetry residual <=1e-12 times matrix scale; feasibility/KKT residual <=1e-9 times max(1, objective coefficient scale). Negative curvature beyond 1e-10 relative scale is an error, not permission to project away a substantive defect. Singular/ill-conditioned solves or uncertified optimum stop the run with a retained receipt. Tiny negative weights within tolerance may be set to zero and renormalized only after rechecking objective/KKT. Tie-break by lexicographic alpha. A separate dense-simplex enumeration is a verification reference.

## 6 Delayed chronological state machine

At each weekly origin, process in this order:

1. Validate timestamps and append only outcomes with availability < decision_time. Never append the current or an immature four-week label.
2. Score stored past forecasts when their labels first mature. Append each maturity record exactly once.
3. Update the bounded scenario library and matured-record queue under the fixed window rules.
4. Issue and save U, R, C, global, contextual and control forecasts. Save state before the current outcome exists.
5. On every fourth eligible anchor in the declared evaluation fold, evaluate policies, save the decision, and simulate the later open fill under the stated ideal convention.
6. Carry actual holdings and cash through every intervening session. Price changes and dividend events update accounting; no future event changes an earlier decision.
7. At maturity, score the issued forecast and append feedback for later decisions.

This is a frozen online algorithm: its parameters do not change after outcome inspection, but its documented state uses past feedback causally. A separate no-feedback replay must not be substituted for it. Retain an offline byte-for-byte equivalence test against incremental/checkpointed execution. Repeated processing after a restart must not duplicate dividends, maturity records or trades.

## 7 Decisions from the same joint distribution

Initial fold wealth is $100,000 in cash. Let current risky dollars h_a>=0, cash c0>=0 and V=c0+sum h_a. A grid target w has components in {0,.05,.10,...,.40}, sum w<=.80. The finite grid includes cash. In addition, include the exact keep-holdings action if current weights meet the same decision-time caps. Market drift may breach caps between decisions; those are not continuous risk limits. If holding is infeasible, rebalancing is mandatory under the stated decision-time constraint.

For one-way proportional fee c in {0,.001,.002,.004}, find post-fee NAV q from

    q + c sum_a |q w_a-h_a| = V.

Use bisection on [0,V], at most 100 iterations, ending only when both bracket and accounting residual satisfy 1e-10 max(1,V). Target risky dollars are q w; cash is q(1-sum w); entry fee is c sum |q w-h|. For the exact keep-holdings action, q=V and fee=0.

For a scenario (g_i,d_i), the hypothetical liquidated terminal wealth factor is

    F_i(w) = [q(1-sum w) + sum_a q w_a ((1-c)g_i,a+d_i,a)] / V.

The decision compares all alternatives on a common liquidated-value basis. It reserves a hypothetical exit fee in the objective; the actual ledger charges that fee only if selling occurs. It does not charge a fictional exit fee every four weeks. When c=0 and returns/dividends are zero, F=1 exactly.

Use CRRA utility U(F)=(F^(1-gamma)-1)/(1-gamma), gamma=4. Cash reserve and positive admitted g ensure F>0. Evaluate with stable log/expm1 arithmetic; invalid or nonfinite utility is an error. This utility preference is fixed for the research model, not Laura's accepted risk preference or a calibrated certainty estimate.

Two policies share distributions, grids, holdings and costs:

    nominal objective = sum_i p_i U(F_i)
    robust objective  = (1-rho) sum_i p_i U(F_i) + rho min_i U(F_i), rho=.05.

The robust expression is the exact worst expected utility over distributions (1-rho)p+rho q_other, where q_other is any distribution on the same finite atoms. It is a fixed contamination sensitivity, not a data-derived coverage bound. It cannot protect against a shock missing from all atoms and may be too conservative.

Enumerate the finite action set exactly. Include hold when feasible; choose it whenever its objective is within 1e-10 of the maximum. Otherwise break objective ties by lower dollar turnover, then lexicographic target. Cash is never forced to lose a tie. Every chosen action logs objective, alternative hold/cash values, worst atom, fees, constraints and a finite-grid optimality certificate.

The primary integrated candidate uses contextual forecasts and robust policy. Nominal policy remains a prespecified control; a better nominal outcome does not replace the primary afterward.

## 8 Realized ledger and measured risk

Actual risky dollars evolve by the observed price gross ratio; dividends add cash based on actual shares held at the event. There is no reinvestment except through an explicit later rebalance. Holdings carry across four-interval decisions, and only the final fold endpoint liquidates all risky shares with the same one-way fee. All benchmark final values use the same liquidation convention.

Use every available daily close for marked NAV and report daily-close maximum drawdown, not intraday maximum drawdown. Also retain weekly/decision NAV, realized turnover, fees, cash, asset contributions, net utility and terminal wealth. Reconcile

    NAV change = price PnL + cash dividends + cash interest - actual fees

to tolerance at every event. Interest is exactly zero in revision 1. Dividend ex-date credit remains an explicit approximation. Do not fill missing marks and silently continue.

Separate the fee bill, the fixed-decision fee drag and the change in decisions induced by costs. A fixed-decision counterfactual reuses the saved dimensionless target sequence with fees set to zero, preserving timing and re-solving the cash equation; it is labeled a counterfactual, not a new tuned policy or a pure return forecast.

## 9 Forecast diagnostics and honest uncertainty

Report primary ES, marginal CRPS, mean squared return error, variogram score of order 0.5, empirical interval coverage and width, and calibration of forecast loss probabilities. Use fixed asset scales from section 3. Variogram score is the equally weighted average over the three asset pairs of

    (|z_y,a-z_y,b|^.5 - sum_i p_i |z_i,a-z_i,b|^.5)^2.

Its purpose is complementary dependence diagnosis; neither it nor ES alone certifies economic value. For discrete quantiles, use the smallest value with cumulative probability >= requested level; report 50/80/95% central intervals with explicit tie treatment. Proper scoring is an evaluation principle, not a guarantee that fitted probabilities are calibrated. No conformal, 99% funding, or recession-probability guarantee is asserted.

## 10 Complexity, interfaces and rejection behavior

Library size n<=520, three assets, 12 features, three experts, feedback records<=104. Exact scoring costs O(n^2*3) distances per issued forecast; expert cross terms reuse that distance matrix. Store A/B after maturity so repeated stacking solves do not revisit all past atoms. Neighbor distances are O(n*12); policy enumeration is O(K*n*3), K<=730 including hold. Raw scenario/record storage is bounded; expected process memory/time must still be measured, not inferred from these orders.

All interfaces exchange typed arrays and manifest-backed record identities. Distinct types represent price gross, cash dividend per starting dollar, total simple return, utility and USD. No generic return vector is accepted at the accounting boundary.

Failures include unknown units, unsupported actions, corrupt hashes, different calendars, insufficient warm-up, absent quotes, immature labels, duplicate feedback, nonfinite arithmetic, invalid probabilities, uncertified optimizer, budget violation and missing controls. Retain nonzero exit receipts. Model failure is not silently converted to a successful cash policy; cash is an ordinary feasible economic action only when the model ran validly.

Checkpoint all transforms, library identities, forecast records, matured records, weights, chronological cursor, holdings, cash and event hashes. Persist data/source/config/environment identity. A started frozen evaluation cannot be rerun by deleting its consumption marker.

## 11 Guarantees and limits

The equations permit proofs of finite scenario moments, positive semidefinite covariance, a unique fee-accounting root, convex regularized stacking, and exact optimality over a declared finite action set. Each proof needs numerical implementation checks.

They do not prove profitable predictions, superior memory, calibration, unseen-tail coverage, continuous-action optimality, causal attribution of S8's loss, or generalization beyond the admitted histories. An adverse all-risky-assets path and a rising benchmark remain counterexamples to universal dominance. Equal/global mixing may be the best supported final model.
