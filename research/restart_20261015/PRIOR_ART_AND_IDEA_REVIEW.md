# Prior art and adversarial idea review

Reviewed 2026-10-15. This is a focused primary-source review of proposed repairs, not an exhaustive novelty clearance. Source publication/version dates below are separate from retrieval date. No standard method below is claimed as a new FREM contribution.

| Primary source and date | Verified scope | Consequence for this project |
|---|---|---|
| Boyd et al., [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf), 2017 | Full paper consulted, including forecast-error risk, constraints and transaction costs. Its methods presuppose forecasts rather than establish their quality. | Cost-aware optimization and uncertainty penalties are established tools. A new solver or penalty alone would not substantiate FREM novelty or profitable forecasts. |
| Elmachtoub and Grigas, [Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005v5), first submitted 2017, version 5 in 2020 | Full HTML consulted for objective, assumptions and failure examples. Decision-focused loss and the SPO+ surrogate have specified optimization assumptions. | Decision relevance is a useful design principle; the results cannot be transferred automatically to this nonlinear, dependent portfolio sequence. |
| Liu and Grigas, [Risk Bounds and Calibration for a Smart Predict-then-Optimize Method](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b943325cc7b7422d2871b345bf9b067f-Abstract.html), 2021 | Official proceedings abstract consulted; full proof not audited. | Calibration guarantees are conditional on mathematical settings, not general profit certificates. Do not cite as a theorem about FREM. |
| Liu et al., [Smart Predict-then-Optimize Method with Dependent Data](https://arxiv.org/abs/2411.12653v1), 2024 | Full HTML consulted for stationary beta-mixing assumptions and experimental setting. | Dependence has existing theory; unverified mixing/stationarity assumptions cannot justify confidence bounds for this project. |
| Herbster and Warmuth, [Tracking the Best Expert](https://mwarmuth.bitbucket.io/pubs/J39.pdf), 1998 | Author-hosted full paper consulted for expert tracking and switching. | Fixed-share adaptation has prior art. Relative loss guarantees are not absolute market-profit guarantees, and this project's delays/context require separate analysis. |
| Ledoit and Wolf, [Honey, I Shrunk the Sample Covariance Matrix](https://ledoit.net/honey.pdf), published 2004; working draft 2003 | Author index/search abstract verified. Full PDF retrieval timed out; exact estimator derivation not checked here. | Shrinkage covariance is an established baseline. No named Ledoit-Wolf estimator is implemented or certified by this restart. |
| DeMiguel, Garlappi and Uppal, [Optimal Versus Naive Diversification](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901), 2009 | Primary-paper abstract available through a university copy; full paper retrieval unavailable. | Equal weighting is an essential comparator. Findings about those datasets are not a universal superiority result or a sample-size calculation for our three ETFs. |

## Candidate mechanisms and selection

| Candidate | Mechanism and assumptions | Falsifier and main objection | Disposition |
|---|---|---|---|
| A: Horizon-aligned simple forecast baseline | Directly model the return interval being held, using only matured targets and scheduled calibration. Assumes exploitable predictable structure and enough development observations. | Does not improve declared forward calibration and economic endpoints under the same policy; more fitting or fresher data could explain any gain. | First baseline to specify if data and compute become admissible. This is a repair/control, not a novelty claim. |
| B: Joint finite-return decision layer | Use synchronized return scenarios, joint risk and exact fees. Assumes scenario probabilities/tails carry useful out-of-time information. | At fixed forecasts and constraints it fails net benefit/downside criteria; apparent gains vanish under common exposure. | Second, separate policy comparison. Reference mathematics implemented; no market estimator or optimizer fitted. |
| C: Decision-focused learned forecasting | Train a loss that reflects a fixed portfolio decision problem. Requires an explicit feasible set, loss, differentiability/surrogate argument, temporal treatment and matched compute. | No improvement over A/B, or improvement depends on extra search/information. Existing SPO literature limits the novelty claim. | Defer; current evidence does not justify this additional complexity. |
| D: New contextual memory mechanism | Use past observable context to predict conditional expert advantage under delayed feedback. Requires demonstrable conditional reliability differences. | Fixed/global/recent-loss controls match it, or mechanism-off ablations preserve performance. | No defensible new mechanism established. Existing S8 contextual increment is too small to supply the missing case. |

The strongest currently testable question is: **Does correctly matching forecast horizon and jointly modeling decision risk improve forward net outcomes over the declared S8 surrogate when forecasts, information, costs, limits and compute are controlled?** This is a proposed diagnostic comparison. It does not yet identify a genuinely new mechanism, an unused admitted evaluation set, or a feasible precision target.

## Adversarial objections and required remedies

1. **Novelty:** every repair above has substantial prior art. Specify a genuinely distinct mechanism or frame a bounded empirical failure analysis; combining standard components is not sufficient.
2. **Identification:** changing forecasts, refitting, covariance and policy together cannot identify the cause of improvement. Stage one-factor comparisons with common controls.
3. **Selection:** the primary failed; selecting a favorable S8 secondary variant or trading around known 2022 losses is outcome-driven redesign. Keep those results closed.
4. **Data:** current-vintage prices and legacy feature access cannot establish historically available, globally untouched evidence. Admit source timestamps and exposure metadata before reading new outcomes.
5. **Uncertainty:** three initialization seeds share the same market history. Date dependence and overlapping holding targets must be included; more seeds do not supply new market regimes.
6. **Fairness:** moving from cash to risky holdings can appear to improve returns without improved prediction. Retain broad benchmarks and add matched constraints/exposure controls.
7. **Mathematics:** log-score gains do not imply finite expected simple-return gains; use the separate checked contracts and test estimator calibration.
8. **Complexity:** memory, JEPA and reservoirs must outperform removal controls at matched information and cost. Do not expand architecture because earlier versions failed.
9. **Generalization:** three ETFs on one exposed path do not support global-stock, crisis-probability, default-aware or client-funding claims.
10. **Feasibility:** standard training admission currently fails; the earlier S6-specific RAM override is retained as historical authorization for that campaign, not silently rewritten. Data and scientific readiness remain unmet regardless of RAM.

**Checkpoint 1 is NOT READY.** The user's supplied pipeline says not to implement a successor before this gate. Audit tooling and mathematical references are complete preparatory work; no new forecasting architecture, fit, protected evaluation or performance claim follows from them.
