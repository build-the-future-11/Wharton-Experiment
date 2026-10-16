# FREM S10 prior art and design choices

Checked 15 October 2026 against the primary sources linked below. This focused review covers the selected mechanism and its main limitations; it is not exhaustive novelty clearance. Publication/version dates are distinguished from this access date. No paper's financial-performance or coverage theorem is transferred to S10.

| Source | Verified date and access | Relevant established result or scope | S10 implication |
|---|---|---|---|
| Gneiting and Raftery, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | JASA, 2007; author-hosted full PDF accessible | Proper scoring rules and the multivariate energy score | Score the empirical joint law; propriety does not mean a fitted forecast is calibrated |
| Yao, Vehtari, Simpson and Gelman, [Using stacking to average Bayesian predictive distributions](https://arxiv.org/abs/1704.02030), [published paper record](https://aaltodoc.aalto.fi/items/5baa1af7-7be4-40ab-9973-b24a02d36062) | First preprint 6 April 2017; Bayesian Analysis publication September 2018; abstract and author paper text available | Combine predictive distributions using proper-score objectives | Optimizing mixture quality is established; S10 uses causal issued forecasts rather than ordinary leave-one-out information |
| Yao, Pirš, Vehtari and Gelman, [Bayesian hierarchical stacking](https://arxiv.org/abs/2101.08954) | Submitted 22 January 2021, revised 20 May 2021; abstract/version record inspected | Input-varying partially pooled mixture weights, including structured/time-series settings | Context-dependent weights with shrinkage are close prior art, not a new general principle |
| Bertsimas and Kallus, [From Predictive to Prescriptive Analytics](https://arxiv.org/abs/1402.5481), [published record](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2018.3253) | First preprint 22 February 2014; Management Science 2019; abstract inspected | Contextual data-driven stochastic decisions, including local learning | Nearest-neighbor scenarios and decision optimization are established; assumptions and guarantees require separate verification |
| Boyd et al., [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) | August 2017 publication; author page inspected | Joint treatment of returns, risk, costs and constraints; forecast quality remains outside that optimizer's contribution | Holdings and costs belong in decisions, but an optimizer cannot manufacture a useful forecast |
| Bertsimas and Van Parys, [Bootstrap Robust Prescriptive Analytics](https://arxiv.org/abs/1711.09974) | Submitted 27 November 2017, revised 8 June 2021; abstract/version record inspected | Robust prescriptive methods address overfitting with an entropic/bootstrap formulation | Robust local decisions have prior art; S10's simpler finite contamination objective is different and inherits no bootstrap guarantee |
| Scheuerer and Hamill, [Variogram-Based Proper Scoring Rules](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf) | April 2015, Monthly Weather Review; NOAA-hosted PDF inspected | Energy-score sensitivity to correlation misspecification can be limited; variogram scoring is complementary | Include dependence-focused diagnostics, not just one favorable aggregate score |
| Pic, Dombry, Naveau and Taillardat, [Proper scoring rules based on aggregation and transformation](https://ascmo.copernicus.org/articles/11/23/2025/index.html) | Published 13 March 2025; full article accessible, abstract/introduction consulted | Propriety alone does not ensure informative characterization; complementary transformations help diagnose forecast properties | Report calibration, marginal scores and dependence diagnostics with an unchanged primary endpoint |

## Candidate selection

| Candidate design | Potential benefit | Main objection | Decision |
|---|---|---|---|
| Larger JEPA/reservoir or transformer ensemble | More expressive temporal structure | Existing component contribution is unsupported; substantial search/compute and weak identification | Defer pending a specific mechanism and fair component experiment |
| Direct differentiable decision-focused training | Align representation with policy utility | Can overfit the chosen utility/cost regime; requires a defensible surrogate and stronger data | Defer from revision 1 |
| Smooth Gaussian/lognormal density mixtures | Continuous density scores with finite exponential moments | Joint dividend/tail and density calibration introduce additional modeling choices | Not selected; would require its own density contract before original NLL testing |
| Global joint empirical scenarios with simple policies | Transparent, finite moments, strong falsifier for complex routing | Sparse tails and limited conditional signal | Required incumbent and controls |
| Delayed contextual stacking with shrinkage and carried holdings | Tests incremental conditional reliability and decision translation separately | Very close to established stacking; may collapse to global or cash | Selected bounded S10 hypothesis, with terminal failure allowed |

The choice favors a model whose major mechanisms can be isolated within existing resource limits. It does not assert that this family is universally superior to tree, neural, econometric or foundation models. Broad SOTA claims would require task-appropriate strong external models under matched information/search budgets; that campaign is outside this revision.

## Strongest falsifiers

1. Global stacking matches or beats contextual stacking on properly issued, matured, out-of-time predictions.
2. The apparent improvement disappears against equal mixing, or exists only in a favorable fold/ablation.
3. Proper-score improvements fail net-wealth objectives under the same policy and costs.
4. Robustness produces a cash-heavy system that misses the broad benchmark without the declared risk benefit.
5. Accounting/fill assumptions explain the apparent economic gain.
6. Available date dependence and sample size make the practical margin unresolved.

No observation above would justify deleting a failed control, replacing the primary with a better secondary, weakening the old thresholds or rerunning the same frozen outcome until it becomes positive.
