# Related work for the integrity and development research pass

Date: 2026-10-19. Literature checked through primary papers, author-hosted copies, publisher abstracts, and official documentation. This is a focused methodological review, not an exhaustive novelty search.

**Proposed contribution:** reproducible identification and correction of specific research-software integrity failures, with bounded developmental comparisons after those corrections. **Novel financial method status: `NOT_ESTABLISHED`.** No market-alpha result or completed integrated FREM architecture is established by this document.

## 1. Source and claim boundaries

The scientific source being repaired is `build-the-future-11/Wharton-Experiment` at commit `fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c`. Its M01–M12 implementations and task adapters are the objects of study. The separate public FREM prototype is a different implementation and a partial prototype; its name cannot establish that the repaired models form a trained Fourier–reservoir–JEPA system.

The local implementation audit is grounded in [the pinned controller](https://github.com/build-the-future-11/Wharton-Experiment/blob/fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c/src/wharton_lab/models/m11/mpc_verified.py), [accounting](https://github.com/build-the-future-11/Wharton-Experiment/blob/fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c/src/wharton_lab/models/m11/accounting.py), [causal task builders](https://github.com/build-the-future-11/Wharton-Experiment/blob/fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c/src/wharton_lab/research/causal_portfolio.py), and [task runner](https://github.com/build-the-future-11/Wharton-Experiment/blob/fc90bc05b4a8df90f5b3ca5da86e5ed9791ff54c/scripts/run_portfolio_research.py). Published names are treated as lineage labels; implemented objectives and interfaces determine claims.

## 2. Checked primary references

### R1. Financial machine-learning benchmarks and capacity

Shihao Gu, Bryan Kelly, and Dacheng Xiu (2020). **Empirical Asset Pricing via Machine Learning.** *The Review of Financial Studies*, 33(5), 2223–2273. [Author-hosted published paper](https://dachxiu.chicagobooth.edu/download/ML.pdf); [DOI](https://doi.org/10.1093/rfs/hhaa009). Checked introduction and conclusion, including the comparison of nonlinear interactions and model depth.

The paper compares predictive methods in empirical asset pricing and attributes an important part of the stronger methods' performance to nonlinear predictor interactions. **Implication for this repository:** a richer model beating linear ridge does not isolate the benefit of its named memory, graph, or latent-prediction mechanism. A subsequent mechanism study needs nonlinear and capacity-aware controls using the same admitted information. The current small synthetic suite does not reproduce this paper's market panel or economic results.

### R2. Multiple testing in finance

Campbell R. Harvey, Yan Liu, and Heqing Zhu (2016). **… and the Cross-Section of Expected Returns.** *The Review of Financial Studies*, 29(1), 5–68. [Publisher abstract and metadata](https://academic.oup.com/rfs/article/29/1/5/1843824); [DOI](https://doi.org/10.1093/rfs/hhv059). Checked the publisher abstract; no claim of a full-paper replication is made.

The paper argues that extensive factor searches change the evidential hurdle for a newly reported finding. **Implication:** the family of attempted models, worlds, ablations, and revisions must remain visible. This development pass will not promote an isolated favorable result to a financial discovery. Its small seed count cannot be made confirmatory by applying a generic threshold borrowed from this paper.

### R3. Repeated model search and retained failures

Halbert White (2000). **A Reality Check for Data Snooping.** *Econometrica*, 68(5), 1097–1126. [Published paper, university-hosted copy](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf); [DOI](https://doi.org/10.1111/1468-0262.00152). Checked the introduction, search-family null, and discussion of dependent-data bootstrap inference.

White addresses whether the best specification found in a search improves on a benchmark after accounting for that search. **Implication:** record unsuccessful attempts and evaluate predeclared paired comparisons, rather than presenting the best observed cell alone. No Reality Check implementation or corrected discovery claim is supplied by the current pass. Reusing historical evaluation outcomes for development does not restore their protected status.

### R4. What the original I-JEPA method actually does

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas (2023). **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.** CVPR 2023. [Paper and version history](https://arxiv.org/abs/2301.08243); [checked full text](https://arxiv.org/pdf/2301.08243).

I-JEPA predicts target-region representations from visible image context. The paper defines a learned context encoder and predictor, a moving-average target encoder, and a masking design. **Implication:** a random projection, supervised covariance head, or concatenation of features alone does not demonstrate implementation of that learning mechanism. A finance adaptation must identify its actual prediction targets, optimization path, collapse control, causal information boundary, and downstream comparison. Image-representation results establish no return-forecasting advantage here.

### R5. Echo-state networks and trained readouts

Herbert Jaeger (2001; checked corrected version dated 2010). **The “echo state” approach to analysing and training recurrent neural networks.** GMD Report 148, with erratum. [Author-affiliated university copy](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf). Checked introduction, training description, and the existence and scope of the appended correction.

The report develops recurrent states driven by input history and learning through output connections. It also provides an explicit correction to an earlier definition. **Implication:** an actual reservoir claim requires a persistent recurrent update and a documented readout/training rule; a fixed random feature map is not enough to establish that claim. Initialization, state resets, burn-in, and trained parameter counts must be recorded. Neither a convenient matrix scaling nor a successful smoke test establishes all theoretical stability conditions.

### R6. Liquid-state computing and its assumptions

Wolfgang Maass, Thomas Natschläger, and Henry Markram (2002). **Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.** *Neural Computation*, 14(11), 2531–2560. [Author-institution paper](https://igi-web.tugraz.at/PDF/130.pdf); [DOI](https://doi.org/10.1162/089976602760407955). Checked the primary paper's abstract and introduction.

The liquid-state framework uses transient recurrent dynamics and trained readouts to process temporal inputs. Its computational statements rely on specified conditions. **Implication:** the presence of memory-like terminology does not prove useful financial memory or state separation. A future reservoir experiment should compare the recurrent route with an otherwise matched route that removes recurrence and with matched finite-history features. This foundational neural-computation work supplies no empirical finance result for FREM.

### R7. Predictability must be evaluated together with trading costs

Nicolae Gârleanu and Lasse Heje Pedersen (2013). **Dynamic Trading with Predictable Returns and Transaction Costs.** *The Journal of Finance*, 68(6), 2309–2340. [Checked publisher abstract](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080); [DOI](https://doi.org/10.1111/jofi.12080).

The paper derives a dynamic policy in which trading costs and the evolution of predictive signals influence allocation. **Implication:** a forecasting metric alone cannot establish economic usefulness. Any later portfolio claim requires executable transactions, positions carried through time, and explicitly charged costs under one accounting convention. The present controller's proportional-fee liability ledger is its own hypothetical model; it is not a replication of this paper's optimization problem or empirical findings.

### R8. Dependence-aware covariance estimation

Whitney K. Newey and Kenneth D. West (1987). **A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.** *Econometrica*, 55(3), 703–708. [Published-paper scan](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1987.pdf); [journal record](https://www.jstor.org/stable/1913610); [DOI](https://doi.org/10.2307/1913610). Checked the journal metadata and retrieved the published-paper scan; the operational ordering requirement was checked directly in R9.

This is the foundational reference for the HAC estimator named in the older repaired evaluator. **Repository-specific consequence:** changing the order of loss observations changes the lagged products used by that evaluator. This is an implementation concern independently demonstrated by the six-value analytic example in the companion specification, not a new econometric result.

### R9. Official HAC implementation contract

Statsmodels developers. **`statsmodels.stats.sandwich_covariance.cov_hac` documentation**, stable site displaying version 0.15.0 when checked on 2026-10-19. [Official documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html). Checked the stated time-series assumptions and Bartlett-weight description.

The documentation specifies a single series ordered as consecutive, equally spaced periods. **Implication:** concatenating reverse-ordered folds, or joining observations across omitted periods as if adjacent, does not satisfy that interface assumption. A new analysis must retain time identifiers and use an explicitly justified gap treatment; changing a function name to “HAC” does not repair the underlying order.

## 3. What can be contributed now

The contribution presently within reach is an auditable set of defect reproductions, bounded repairs, timestamp-admission contracts, and preserved before/after verification receipts. These are useful engineering and reproducibility results. They do not establish that the defects are previously unknown in the wider literature.

The next scientific question is model-specific: under the declared synthetic information set, does a particular implemented component improve its own declared loss against both its ablation and the information-matched baseline? The [development specification](../research/DEVELOPMENT_RESEARCH_SPEC_20261019.md) fixes how to ask that question without pooling incompatible targets or presenting development results as confirmation.

A later novelty claim would require a distinct mechanism stated mathematically, comparison against the closest implementations, controls for additional features and model capacity, and evidence that the mechanism explains an effect beyond these established ingredients. Current status remains `NOT_ESTABLISHED`; complete integrated FREM training and market alpha are outside the evidence admitted by this pass.
