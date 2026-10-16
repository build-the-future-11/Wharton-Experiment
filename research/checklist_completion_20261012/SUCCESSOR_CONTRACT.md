# Proposed successor: chronological residual relevance

Status: specified development proposal, NOT FROZEN, NOT RUN, novelty not established. S2/S3 remain closed. The present request authorizes useful bounded research work, not outcome-conditioned retries or waiver of resource/data gates.

Question: does dimension-independent residual retrieval add out-of-time log-score information beyond asset identity, recency and a recent frozen base? This is an incremental empirical hypothesis, not a claim to invent kernel regression or online residual correction.

Mechanism: for unit keys use squared Euclidean distance, equivalent to twice cosine distance, rather than dividing again by dimension. Fixed temperature 1 is the first isolated intervention. Keep the original metric as control. Do not search temperatures on spent outcomes. Recent frozen base fitting and conservative gate selection are distinct interventions, not silently combined repairs.

Falsifier: nearest retrieval does not improve held-forward residual relevance versus age/asset-matched random and recent averages; or correction worsens held-forward NLL. Retain memory-off as the valid outcome. No promotion merely because a selected point estimate is positive.

Proposed bounded development: three optimization seeds (11,29,47), two chronological outer development blocks and four matched arms (off, old-distance, dimension-independent-distance, matched random), 24 prediction cells, six shared frozen base fits. Maximum 30 minutes aggregate worker time, one CPU worker/thread, 768 MiB process ceiling AND normal 512 MiB admission. No automatic retries. Five epochs maximum with identical base selection; report incomplete cells rather than drop them. The budget is a cap, not an obligation to exhaust it.

Data prerequisites: identify exact pre-2025 development-only observations and enough independent dates for nested chronology; mark all previously inspected assets/windows as development. Select fit, checkpoint-selection, calibration, residual-bank and gate blocks strictly in order with label-maturity purges. Exact dates and data hashes must be specified before this can become runnable. No protected 2025+ inputs are needed for this development proposal. Current public rows cannot establish point-in-time eligibility or global investment performance.

Conservative selection: zero correction included; select only on nested training/development blocks, never outer evaluation. Require stable benefit across inner held-forward blocks rather than any positive single-validation improvement. Exact inner splits and penalty require a prospective feasibility/power review, not arbitrary post-result assignment.

Mathematical audit of scale correction: if standardized residual Z has conditional mean m and second moment q around the old base, shifting the predictive center by a gives E[(Z-a)^2]=q-2am+a^2. For a=m this is q-m^2, nonnegative by Jensen. Student-t scale is standard deviation times sqrt((nu-2)/nu), for nu>2; it is not the standard deviation itself. A risk correction must specify the intended center and degrees of freedom before applying a second-moment multiplier. This identity is not evidence of a demonstrated predecessor validity defect and does not reopen S3.

Prior-art boundary: kernel-distance scaling, residual adaptation, shrinkage and rolling-origin evaluation are established ingredients. Cawley and Talbot (JMLR 2010, https://www.jmlr.org/papers/v11/cawley10a.html; checked 12 October 2026) documents model-selection overfitting as a general risk, not a proven cause of this S3 result. A full retrieval/residual-learning novelty comparison remains required before an IDEA READY claim.

Stop now before model implementation/execution: memory admission fails, exact admitted chronology is not fixed, independent evidence is not identified and the contribution review is incomplete. A document cannot turn these conditions into RUN READY.
