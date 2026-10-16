# Prospective CL-16: minimum-variance objective ties

Recorded 15 October 2026 before any S10 market evaluation. CL-14's absolute 1e-10 **utility** tolerance applies to policy utility. Minimum-variance benchmark values have squared simple-return units; no variance tolerance was specified.

For the 45-point minimum-variance grid, compute the declared binary64 quadratic variance for every candidate. The best set contains exactly the candidates whose computed variance equals the global minimum in binary64. Within that set, use the existing actual-dollar-turnover tolerance 1e-10*max(1,V), then lexicographic weights. Do not import the utility tolerance into a different unit. There is no approximate variance tie threshold or new fitted parameter.

This is a prospective deterministic numerical clarification, not a prediction of better returns. It remains subject to analytic covariance and permutation/order tests. Earlier design versions and utility rules remain preserved; a future scientific freeze must include this addendum.
