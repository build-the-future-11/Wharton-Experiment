# FREM-S14 0.2.0 semantic-reproducibility successor

## Scope and status

This is a new, prospective historical-proxy development candidate. It preserves
S14 Version 8 (0.1.9) unchanged as an invalid execution: the independent
readback found cross-runtime summary disagreement in 90/114 diagnosed cells,
Fold 2 did not reconstruct its preprocessing, and the original suite exhausted
its frozen clock before notebook-level verification. Those artifacts are not
performance evidence and are not reused as a result.

The candidate is **not yet numerically qualified, launched, or evaluated**.
No protected outcomes are in scope.

## Validity repairs frozen before execution

| Defect | Cause | Repair | Acceptance evidence |
| --- | --- | --- | --- |
| Date-level NLL changed across supported SciPy/NumPy runtimes | Scalar reduction was recomputed from platform-specific likelihood calls | Persist target-runtime NLL, Brier and coverage vectors; derive reported scalar scores with ordered `math.fsum` reductions | Target qualification and an independent verifier agree on the saved score vectors and all scalar reports |
| Fold 2 preprocessing did not reconstruct | Train-only medians/MAD and target standard deviation depended on NumPy reductions | Define these scalers using sorted Python-float medians and `math.fsum` population standard deviation | Exact byte equality of retained and independently rebuilt scaler arrays on target and verifier runtimes |
| Mandatory readback exceeded Version 8's fold clock | `2800` seconds covered fits and in-run semantic verification together | Increase only the successor fold ceiling to `4000` seconds while retaining the global notebook ceiling of `13500` seconds and reserving a separate suite-verification command | All three fold campaigns complete their in-run semantic verifier; the outer 13500-second deadline remains enforced |

The architecture, seven-ETF cohort, 19 arms, three seeds, folds, practical
NLL threshold (0.01), bootstrap construction, data hashes, Internet-off
policy, private Kaggle visibility and no-imputation rule are unchanged.

## Execution order

1. Build a fresh hash-bound 0.2.0 wheel, release archive, notebook and private
   Kaggle metadata; run static parsing and dependency-free release checks.
2. On the exact private T4 target, run all warning-strict tests and the fixed
   19-arm generated qualification, including its independent verifier.
3. Only if both pass, run the fixed 171-cell historical-proxy suite once.
   Preserve every cell and the complete outer manifest regardless of outcome.
4. Run the target suite verifier and independently re-run the verifier from a
   separately controlled environment. Do not interpret any metric unless both
   pass.
5. Classify the predeclared result as supported, mixed, negative, inconclusive
   or invalid. A positive outcome is neither required nor promised.

## Claim boundary

Even a fully verified historical-proxy result is exploratory and cannot establish
tradable profitability, point-in-time data validity, or protected confirmation.
