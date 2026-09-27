# Lockbox summary

Opened once after freeze (2026-09-26T14:03:48Z). Do not re-open without amendment. Raw numbers below are unchanged; their interpretation was revised on 2026-09-27 (protocol/DECISIONS.md D-050..D-053).

| Dataset | Selected (B_RIDGE) MSE | Persistence MSE | Ridge lower? | Interpretation |
|---|---:|---:|---|---|
| synthetic seed=997 | 9.89328e-05 | 0.0281491 | True | **INVALID — leakage.** Features include the contemporaneous latent that generates the label; ridge MSE equals the noise variance (0.01² = 1e-4). |
| ETF offset=800 (asset 0 = AGG) | 2.14083e-05 | 2.11771e-05 | False | **UNSUPPORTED.** Negative result stands (minor label leak in the vol feature only makes it more conservative). |

- H1 (forecast beats persistence): **not supported on any valid evidence.** Synthetic arm is leakage-contaminated; ETF arm is negative.
- H2 (EWMA cov finite Frobenius error): true on both panels, but trivially so — it is a numerical sanity check, not a performance claim.
- Selection: B_RIDGE over M04 by the 1% simplicity tie-break on "val folds 0–2". Those folds are the same train/test split as full fold 3 (D-052), and the pooled MSE is dominated by the leaky synthetic track, so selection is not evidence of skill.
- Freeze fingerprint `c33889d…` does not match any commit (no commit existed at freeze time); lockbox-relevant source files predate the freeze by mtime. Status: PARTIALLY VERIFIED.

Post-lockbox repaired-pipeline evidence (exploratory, not confirmatory): `REPAIRED_EVAL_SUMMARY.md`.
