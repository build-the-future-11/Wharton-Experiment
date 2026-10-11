# M07 — Eigen-JEPA

## Objective
Predict **forward covariance** for trailing windows **20** and **60** with PSD parameterization (Cholesky `L L^T`).

## Outputs
- Leading eigenvalues (concatenated in `predict`)
- Eigenspace projectors `P = U U^T`
- **Small eigengap flags** when consecutive eigenvalue gaps < threshold

## Metrics
- **Chordal / projector Frobenius** distance between predicted and empirical subspaces
- **Sign-flip invariance**: `P(U)` equals `P(-U)`
