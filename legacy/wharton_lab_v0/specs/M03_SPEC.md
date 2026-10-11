# M03 — Conditional Nonlinear Factor Model

## Structure

\[
r_{i,t} \approx \beta(c_{i,t})^\top f_t, \quad \beta: \mathbb{R}^d \to \mathbb{R}^K \text{ (MLP)}.
\]

## Factor learning (train)

PCA on demeaned training panel \(R \in \mathbb{R}^{T \times N}\) yields \(f_t \in \mathbb{R}^K\).

## Factor forecast (predict)

Each \(f_{t,k}\) follows AR(1) fit on train history:
\[
f_{t+1,k} = c_k + \phi_k f_{t,k} + \eta.
\]

**Binding:** at predict time use \(\hat{f}_{t+1}\) only — not contemporaneous cross-sectional factors from the future.

## Prediction

\[
\hat{r}_{i,t+1} = \beta(c_{i,t})^\top \hat{f}_{t+1}.
\]
