# M02 — Factor-Residualized Cross-Sectional Learning-to-Rank

## Factor exposure (historical)

Rolling OLS on \(s \in [t-W, t-1]\):
\[
r_{i,s} = \alpha_{i,t} + \beta_{i,t}^\top f_s + \epsilon_{i,s}.
\]

\(\beta_{i,t}\) uses only past returns and factor returns.

## Residual label (at maturity)

When the forward window realizes factors \(f_{t+1}\) (label maturity only):
\[
\tilde{y}_{i,t} = y_{i,t} - \beta_{i,t}^\top f^{\mathrm{realized}}_{i,t+1}.
\]

## Pairwise ranker

Within each date \(d\), sample pairs \((i,j)\) and minimize logistic loss on
\(\mathbb{P}(\tilde{y}_i > \tilde{y}_j) \approx \sigma(s(x_i) - s(x_j))\).

## Primary metric

Date-level Spearman IC:
\[
\mathrm{IC} = \mathbb{E}_d \left[ \rho_{\mathrm{Spearman}}(s_d, \tilde{y}_d) \right].
\]
