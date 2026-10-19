# FREM-S12 literature and novelty boundary

Date: 18 October 2026. This review closes the S12 idea gate only. It does not
establish empirical support.

## Primary anchors

- Rahimi and Recht, *Random Features for Large-Scale Kernel Machines* (NeurIPS
  2007): random Fourier features approximate shift-invariant kernels in a
  finite randomized feature space.
  https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html
- Reservoir computing / echo-state work establishes fixed recurrent reservoirs
  and fading-memory state representations; universality results do not imply
  finite-sample forecast superiority.
  https://pubmed.ncbi.nlm.nih.gov/30317134/
- Berrisch and Ziel, *Multivariate Probabilistic CRPS Learning* (International
  Journal of Forecasting 2024): online multivariate probabilistic forecast
  aggregation with established asymptotic learning properties.
  https://doi.org/10.1016/j.ijforecast.2024.01.005
- Scheuerer and Hamill, *Variogram-Based Proper Scoring Rules* (Monthly Weather
  Review 2015): energy score can be insensitive to misspecified dependence;
  variogram scores provide a necessary dependence diagnostic.
  https://doi.org/10.1175/MWR-D-14-00269.1
- Salinas et al., *DeepAR* (International Journal of Forecasting 2020): global
  autoregressive recurrent probabilistic forecasting across related series.
  https://doi.org/10.1016/j.ijforecast.2019.07.001
- Lim et al., *Temporal Fusion Transformers* (International Journal of
  Forecasting 2021): attention/gating architecture for interpretable
  multi-horizon forecasting.
  https://doi.org/10.1016/j.ijforecast.2021.03.012

## Boundary

Random Fourier features, reservoirs, retrieval, neural gates, delayed online
feedback, proper-score aggregation and multivariate probabilistic forecasting
all have substantial prior art. S12 therefore makes **no claim that any one
component is algorithmically novel** and no regret-optimality claim.

The defensible contribution is an empirical mechanism study with unusually
strict evidence controls: a differentiable delayed router over finite joint
scenario laws, mandatory comparison with a recognized fixed online stacker and
non-memory distributional controls, repeated release-vintage folds, a frozen
dependence guard, compute matching, complete cell retention and independently
reconstructable cloud artifacts.

The study is worth running only if it can distinguish learned causal routing
from extra tuning, parameters, information, compute and policy choice. A simple
control win or a failure to clear the fixed practical margin closes P0.

## Baseline decision

Monthly six-dimensional research-portfolio panels are much smaller than the
large related-series regimes used to motivate DeepAR or TFT. P0 therefore uses
a parameter-matched causal autoregressive MLP as the neural control, alongside
rolling empirical, shrinkage Gaussian, shrinkage Student-t and regularized VAR
finite laws. DeepAR and TFT remain contextual prior art, not automatically
credible controls for this small-data setting. Any later addition is a new
versioned protocol.

## Novelty disposition

`IDEA_READY_FOR_BOUNDED_EMPIRICAL_MECHANISM_STUDY`

This disposition authorizes implementation against generated data. It does not
authorize market acquisition, development consumption, confirmation or claims
of originality beyond the stated empirical systems contribution.

