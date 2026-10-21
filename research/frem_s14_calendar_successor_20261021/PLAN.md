# FREM-S14 0.1.9 source-calendar successor

Version 0.1.8 is terminal as a pre-market source-admission failure. It passed
107 target tests and the full 19-arm generated CUDA qualification, but its
declared eight-ETF source cannot satisfy the frozen balanced-calendar contract.
No historical model cell ran.

The 0.1.9 successor fixes the cohort before market outcomes are available:

- fixed cohort: SPY, IWM, DIA, TLT, IEF, GLD and SLV;
- QQQ excluded solely because the exact provider bytes contain one additional
  date, 2011-02-17;
- exact source-byte SHA-256 allowlist frozen in the decision record and intake;
- no calendar intersection, row deletion, forward fill or price imputation;
- unchanged 19 arms, three seeds, three folds, thresholds, inference, 171-cell
  denominator, time ceiling and protected boundary.

The new release must pass source and clean-wheel tests, static checks,
reproducible packaging, saved-source readback, all 107-or-more target tests,
all 19 generated CUDA qualification arms and independent qualification
verification before historical intake. Every historical cell must then be
retained as complete, failed or not started and independently reconstructed.
Any valid negative result is terminal for this successor.
