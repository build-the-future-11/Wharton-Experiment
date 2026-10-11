# Source map, research claims and redistribution

The bibliography in `paper/references.bib` contains the primary works used for the methods manuscript. The following table records their role; it does not imply this implementation reproduces their reported results.

| Claim or design context | Primary source | Interpretation in FREM |
|---|---|---|
| Joint latent prediction | [Assran et al., I-JEPA, 2023](https://arxiv.org/abs/2301.08243) | Motivation for the temporal EMA-target auxiliary objective, with a separate JEPA-off control. |
| Temporal JEPA prior art | [Ennadir et al., 2025](https://arxiv.org/abs/2509.25449); [Feng et al., 2026](https://arxiv.org/abs/2609.31680) | Temporal JEPA is not novel by itself; usefulness depends on the specific setting and backbone. |
| Fixed reservoir dynamics | [Jaeger, corrected echo-state report](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf) | A sufficient contraction condition is used; no few-data profitability theorem is inferred. |
| Fourier/reservoir neighbors | [Laha, 2025](https://arxiv.org/abs/2511.14775); [Schertler et al., FRESCO, 2026](https://arxiv.org/pdf/2606.24969) | Existing combination space. FREM’s separate dense reservoir and trailing DFT are distinct from FRESCO’s frequency-domain recurrence. |
| N-gram conditional memory | [Cheng et al., Engram, 2026](https://arxiv.org/abs/2601.07372) | Conceptual neighbor, not a reproduced language-model lookup architecture or its efficiency claim. |
| Representation regularization | [Bardes et al., VICReg](https://arxiv.org/abs/2105.04906) | Motivates variance/covariance anti-collapse terms. |
| Continual backpropagation boundary | [Dohare et al.](https://arxiv.org/abs/2306.13812) | Unit replacement is a different mechanism from FREM’s bounded feedback controller. |
| Boosted trees | [Chen and Guestrin, XGBoost](https://arxiv.org/abs/1603.02754) | Tree prediction and residual correction; no graph-traversal interpretation. |
| Linear and patch sequence controls | [Zeng et al., DLinear](https://arxiv.org/abs/2205.13504); [Nie et al., PatchTST](https://arxiv.org/abs/2211.14730) | Task-specific architectural controls; not official benchmark reproductions. |
| Backtest selection risk | [Bailey et al.](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | Supports retaining failures and protecting evaluation from repeated development selection. |
| Probabilistic forecast scoring | [Gneiting and Raftery, 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) | Proper scoring framework; calibration is measured, not assumed from a chosen density. |
| Dependent-observation bootstrap | [Künsch, 1989](https://doi.org/10.1214/aos/1176347265) | General block-bootstrap motivation; the exact circular/date-balanced implementation is specified separately. |
| Multiple hypotheses | [Holm, 1979](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) | Complete secondary-family error adjustment. |
| CVaR | [Rockafellar and Uryasev, 2000](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf) | Scenario tail-risk mathematics, with explicit fractional mass at weighted tail boundaries. |
| Macro vintages | [Federal Reserve Bank of St. Louis API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) and [ALFRED](https://alfred.stlouisfed.org/help/downloaddata) | Actual observation/vintage semantics; the adapter adds an explicit conservative availability policy. |
| Taiwan trading context | [Taiwan Stock Exchange investing guide](https://www.twse.com.tw/en/about/company/guide.html) | Daily price bands, execution context and rule distinctions; not a generic foreign portfolio cap. |
| Competition source boundary | [Wharton official FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/) | Participant materials remain the authority for the current detailed competition requirements. |

## User-specific evidence

`docs/laura_case_evidence.json` identifies the current user preparation document, its revision and relevant tabs. `docs/laura_case_excerpts.json` preserves the relevant copied-case and prior reported negative-result excerpts. The source is [ROI - Wharton Global Youth Investment Challenge Prep](https://docs.google.com/document/d/1kjLz0M7yy8jxIEbeUnsFML1dLQ2_elf8U9dqq0Gka-U/edit). This is private project evidence supplied through the user's workspace, not an independently certified organizer publication.

Cash-flow facts in the mandate are supported by that source. Team policy choices remain distinguished from case obligations and securities law. Earlier APEN and graph-memory outcomes are reported historical findings whose raw run files were not independently rechecked here.

## Evidence not present

There is no market dataset, trained checkpoint, measured score, profit series, calibrated recession probability, benchmark timing, or demonstrated data-efficiency result in this release. A theorem about forgetting or finite storage is not a financial performance theorem. The related-work review sets a candidate novelty boundary, not an exhaustive novelty certification.

## Rights and external release

No external paper PDFs, proprietary price histories, paid dataset exports or API credentials are redistributed in the package. Source URLs and bibliographic metadata are included. The original user protocol and user-specific case excerpts are retained for this requested private research handoff. Review that material before sharing a public artifact.

Third-party dependencies retain their respective licenses. A source-code publication license, institutional authorship statement and final venue disclosures have not been chosen on the user's behalf. A public release should add those decisions after reviewing the project’s ownership, source rights and publication requirements. This package is a private implementation deliverable; its existence does not authorize public redistribution of later vendor data or participant-only materials.
