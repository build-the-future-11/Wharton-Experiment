# Release verification: scope and limits

**Historical authoring-time record: source was authored and statically reviewed; scientific status remains NOT EVALUATED.**

This document preserves the original no-runtime authoring boundary. Runtime work occurred later and is recorded separately: 163 FREM tests, 67 RE-JEPA v0.3 tests, 45 RENG-JEPA v0.1 tests, full-scope Ruff, static validation, and three package builds passed during the 2026-10-11 repository cleanup. See `research/repository_cleanup_verification_2026-10-11.json`. No admitted market data, market model run, protected evaluation, or empirical result was produced.

The user explicitly asked for the full implementation and research package without model runs or experiments. This verification respects that boundary. The machine-readable inventory in `STATIC_VERIFICATION.json` contains the exact final Python/JSON counts, authored test-function inventory and syntax findings.

## Performed

- Parsed every packaged Python source, test and helper using Python's abstract syntax tree parser, then compiled the syntax tree to an in-memory code object **without executing it**.
- Checked that referenced relative Python modules exist, and parsed project JSON and TOML syntax.
- Independently reviewed model/data/evaluation/portfolio interfaces and compared the manuscript's equations and stated behavior with the implementation.
- Reviewed primary-source terminology, temporal JEPA/Fourier-reservoir prior art, the Taiwan price-band distinction and the provenance of Laura's copied case.
- Built the methods paper with LaTeX and BibTeX, checked resolved citations and layout diagnostics, rendered every page, and visually inspected the contents, architecture, equations, liability section, tables and references.
- Inspected the release archive and verified its file hashes and archive integrity as file operations. These checks do not execute the research package.

## Concrete corrections made during static review

| Issue | Correction in the delivered source/specification |
|---|---|
| Learned uncertainty lost at scoring | Evaluators require saved per-observation Student-t degrees of freedom rather than silently using a fixed value. |
| Misleading observation/seed comparisons | Exact outcomes, execution/availability times, full configured cells and the complete seed-observation grid are checked. |
| Unequal mixed-frequency weighting | Repeated observations are averaged within asset/UTC date before equal asset and date weighting; tuning, scoring and reporting share that hierarchy. |
| Too-short bootstrap blocks | Actual execution/outcome-maturity overlap constrains block length, beyond a nominal horizon alone. |
| Mutable confirmatory analysis | Exact protocol bytes and analysis-source hashes are retained; confirmation settings and primary/sensitivity status are enforced. |
| Missing multiplicity family | Complete prescribed secondary comparisons receive Holm adjustment; partial families cannot establish support. |
| Unfrozen provenance | Raw panel/macro files and their sidecar source manifests are hashed. |
| Repeated protected use | A persistent consumption receipt prevents accidental repeat use of the same freeze, including after an interrupted attempt. |
| Suite settings bypass validation | Every merged variant passes the same validation, including duplicate/configuration and budget checks. |
| Unequal or incomplete time limits | Baselines share the declared per-fit ceiling, and matrix/suite timers include preparation and completion; native calls remain cooperative. |
| All-missing perturbation scale | Nonfinite severity is rejected; explicit zero-median/unit-width fallbacks are recorded. |
| Changed macro schema across cutoff | The declared series universe survives pre-protected filtering, including all-missing series. |
| Log returns passed as simple expectations | Dated return/horizon contracts and the bounded scenario bridge make the distribution change explicit. |
| Null forecast blocks intended fallback | Null remains missing and the risk-only path requires an explicit unavailable-forecast declaration. |
| Incorrect entry-fee basis | The round-trip ledger and allocation layer use consistent post-fee self-financing accounting. |
| Incomplete selected checkpoint | Fitting that cannot complete a requested validation fails instead of presenting an unselected checkpoint as valid. |
| Mutable online checkpoint | Frozen checkpoint state is retained separately from any explicit adaptive-evaluation state. |
| Paper/code discrepancies | JEPA loss normalization, residual-tree clipping order, retention-policy comparison, constraint scope, statuses and liability-module mapping were reconciled. |

`INDEPENDENT_REVIEW.md` retains the original observations and its final source-review addendum. Keeping the earlier finding makes the review auditable; it does not mean the final code still has that visible defect.

## Not performed during the original authoring session

At authoring time, no project modules were imported, no environment/dependency installation was tested, and no authored test was executed. Later engineering verification supersedes only those runtime-compatibility unknowns. There has still been no market-data acquisition, admitted-market neural training, baseline fit, model inference, portfolio optimization, or backtest. No market forecast, execution, or funding outcome was produced.

Passing tests and syntax checks do not prove complete numerical stability, scientific validity, API behavior across every dependency version, provider compatibility, or absence of every bug.

## Remaining execution gates

1. Install in a clean Linux/macOS environment and record the actual resolved dependencies.
2. Execute all authored integrity tests and retain any failures.
3. Acquire and validate licensed data with truthful time, action, FX, eligibility and terminal-outcome provenance.
4. Perform the bounded first development exercise and verify artifacts, fitting cutoffs, comparison completeness and actual compute.
5. Select development work within its declared budget, then freeze before protected confirmation.
6. Resolve the official participant universe/rules and the team's numerical policy before Laura-specific allocation.
7. Complete empirical calibration, portfolio evidence, claim-to-artifact audit and final manuscript results.

The implementation's stated scope is also material. The portfolio bridge is one synchronous hypothetical allocation, and the supplied realized-return ledger accepts synchronized nonoverlapping intervals. It is not a full asynchronous settlement/holdings simulator. A positive immediate cash floor does not itself certify all ten operating liabilities. Zero-recovery defaults require a point-mass outcome/wealth treatment rather than deletion from the study.

## Repeating this static check

```bash
python scripts/validate_source.py --output docs/STATIC_VERIFICATION.json
```

This helper uses the standard library and does not import `frem`, pytest, PyTorch or any scientific model. The inventory records its narrower result explicitly. After any code edit, recheck the affected runtime behavior in the later authorized execution session; a new static PASS does not substitute for that evidence.
