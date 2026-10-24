# FREM-S14 0.2.4 adversarial release and execution checklist

| Flaw | Evidence / cause | Severity | Resolution | Acceptance evidence |
| --- | --- | --- | --- | --- |
| 0.2.0 wheel and module identities disagreed | Wheel metadata said 0.2.0; `frem_s14.__version__` and its test said 0.1.9 | Critical release identity defect | Preserve 0.2.0 unrun; advance immutable version and require equality | 0.2.4 static verifier checks wheel metadata and module literal |
| Packaged protocol contradicted runtime fold budget | 0.2.0 README/spec/config said 2,800 seconds; frozen successor runtime used 4,000 | High reproducibility defect | Align packaged config and prose without changing runtime or the 13,500-second outer cap | Extracted 0.2.4 config equals installed `suite_config()` at 4,000 seconds |
| Release lint was invocation-directory dependent | `frem_s14` first-party classification was implicit | Medium engineering defect | Declare first-party package explicitly and run fail-fast from the package root | Checkout and extracted payload Ruff lint/format pass |
| Intermediate builds did not satisfy their intended gate | 0.2.1 through 0.2.3 exposed style, archive-timing and fail-fast defects | High provenance risk if overwritten | Preserve every version under unique hashes and never upload it | 0.2.4 receipt lists all superseded unrun releases |
| Local numerical verification is unsafe now | 330.23/358.36/331.69 MiB available against 512 MiB; pageouts rose each sample | Critical execution blocker | Stop before NumPy/Torch/XGBoost/SciPy imports | Fresh passing `LOCAL_ADMISSION.json` required |
| Exact private target is inaccessible | Browser is at an ambiguous profile chooser; existing owner `ryangomez010` is not authenticated unambiguously | Critical external blocker | User opens the already-authenticated owner profile; do not guess or create an account | Visible owner identity before upload and saved-version readback after upload |
| CUDA/numerical qualification is absent for 0.2.4 | No target run exists | Critical evidence gap | Run warning-strict target tests and all 19 generated arms exactly once | Complete target qualification plus independent semantic verifier |
| Historical proxy evidence is absent | 0/171 valid 0.2.4 historical cells | Critical scientific gap | Run fixed suite only after qualification and retain failures | Complete suite, target verifier and separate verifier all pass before interpretation |
| Data cannot establish tradable or point-in-time performance | Adjusted-close survivor ETF proxy lacks PIT adjustment vintages, spreads, fills, impact and terminal instruments | High claim limitation | Keep results exploratory; do not promote to protected or investment claims | A separately admitted point-in-time study would be required |
| Conference readiness is not established | No valid S14 result, completed results paper, current venue-rule audit or independent replication | Critical publication blocker | Finish evidence first, then update manuscript and venue package from verified artifacts | Compiled/inspected paper, claim map, venue checklist and reproducibility receipt |

No threshold, cohort, arm, seed, fold, endpoint or protected boundary was changed
in response to an observed scientific outcome. Versions 0.2.0 through 0.2.3
were never uploaded or numerically executed.
