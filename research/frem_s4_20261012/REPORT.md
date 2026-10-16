# FREM-S4 development result — 12 October 2026

Completed: 6 recent-base fits, 24 prediction cells and all seeds/windows. 247 tests passed before fits. Wall time 64.76 seconds; maximum sampled worker RSS 401.84 MiB under 768 MiB; one thread, nice10. Normal memory admission was 76.91 MiB against 512 MiB; the user's renewed slow-mode exception is retained in the run.

| Development window | Memory-off NLL | Original-distance NLL | Euclidean-distance NLL | Random NLL |
|---|---:|---:|---:|---:|
| fresh_2023 | -1.332779 | -1.332779 | -1.332779 | -1.332779 |
| fresh_2024 | -1.411278 | -1.411278 | -1.411278 | -1.411278 |

All 18 non-off seed/window/arm selections chose zero correction. The predeclared gate required more than 0.01 nat validation improvement in each of two chronological halves. Euclidean nonzero candidates were harmful in the first 2023 validation half; 2024 gains were too small or inconsistent to pass. The rule was not weakened after this result.

This establishes conservative rejection of the proposed mean correction under this protocol. It does not establish that distance scaling intrinsically has no value: with the final gate at zero, the deployed predictions cannot discriminate retrieval rules. Recent fitting, mean-only correction and conservative gating are shared conditions; comparison to S3 cannot isolate their individual causal effects. The 2023/2024 outcomes were previously inspected and reused explicitly as development, so this is not fresh generalization or confirmation. No current-return prediction or investment profitability follows.

Fresh standard-library audit verified 3,168 paired prediction rows, all 81 frozen source/data identities, selection eligibility and complete arms/seeds. Maximum Student-t NLL reconstruction difference was 1.9984e-15. This is not independent retraining. Five new tests cover distance conversion, recency/purge chronology, null rejection, future-label invariance and serialization.

Two pre-freeze Ruff style findings remain in the frozen runner/test source (unused months binding and adjacent-pair zip style); runtime tests pass. Do not change frozen scientific source in place to hide these cosmetic findings. The original lint output is described here; no science claim depends on style cleanliness.

Artifacts: outputs/frem_s4_20261012_attempt1/freeze.json, completion.json, resource_admission.json, tests/console.log, per-cell checkpoints/predictions/selection/state, research/frem_s4_20261012/audit.json. Retain this terminal development result without outcome-driven retries. Canonical panel, protected confirmation, independent reproduction and approved Laura allocation remain unresolved.
