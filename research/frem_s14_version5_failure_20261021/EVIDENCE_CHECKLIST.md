

## 21 October 2026 — Version 5 evidence

- [x] Preserve exact Version 5 terminal failure: 107 collected, 105 passed, 2 failed, 0 skipped; qualification and historical cells did not start.
- [x] Reproduce both fixture failures on the exact 0.1.7 wheel; correct only the fixture's directory contract, retaining every assertion.
- [x] All 59 dependency-free tests pass with warnings as errors after correction; scoped static checks pass.
- [x] Verify local versioned source/wheel/archive/notebook and retained saved-code identity; keep every frozen 0.1.7 artifact.
- [ ] Download and integrity-verify Version 5 outputs; Chrome blocked the normal archive action, so visible-log evidence is not represented as downloaded bundle verification.
- [ ] Build a new immutable corrected test payload and pass all 107 tests plus 19 generated CUDA arms before the unchanged 171-cell historical suite.

See `research/frem_s14_version5_failure_20261021/FAILURE_RECEIPT.json`. No GPU passes transfer to the corrected payload and no scientific result is claimed.
