# Kaggle live execution record — 16 October 2026

Current status: target setup and namespace probe passed; external saved-probe
retrieval and numerical smoke remain pending. This is not end-to-end completion.

The private notebook is `ryangomez010/wharton-s10-corrected-cpu-end-to-end-2026-10-16`.
Accelerator None and Internet On were observed before setup. The isolated runtime
actually installed Python 3.12.14 and NumPy 2.5.4; the kernel remained Python 3.13.15.

The first full-root probe rejected hidden cgroup ancestors. Its output is retained
in saved Version 1 (script version 358975473), the original console capture, and
`wrapper_before_namespace/`. No numerical work was performed by that probe.

The recorded decision `research/decisions/frem_s10_kaggle_namespace_20261016.json`
introduced a separate fixed 12-origin generated smoke contract. The production
contract and original source archive remain unchanged. The compatible namespace
probe returned SMOKE_ONLY_ADMITTED, with hidden ancestors UNKNOWN and production
admission false. Exact target metadata and digest are in `TARGET_RUNTIME.json`.
Thirty-seven local stdlib wrapper tests passed; the failed first test expectation
is retained in `namespace_tests_attempt1.log` and corrected verification in
`namespace_tests_attempt2.log`.

The namespace probe cell executed twice due delayed UI feedback. The second call
reused the immutable probe; wrapper invocation/setup charges were retained. No
model origins ran. Do not reset its ledger to reclaim the extra reservation.

Quick Save requests for the namespace outputs have not yet produced a verified
new version. The duplicate original source transport was replaced by a comment in
the draft after preserving Version 1 and local transport bytes. Editor refresh
then became stuck loading; Kaggle reported inability to connect to its reCAPTCHA
service. Chrome disconnected and was reconnected by the user. No retrieval
attestation was fabricated, and no model command was started without it.

Scientific status remains NOT_EVALUATED. This record concerns generated CPU
engineering, not production qualification, market results, GPU support or a
completed empirical study. Final platform stop is not yet verified.

## Final live verification — 16 October 2026

KAGGLE_GENERATED_CPU_END_TO_END_VERIFIED. Private Version 2 (358983684)
externally retrieved probe matched its producer SHA-256. The single numerical
campaign interrupted at origin 3, resumed to 12, reconstructed evaluation with
exact binary64 equality, validated completed reuse and exported successfully.
The model active time was 2.593 seconds; this excludes setup, admission, wrapper,
UI and persistence time. The wrapper reserved 593 of its 600 command seconds.

Private final Version 3 (358985139) saved successfully. Its terminal archive SHA-256
is `1b13386989ffc25384bbbcc56a0a9278b5b2c23ab0020b05ba36fd9258b63180`.
All 116 inventory hashes passed locally. The nested smoke export contains 48
verified artifacts plus generated evaluation and inventory files; its evaluation
matches the recorded stage. The separately retrieved finalization receipt reports
GENERATED_CPU_ENGINEERING_COMPLETE. Kaggle UI confirmed Off and Session stopped.

The save failure was diagnosed as a duplicate embedded payload exceeding Kaggle's
1 MB kernel source limit. Removing the duplicate payload from the draft resolved
it; Version 1 and all earlier evidence remain preserved. The source package was
unchanged. No GPU, market run, production qualification or protected evaluation
was performed. Scientific status remains NOT_EVALUATED. This evidence supersedes
the pending status above; it does not claim all possible bugs are absent.

Receipt: `research/frem_s10_kaggle_live_20261016/VERIFICATION_RECEIPT.json`.
Final URL: https://www.kaggle.com/code/ryangomez010/wharton-s10-corrected-cpu-end-to-end-2026-10-16/output?scriptVersionId=358985139
