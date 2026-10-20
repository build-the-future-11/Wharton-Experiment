# S13 implementation verification — 19 October 2026

S13 is implemented and packaged, with numerical execution **blocked** by the
unchanged 512 MiB resource admission rule. This is not RUN READY or a scientific
result. The proposed default matrix has 15 arms and three seeds (45 cells).

Passed: 13 standard-library tests from source and the same 13 from a clean
installed wheel; Ruff over all new package/test/builder Python; formatting;
15-file AST parsing; three notebook code-cell compilations; exact payload/member
hash readback; source-versus-wheel equality; two byte-identical wheel, ZIP and
notebook builds. The two 13-test executions overlap and are not 26 distinct tests.

The actual numerical pytest attempt exited 2 in the admission hook before test
collection. Available memory was 403.390625 / 278.59375 / 195.328125 MiB against
512 MiB. Pageouts were stable at 392546 on that attempt. Earlier attempts also
failed, including rising-pageout observations. No numerical, gradient, fitting,
CUDA, generated end-to-end, market, or protected check is claimed as passed.

The first clean-wheel installation attempt hit a pre-existing uv cache-path
error. Its subsequent import-test failure is retained. Installation succeeded
using an explicit writable temporary cache, and the second installed-wheel
standard-library run passed. This is packaging evidence only.

Artifacts and exact hashes: RECEIPT.json and package_verification.json.
Retained commands: checks.json and the corresponding logs. Earlier build bytes
are preserved under package_attempt1. No prior research source or result was
modified. No external upload, message, market download, Kaggle run or quota use
occurred. The next necessary evidence is the included CUDA test/qualification
notebook on a compatible admitted host.
