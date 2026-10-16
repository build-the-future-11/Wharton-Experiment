# Scoped prelaunch review — 16 October 2026

No unresolved launch blocker was identified for the **metadata/persistence probe only** in the exact artifact below. Numerical smoke still requires actual compatible runtime observations, retrieved persistence evidence, exact identity matching, fresh host admission and remaining time. This review does not grant Linux production qualification or scientific admission.

- Runner SHA-256: `dee5c6dfa9ff584c20e33f88842fbe986a43ef2e43a467924609f3cea177e1e1`.
- Notebook SHA-256: `91d1786c6aa87760669601e086387f7085be9c80b18d8e0ae77592776ca422d3`.
- Retained verification: `NOTEBOOK_TEST_RECEIPT.json`, PASS, 19 standard-library tests, return code 0. The reviewer independently recomputed all five listed file hashes; all matched the receipt. The receipt also records 12 sections, eight parsed code cells and verified embedded payload hashes.

The review read the runner, notebook builder, tests, maintained smoke CLI/contracts, usage instructions, report and retained UI observations. It did not import numerical libraries, execute the model, open the browser or perform cloud calls. Tests were run by the implementation agent; this reviewer inspected their retained receipt and relevant fixtures. This is a separate-agent review within the same session and host, **not external independent replication**.

## Findings and repair readback

| Finding during review | Repair observed in the reviewed artifact |
|---|---|
| Mutable source inventory allowed modified source to retain the claimed sealed identity. | `verify_source` pins the inventory bytes as well as the candidate document, then compares the entire extracted inventory. A source-plus-inventory tampering fixture is retained. |
| Exit codes alone could count an interruption before any checkpoint as the intended resume exercise. | Stage validation requires exactly three committed origins for interruption, twelve for completion/evaluation, validated completed reuse and generated-only export scope. |
| Selector loop overwrote the command UUID, breaking start/end pairing. | Separate selector variable; a real tiny stdlib child tests streaming and matching command receipts. |
| Setup, snapshots and restored probe overhead were outside the original time accounting. | Bootstrap markers, inherited wall reservation, source/setup and export allowances, cumulative wrapper wall watchdog and absolute cutoff are recorded. Nominal child-command reservations are 593 seconds under the 600-second cap. Repeated setup is charged. |
| Outer termination could orphan a child in its separate process group. | SIGTERM raises a controlled exception through child-group cleanup. A real stdlib process fixture checks termination; the outer deadline margin is ten seconds for its bounded cleanup waits. |
| Local stage completion could be reported before final full-state export succeeded. | The engineering receipt says final export is pending. Only successful export followed by a deadline check produces the finalization receipt. A failed-final-export fixture verifies this boundary. |
| Retrieval attestation only recomputed the supplied file's own hash. | The producer displays its probe digest; `attest` requires an independently supplied expected digest and compares actual downloaded canonical bytes. Version provenance remains an explicit operator attestation, not authenticated provider evidence. |
| Incompatible Python could fail before preserving a useful probe. | A stdlib compatibility gate precedes maintained package import. Unsupported Python retains `NOT_RUN_PYTHON_INCOMPATIBLE`; the sealed manifest is read directly. A blocked probe can prove retrieved persistence but cannot authorize numerical work. |
| Failed/abandoned commands or incomplete wrapper history could be retried without reliable cost evidence. | Retained command and wrapper records block automatic retry/continuation. Restored state does not reset charges; archive-only inventory metadata is excluded from subsequent state exports. |

## Save route and remaining limits

The notebook/usage route agrees with `observations/SAVE_OPTIONS.json`: one interactive CPU session, accelerator None, then **Quick Save → Save output for this version**. The observed default Never save output is insufficient. Save & Run All is explicitly identified as a fresh execution, not the chosen save operation. Retrieval must identify the exact private version, compare the downloaded bytes with the producer digest and preserve the source/runtime binding. The deadline is fixed at **2026-10-16 16:30 UTC / 22:00 IST**.

These are local engineering checks. Actual Kaggle interpreter/dependency compatibility, Linux host admission, remote output persistence, model execution and platform-stop observations must be established separately and retained by the campaign operator. Current UI availability and subsequent cloud observations may change after this prelaunch review; earlier blocked observations must remain historical evidence. No test here establishes provider billing finality, account-wide exclusion, external scientific reproduction, financial efficacy or publication readiness. Explicit platform Stop Session and account/session inspection remain required; process exit is not platform termination evidence.
