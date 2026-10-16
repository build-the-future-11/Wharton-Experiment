# Runtime implementation integration report — revision2, 2026-10-16

Local production integration is implemented; actual research execution remains unadmitted. No market fit, market ledger, protected evaluation, or full-scale generated qualification was executed by this agent.

The owned production changes are `runtime/{pipeline,worker,results,receipts,declarations,consumption,supervisor,continuation,saved_audit}.py`, `cli.py`, and three dedicated runtime test files. Their exact hashes appear below and in `IMPLEMENTATION_INTEGRATION_RECEIPT_V2.json`. The operation-specific source/environment/verification/qualification producers are owned by the parent and other agents; this receipt does not replace their audits or a final complete-candidate verification.

## Implemented behavior

- Exact source/configuration/environment/verification admission, prospective freezes, fixed guarded worker commands, and actual persistent consumption.
- Pre-2010 precision archive-to-target/forecast/score linkage, diagnostic/final-state checks, saved bootstrap regeneration, and guarded prerequisite re-audit before campaign values.
- Fixed192-cell persistence and saved-output primary/ablation/report regeneration inside supervised campaign and qualification paths.
- Independent saved-audit operation with its own consumption namespace; full saved decoding is no longer invoked outside supervision by the CLI.
- Fixed conditional advancement: verification → precision → qualification → campaign. Current catalog evidence is required. Failed precision, consumed failures, ambiguous interruptions, and valid terminal campaign outcomes stop automatic advancement. Proven unconsumed resource refusals can use fresh attempt directories while retaining their original declaration.
- Exact sampled RSS peaks, positive RSS, integer pageouts, and ordered lifecycle/consumption-hash verification. Finalization failure after consumption is retained as `FAILED_CONSUMED`.
- Module CLI guard, explicit execution switches, safe nested freeze paths, fixed fresh bytecode namespace, immutable outputs and a locked hash-linked continuation journal.

## Actual verification evidence

- `runtime_metadata_tests_attempt3.log`: **46 passed, 1 deselected, 5.36s**. These are bounded metadata, mocked dispatch, lifecycle and fault tests. The numerical generated precision test was deliberately not repeated during low-reserve final integration.
- `continuation_tests_attempt2.log`: **15 passed, 0.71s**, including the subsequently added signed-journal stage-skip regression. This overlaps the preceding suite and must not be added to it as independent test coverage.
- `runtime_final_mypy.log`: **13 source/test files pass strict mypy**.
- `runtime_final_lint.log` and `runtime_final_format.log`: **PASS**, including all13 owned files.
- `runtime_tests_attempt5.log`: earlier in-session **26 passed, 444.61s**, including real generated precision kernel/archive/scoring and saved-law regeneration. The source-loader boundary is injected with an independently validated generated source view. This is not a market precision result or final full-candidate verification.

## Preserved failures and limitations

- `runtime_existing_tests_attempt1.log` retains **19 failures and27 passes** from a sandbox where `/bin/ps` accounting was denied. Those failures were not changed into successful guard receipts. Final integration requires the parent's authorized process-accounting context.
- `runtime_metadata_tests_attempt2.log` retains two failed regression assertions for nonpositive RSS because the guard raised `ContractError` rather than the expected `IntegrityError`. Both values were already rejected; the guard now consistently rejects them as integrity failures, and the corrected suite passes. All earlier lint/type failures remain retained as well.
- Full-scale memory/runtime/output behavior remains unmeasured until actual guarded qualification. `load_campaign` retains typed results for all192 cells; a full-size memory excess must fail the unchanged ceiling rather than be relabeled as success.
- The engineering verifier observes cumulative child peak RSS after command completion and enforces command timeouts. It does not provide live process-group memory enforcement. Scientific, qualification, and saved-audit workers use the live supervisor. Both types of observation are software measurements, not a hard kernel resource guarantee.
- Generated, retrospective, and protected evidence remain distinct. No claim of perfection, positive empirical results, point-in-time data, independent scientific reproduction, or completed research is made here.
- Required final work remains: exact-candidate full verification/catalog/build evidence, fresh passing resource admission, consumed pilot/qualification/campaign execution if admitted, evidence audit, canonical-state reconciliation, and the research terminal disposition. Protected confirmation is outside this development continuation.

Detailed manifest schemas and execution boundaries are in `RUNTIME_INTEGRATION.md`. Package source is frozen by this agent after this report; subsequent changes require a new exact-candidate receipt.

## Post-report integration correction

The independent packaging review found a real interface mismatch: complete provenance now binds `README.md` and `MANIFEST.in`, while the declaration validator still rejected those extensions. The explicit supported-name allowlist now includes both. The focused regression feeds a complete generated provenance inventory through the actual declaration code validator, confirms all source/test/build names are accepted, and proves that changing `MANIFEST.in` still invalidates its retained hash. All `.py` test paths remain supported; no other extension restrictions conflict with the current provenance inventory.

`freeze_inventory_regression_attempt1.log`: **1 passed, 24 deselected**. Scoped strict mypy, lint and format checks also pass. This was a metadata-only repair, with no numerical or scientific execution and no research-method change. The first report/receipt remains unchanged; this revision supersedes only its exact final owned-source identities. No package mutation is planned after this revision.

## Exact owned file hashes

| File | SHA256 |
|---|---|
| `successors/frem_s10/src/frem_s10/runtime/pipeline.py` | `2daa130ca73a82a3874fa6a3c973be490af9359ed17f77210a7e247020c3cc8b` |
| `successors/frem_s10/src/frem_s10/runtime/receipts.py` | `69780b330121a43b6269070a4b51f8cb08ea8c4adde8f2ee695c0a5aa09f1ba5` |
| `successors/frem_s10/src/frem_s10/runtime/saved_audit.py` | `bed576ba293d167bd00c151cf06b085a85935023088f7eca9458861a43775819` |
| `successors/frem_s10/src/frem_s10/runtime/continuation.py` | `feed6d06798ba460cfd7e5cfa7dc6e585f094602c3428da843dee685c3905538` |
| `successors/frem_s10/src/frem_s10/runtime/supervisor.py` | `7502613f1ed38696cffb01195300cd49736a4edd380f703b0d9932a64ef8059f` |
| `successors/frem_s10/src/frem_s10/runtime/consumption.py` | `ddaf2c68674d7c6805c52a931af9d3600b7e272e223a2aa32e6b8f12fade69af` |
| `successors/frem_s10/src/frem_s10/runtime/results.py` | `f0ec483a426c8a8bc6be0e8aa3ac54ad44f4b4f583e8dc36abbe82b01bc3d309` |
| `successors/frem_s10/src/frem_s10/runtime/worker.py` | `9ca5d243b8ab22365f4a77668edd6cea59934587414fa67db07b9d042804b3ce` |
| `successors/frem_s10/src/frem_s10/runtime/declarations.py` | `ff749dc5b38b01e440217a29f457bea283d6d02e5368c2cb6950cf1bac6ca786` |
| `successors/frem_s10/src/frem_s10/cli.py` | `b937fbb1c14847007edd7c663ae4b2acb10fff558f5adeec5fa92593c725ffd5` |
| `successors/frem_s10/tests/test_production_runtime.py` | `7d7989aff4d57d256893225c1e65bb55d2598465db6c20bfe924ff66155f820a` |
| `successors/frem_s10/tests/test_continuation_audit.py` | `00a2fe7f7db2b264837bb0d8275227ba25ddd742f52397dbfd41b3d292081118` |
| `successors/frem_s10/tests/test_saved_audit_runtime.py` | `376c7a3e7c162d42c2e5cfe4df33d31b322e85edf151b48f0e16fee2cc98fad0` |

Inventory SHA256: `3c7e6692623083e3bb1c1929bae1450686d49ce8adddbfd65536d7e950a874e3`.
