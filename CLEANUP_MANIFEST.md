# Cleanup manifest - 2026-09-27

No unique data, manuscript, experiment, lock, secret, credential or historical receipt was deleted. All 3170 protected historical/model/lockbox/manifest files in the start snapshot remained byte-identical in the preservation check. `.serena/` existed untracked at start and was left untouched.

| Item | Action | Reason / evidence |
|---|---|---|
| Two instruction PDFs in Downloads | Copied into ignored data/source_documents; originals retained | Source recovery; SHA-256 recorded in protocol/RECOVERED_DOCUMENTS.json |
| Duplicate Trading Notes PDF with '(1)' suffix | Hash equality proved; no copy or deletion | Same SHA-256 as canonical recovered file |
| Earlier IPS Markdown/PDF and builder | Preserved and marked superseded; default script no longer emits the old IPS | Old font/extra section conflicted with recovered instructions |
| Historical matrix, orphan receipts and spent lockbox | Preserved unchanged | Integrity and provenance |
| Current top-level status/reproduction docs | Reconciled to current source recovery and package state | Git records prior text |
| .local_tmp | Ignored workspace scratch for tests, render images and isolated browser profile | Keeps generated scratch outside deliverables and off the internal disk |
| Private source PDFs | Ignored | Local reference, not a redistributed competition packet |

No cross-project deduplication, cache deletion or PRO-BLADE-wide cleanup was attempted.
