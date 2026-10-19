# FREM-S12 Kaggle execution harness

`frem_s12_campaign.ipynb` is a thin, auditable launcher around the packaged
`s12-research` CLI. It does not contain a second implementation.

Attach a read-only Kaggle dataset containing:

- `frem_s12-0.1.4-py3-none-any.whl`, verified by the notebook's frozen SHA-256;
- Kaggle's preinstalled `numpy==2.1.3` on Python 3.13, or `numpy==2.5.4` on
  Python 3.12; no dependency download is allowed in the scientific notebook;
- the exact raw archive and its acquisition/data-admission receipts for a
  scientific cell;
- the environment receipt produced by a successful qualification run.
- the complete 144-cell search-selection receipt for fixed-matrix cells.

Set the configuration cell to one of `qualify-kaggle`, `generated-smoke`,
`run-search-cell`, `select-search`, `run-cell`, or `analyze`. Scientific search
and fixed-cell modes require `EXECUTE = True` and every receipt argument.
Results are written only below `/kaggle/working`; existing paths fail closed.
Download the complete output directory before ending the session.

P0v5 installs into a fresh isolated target rather than modifying Kaggle's base
environment, fixes numerical libraries to one reference thread, preserves the
installer transcript, and writes a final no-overwrite manifest containing the
SHA-256 and byte count of every retained output. Qualification runs the largest
declared search configuration and latest fixed fold three times and projects
from the slowest observation; artifact byte counts must be stable across all
repeats.

P0v6 additionally makes every packaged target command warning-strict, rejects
output symlinks and special files, and requires independent local verification
of a downloaded bundle with `s12-research verify-run-bundle`.

P0v7 also requires the original acquisition receipt for every scientific cell,
reconstructs validation-selected controls and exact calendars from retained raw
evidence, rejects foreign campaign files, and distinguishes bundle integrity
from actual next-stage eligibility. For qualification intake, pass
`--expected-mode qualify-kaggle`; proceed only when
`eligible_for_next_stage` is true.

Keep every saved-version `consumption/` directory with the campaign. A first
attempt claims its cell even if execution later fails. Do not delete the claim
or rerun elsewhere; only the exact checkpoint-resume path may reuse it. Kaggle
does not provide a cross-version atomic filesystem, so the final audit must
reconcile every saved version and reject duplicate/forked claims; a local claim
directory alone is not proof that no hidden attempt occurred.

P0v3 saved version `359671420` failed before package import on the unsupported
runtime contract. Engineering-only capability saved version `359681231`
subsequently observed Python 3.13.15 / NumPy 2.1.3, recursive read-only inputs
and blocked egress. The exact P0v4 qualification notebook has not yet completed;
a local runtime receipt or capability probe cannot admit science.
