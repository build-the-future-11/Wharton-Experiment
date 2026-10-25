# Implemented changes and decisions

Starting revision: `16ab217f` on `codex/frem-research-execution-20261020`.
Frozen S14 source identity remains `cba3c5c4bb9d0f4ab3ae12d85b465baa13d100c8b36fe15c2c91331844c706e2`.

- F01: corrected manuscript status, current cohort and fold ceiling; preserved
  predecessor facts and invalidity; shortened abstract after visual review found
  two orphaned lines on a separate page. Added current primary-source novelty
  constraints. No empirical values invented.
- F06: `scripts/pack_results.py` streams one stopped result tree into an exclusive
  ZIP, verifies inventory and byte equality, refuses overwrite/symlinks, retains
  failure receipts. O(B) byte work, O(number of files) metadata, 1 MiB stream
  chunks; no claimed measured speedup. Eight regression tests.
- F07: `scripts/validate_kaggle_package.py` verifies immutable identities,
  embedded payload, private/GPU/offline target metadata and dataset allowlist.
  Fourteen total new tooling tests including six package-tamper tests.
- `kaggle/`: exact unchanged notebook/archive/metadata, input/experiment manifests,
  requirements and concrete launch/return/recovery instructions. Frozen model
  code and run entrypoints are reused, not silently replaced.
- Canonical state, handoff, verification notes and hash-chained research journal
  record this session and its failed CPU-admission attempt.

Selected improvements: evidence transport, wrong-version rejection, paper
truthfulness and novelty qualification. Deferred: whole-suite resume, a stronger
retrieval novelty comparison, broad OOD/scaling and S15 models. These require the
recorded dependency gates and new versioning. Rejected: widening tolerance,
dropping failed seeds, calendar imputation, CPU substitution, unbounded tuning
or declaring a positive finding from unit tests.

Rollback: new external tools/handoff can be reverted independently. Original
paper and canonical-state snapshots remain under the session's `before/` folder.
Do not revert by deleting retained failures or editing frozen 0.2.4 artifacts.
The initial packer style check failed (SIM117/format), then was corrected; all
new tests and final scoped lint pass. No deleted historical data or checkpoints.
