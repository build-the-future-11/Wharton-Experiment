# FREM-S13: reservoir–engram JEPA development system

This is a separate implementation of a new development hypothesis. It preserves
all S12 and earlier source, protocols and outcomes. **Code is implemented; numerical
and CUDA qualification have not yet passed.** Do not describe it as flawless,
scientifically validated, or ready for an expensive market campaign until the
included qualification succeeds on the target runtime.

The model retains a multiscale contracting reservoir, causal Fourier features,
discrete trailing ngrams, bounded engram retrieval, an EMA target JEPA objective,
training-only neurogenesis and separate return/risk routing. It adds Student-t
mixture uncertainty, variance/covariance representation regularization, strong
baselines, a validation-selected XGBoost distribution blend, purged chronological
evaluation and independently reconstructed financial ledgers.

See [CHECKLIST.md](CHECKLIST.md) for implementation and acceptance evidence,
[SPECIFICATION.md](SPECIFICATION.md) for mathematics and limitations, and
[DATA_CONTRACT.md](DATA_CONTRACT.md) for real input requirements.

## Installation and local commands

Use Python 3.11–3.13 on Linux or macOS. CUDA requires a compatible installed
PyTorch, XGBoost and NVIDIA driver. Device fallback is rejected. This is a single
GPU implementation; an extra Kaggle GPU is not automatically used.

```bash
python -m pip install ./frem_s13-0.1.0-py3-none-any.whl
frem-s13 plan --config configs/development.json
frem-s13 preflight --output host-admission.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -W error -q
frem-s13 qualify --device cuda --output qualification-001
frem-s13 verify --bundle qualification-001
frem-s13 run --config configs/development.json --data panel.csv \
  --qualification qualification-001 --output development-001
frem-s13 verify --bundle development-001
```

Each output directory must be new. `qualify` executes all 15 registered arms on
generated data, including XGBoost, with two small neural epochs. It does not
train on a market dataset. A complete qualification from the exact source,
configuration, dependency versions and operating runtime is required by `run`.
No source/import success alone admits training. Admission requires at least
512 MiB available and no rising pageouts in three samples. OOM, nonfinite
gradient, missing baseline, numerical disagreement and budget exhaustion stop
the campaign and retain the failure with its completed denominator.

The default development configuration contains 45 cells: 15 arms × 3 seeds,
at most 20 epochs per neural cell, 300 XGBoost rounds, and a four-hour total
wall budget. These are ceilings, not a measured cost promise. The user must
choose the appropriate declared data contract, horizon and untouched boundary
before a real campaign. Do not infer permission to run all cells from this file.

## Kaggle

The release builder creates `dist/frem_s13_kaggle_bundle.zip` and
`kaggle/FREM_S13_GPU.ipynb`. Attach the exact ZIP to a private Kaggle input dataset,
open the notebook, select a compatible GPU, and run its default qualification.
The notebook verifies the ZIP digest and every payload member, installs the
wheel into an isolated directory without changing Kaggle's base environment,
checks dependency requirements, runs the complete test suite and then the
generated matrix. It fails if dependencies are absent or incompatible; it does
not install arbitrary latest versions or silently move to CPU.

For development, attach exactly one `panel.csv` plus
`panel.csv.manifest.json` matching DATA_CONTRACT.md, then set the notebook's
`MODE` to `development`. The notebook runs qualification first, locates the
declared data, then executes the bundled `configs/development.json`. Edit that
config in the source and rebuild the bundle if feature count, horizon or
protected boundary differs. Retain all notebook outputs and use `verify` after
downloading them. No public upload, external transmission, Kaggle run, data
download or scientific result was performed by authoring this bundle.

## Recovery and inference

Immutable epoch checkpoints include online/target weights, reservoir mask,
engram bank, optimizer, scheduler, AMP scaler, RNG state, best validation state,
preprocessing, source/data/config/runtime identity and training history.
`--resume-checkpoint` supports a run originally configured with one neural arm
and one seed, into a fresh output directory after a retained failure. It resumes
after that completed epoch and subtracts the prior failed attempt's wall time
from the unchanged campaign budget. Completed runs cannot be reopened. A
partially completed epoch is replayed from its prior checkpoint; its failed
output is never deleted. Cross-runtime and changed-config resume are rejected.

The `S13.forward(context, decision_times, asset_ids)` API consumes only trailing
context. `predict` batches the held-out slice; `load_checkpoint` requires a
matching binding and checksum and uses restricted tensor deserialization.
Use `best_state` for inference, not the last optimization state. Restoring a
checkpoint is only appropriate for trusted, hash-verified artifacts.

## Evidence limits

This implementation is an adaptation of established mechanisms, not a novel
theorem or an established trading advantage. Historical rights, universe
completeness, terminal outcomes and feature availability are supplied assertions
that require independent source audits. The balanced-panel adapter intentionally
refuses missing rows. The system does not certify global equities, survivor-free
coverage, exchange calendars, broker fills, borrowing, taxes, FX, a competition
mandate or calibrated recession probabilities. Protected confirmation is not
exposed by the CLI and requires a separate prospective protocol.
