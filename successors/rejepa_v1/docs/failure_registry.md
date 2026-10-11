# Failure and negative-evidence registry

## Inherited scientific evidence

- The legacy 1,181-cell matrix is leakage/degeneracy contaminated.
- The spent legacy synthetic lockbox arm is invalid and the ETF arm is negative.
- The repaired legacy pipeline is exploratory null versus historical mean.
- A later one-path synthetic memory result was worse than memory-off.

These findings are not rerun, repaired away or claimed to be solved by v0.2.

## Current engineering failures retained

- An earlier expanded-suite invocation omitted `PYTHONPATH`; collection failed
  before tests ran. The documented invocation passed.
- One combined lint/test/build command was interrupted without a receipt; each
  check was rerun independently.
- During v0.2 expansion, the first run passed 46 tests and failed one assertion
  because the new horizon-conditioned no-world control intentionally differs
  from raw context. The contract assertion was corrected; no model behavior was
  weakened to force a pass.
- The first v0.2 report-generation command failed because the requested
  `reports/` parent did not exist. The writer was corrected to create parent
  directories, a nested-path regression test was added, and the command was
  rerun.
- The first two v0.3 engineering receipts used the declared seed for training
  and latent worlds but constructed the model before seeding. Their model hashes
  differed. Both receipts are retained; CLI seeding now occurs before model
  construction.
- Two subsequent seeded runs had identical losses, predictions and manifests but
  different raw `torch.save` archive hashes. PyTorch archives are not treated as
  canonical tensor identities. v0.3 now records and verifies a deterministic
  hash over sorted state names, dtypes, shapes and raw tensor bytes; repeated
  fresh-directory runs must match that state hash.

Future OOM, timeout, non-finite, missing-cell, baseline, provenance and protected
boundary failures must be appended and preserved.
