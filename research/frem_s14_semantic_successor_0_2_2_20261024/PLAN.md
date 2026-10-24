# FREM-S14 0.2.2 clean target candidate

Version 0.2.2 is the exact 0.2.1 scientific runtime plus release-scope import
ordering and formatting corrections in packaged tests. The module version is
advanced with wheel metadata so every new artifact has a distinct identity.

The 0.2.0 and 0.2.1 releases remain immutable and unrun. No target numerical
evidence transfers. Before upload, 0.2.2 must pass:

- all 59 dependency-free tests from source and a clean installed wheel;
- release-scope Ruff lint and formatting;
- wheel/source, payload, configuration/runtime budget, notebook and embedded
  release verification;
- a second deterministic wheel build with identical bytes; and
- the unchanged local 512 MiB/stable-pageout admission check.

If the local gate fails, do not import numerical dependencies. The exact private
Kaggle target remains the only authorized numerical path. Historical metrics may
be interpreted only after target and separately controlled semantic verification
both pass.
