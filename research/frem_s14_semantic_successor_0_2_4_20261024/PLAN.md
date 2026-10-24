# FREM-S14 0.2.4 invocation-stable target candidate

Version 0.2.4 is the final pre-launch packaging correction for the frozen 0.2.x
semantic-reproducibility successor. It makes Ruff first-party classification
explicit and requires fail-fast validation before the immutable build. It does
not alter any scientific runtime behavior beyond the package version literal.

Acceptance requires:

- pre-build lint and format from the package root with `set -e`;
- all 59 dependency-free source tests;
- a new immutable wheel, archive, private GPU/Internet-off metadata and embedded
  notebook whose identities agree;
- extracted-archive lint/format and all 59 tests against the clean wheel;
- static source/payload/budget/notebook verification;
- a fixed-epoch second wheel build with identical SHA-256; and
- a fresh local resource admission receipt before any numerical import.

Any failed gate stops the candidate. Versions 0.2.0 through 0.2.3 remain unrun
and must not be uploaded. No historical metric may be interpreted until exact
target and separately controlled semantic verification both pass.
