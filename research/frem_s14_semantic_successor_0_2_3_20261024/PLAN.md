# FREM-S14 0.2.3 verified-archive target candidate

Version 0.2.3 contains the same frozen 0.2.x scientific runtime and only the
pre-launch identity, documentation/configuration, and test-source quality
repairs described in the recorded decisions. Versions 0.2.0 through 0.2.2 are
preserved and were never uploaded or executed.

Acceptance requires checks against the immutable archive itself:

- release-scope Ruff lint and format pass for both checkout and extracted
  payload test/source bytes;
- all 59 dependency-free tests pass from source and from the extracted archive
  with the clean installed wheel first on `PYTHONPATH`;
- wheel metadata, module version, source identity, fold budget, payload hashes,
  private GPU/Internet-off metadata, notebook syntax and embedded release agree;
- a second fixed-epoch wheel build is byte-identical; and
- local numerical work remains behind the 512 MiB/stable-pageout gate.

Only after those checks may this exact release be uploaded to the existing
private Kaggle owner. Target numerical/CUDA and historical evidence remain
entirely unrun, and no protected outcomes are in scope.
