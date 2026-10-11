# Legacy evidence archive

`wharton_lab_v0/` preserves the predecessor Wharton research package and its
negative, invalid, interrupted, and null results. The move from the repository
root is organizational only: the 3,373 predecessor files that existed at Git
HEAD `8e3d790` were verified byte-for-byte identical before this cleanup.

The additional `candidates/` records are retained because they contain later
engineering and scientific receipts, including the negative one-path
engram-reservoir result. Reproducible third-party dependencies under `.deps/`
remain on the local disk but are ignored by Git; they are not evidence.

The clean external checkout at
`wharton_lab_v0/Wharton-GYIC-Model-Stochastic-Engram-Reservoir-JEPA-for-Regime-Aware-Financial-Forecasting/`
is intentionally ignored as a nested repository. Its provenance is:

- remote: `https://github.com/build-the-future-11/Wharton-GYIC-Model-Stochastic-Engram-Reservoir-JEPA-for-Regime-Aware-Financial-Forecasting.git`
- branch: `main`
- commit: `4e11e7e`
- observed status during cleanup: clean and aligned with `origin/main`

Do not delete, rewrite, or reinterpret this archive to improve a later result.
The current implementations live at the repository root and under
`successors/`.
