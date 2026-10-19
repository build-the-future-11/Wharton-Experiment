# FREM-S11 P1 engineering verification

Date: 18 October 2026. Candidate source identity:
`20b462e9eed989362dc34198e4325e5e6dd678357eb366055f642016ea1a9fca`.

## Passing checks

- `28 passed in 18.80s` with warnings treated as errors, one BLAS thread and
  the complete `successors/frem_s11/tests` collection.
- Ruff strict lint passed; Ruff format check passed for all 27 checked files.
- Strict mypy passed for all 13 source modules under Python 3.12.
- Source distribution and wheel built with no isolation using setuptools
  84.0.0 and wheel 0.48.0.
- Clean no-dependency wheel installation passed. The installed console command
  displayed all five fixed subcommands, embedded `protocol_v2.json` was
  present, and importing the CLI did not import NumPy.
- Git whitespace validation passed.

Build artifact identities:

- wheel: `92089ee5ff4d3aac520cc41d958fb063c33c27ff72097376b94ce38481b47e52`
- sdist: `bb92efa059f65d112620ee495d0f72ba2ca0fde0c159a39d7c21323496eb1b40`

The build products were kept outside the repository under `/tmp`; generated
`build/` and egg-info directories created inside the package were removed.

## Retained failures and repairs

- Initial generated verification found formatting failures in 14 files.
- Strict typing found two NumPy return-type leaks and an invariant-list
  annotation issue.
- A revision test attempted to construct an invalid gapped calendar instead of
  reaching the intended axis mismatch.
- Exact floating-point equality assertions failed for a value-state and a
  constant block interval.
- The initial policy test encoded the pre-P1 fee convention. P1 now explicitly
  applies the fixed 10 bps one-way turnover cost to investable wealth before
  return realization; the hand calculation was updated before data access.
- One complete verification attempt was interrupted after abnormal host-pressure
  delay; its partial output showed failures and was followed by isolated repairs
  and the final complete 28-pass run. This was engineering only.

No expected scientific values were changed, no market data was used, and no
failure or predecessor result was removed.

## Host admission

Two actual source-bound receipts are retained. The final exact-candidate receipt
is `host_admission_attempt2.json`, SHA-256
`21e99e3087d553cebb0e362c223c037aa186d7b52d846896b7b1a6f3f4b8f364`.
It is `BLOCKED`: available memory was 179.906, 161.516 and 219.125 MiB,
below the fixed 512 MiB floor, while pageouts increased from 2,061,218 to
2,061,237. Therefore acquisition, parsing and scientific execution were not
attempted.

## Evidence boundary

This receipt establishes engineering verification of P1 only. Scientific status
remains `NOT_EVALUATED`; market fits, market score rows, market policy ledgers,
protected accesses and confirmation runs are all zero.
