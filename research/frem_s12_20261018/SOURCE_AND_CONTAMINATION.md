# FREM-S12 source, rights and contamination contract

Date: 18 October 2026. No S12 archive was downloaded or parsed during this
review.

## Official source finding

The Kenneth French Data Library states that its US six-portfolio historical
archives are annual data cuts; the July cut contains data through July and is
released in August. Historical archives exist for the matched 2x3 families
formed on size and book-to-market, operating profitability and investment.

Authoritative pages:

- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/six_portfolios_archive.html
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/six_portfolios_me_op_archive.html
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/six_portfolios_me_inv_archive.html

P0 binds the August 2016 CSV archive for development and the August 2024 CSV
archive for common-cutoff revision stress. The exact URLs are stored in
`successors/frem_s12/src/frem_s12/sources.json`. Development may parse numeric
rows only through December 2015. The 2016 remainder and every post-cutoff 2024
row are prohibited from development.

## Families and units

- `size_bm`: six size/book-to-market portfolios.
- `size_op`: six size/operating-profitability portfolios.
- `size_inv`: six size/investment portfolios.

All are reconstructed research portfolios, share underlying US stocks and are
not independent markets. A forecast origin remains the statistical unit; family
and fold are clustering levels. Cross-family results cannot be multiplied into
eighteen independent asset histories.

## Prior exposure

S11's package and generated tests reference size/book-to-market semantics and
fixed calendar endpoints, but no S11 archive or numeric market row was acquired.
Earlier FREM studies inspected ETF and other public-period outcomes. Therefore:

- S12 P0 is development only, not independent confirmation.
- No 2016-remainder or later portfolio-family outcome is declared protected by
  this session.
- A future confirmation source remains `TBD` and requires a repository-wide
  date/series/metric exposure audit plus a separate freeze.

## Rights boundary

Public download availability is not treated as redistribution permission. Raw
archives stay local/private and are represented in reports by source URL,
timestamp, byte count and SHA-256. No raw archive enters a public release unless
permission is separately verified.

## Admission disposition

`SOURCE_CONTRACT_VERIFIED_RIGHTS_LIMITED_CONFIRMATION_UNRESOLVED`

This supports writing and testing the acquisition/parser contracts. It does not
itself admit data or authorize a scientific run.

