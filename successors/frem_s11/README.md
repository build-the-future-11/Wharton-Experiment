# FREM S11

S11 is a separate successor study. It does not revise or reopen S8, S9, S10,
or the original FREM-P0 hypothesis.

The study tests whether a causal mixture of four finite-scenario specialists
improves joint monthly return forecasts for the six Fama/French size and
book-to-market research portfolios. The primary evidence is predictive energy
score. Portfolio utility is secondary and cannot rescue a failed forecast
gate.

`protocol.json` preserves the initial P0 freeze. The pre-parse validity audit
found reversed LoBM/HiBM semantic labels, inaccurate rolling-law terminology,
stale target-weight turnover accounting, and under-bound CLI arguments. No
archive had been acquired or parsed. Corrected `protocol_v2.json` is therefore
the operative P1 freeze; P0 remains evidence and must not be overwritten.

Three release vintages are declared before acquisition:

- August 2008: revision reference through July 2008.
- August 2016: development source through July 2016.
- August 2024: revision stress only through July 2016. Later rows remain
  outside development and may not be inspected by the development command.

The official historical index supplies the 2008 vintage as TXT and the 2016
and 2024 vintages as CSV. `sources.json` freezes those exact official URLs.
The original P0 contract is retained in `protocol.json`; only corrected P1 in
`protocol_v2.json` may execute.

These are research portfolio returns reconstructed from CRSP inputs. They
improve universe breadth and permit explicit revision analysis, but they are
not an individual-security point-in-time panel, a redistribution grant, or a
deployable trading feed.

## Fixed sequence

1. Verify a supported Python 3.12 host under the existing 512 MiB and stable
   pressure-counter rule.
2. Acquire the exact declared archives without overwrite and retain URL,
   timestamp, byte count, and SHA-256 receipts.
3. Run the normal generated tests and static checks.
4. Parse development rows only and produce a data-admission receipt.
5. Run the two fixed development folds once. Preserve all specialist,
   incumbent, policy, and uncertainty results.
6. Run revision stress on common historical cutoffs. Do not inspect post-July
   2016 rows from the 2024 archive.
7. Stop after a valid negative or inconclusive result. A later confirmation
   needs its own freeze and genuinely unused outcomes.

The CLI enforces this sequence with `admit-host`, `acquire`, `admit-data`,
`development`, and `revision-stress`. Development requires separate host,
archive, data-admission, and one-shot consumption receipts. It writes every
month, score, stack weight, realized allocation, wealth path, benchmark path,
summary, provenance identity, and file hash without overwrite. A failed run
retains its attempt and failure receipts.

The latest local host receipt is
`research/frem_s11_20261018/host_admission_attempt2.json`. It is `BLOCKED`:
available memory was 179.91, 161.52, and 219.13 MiB, below the fixed 512 MiB
floor, while pageouts rose from 2,061,218 to 2,061,237. No data was
acquired, parsed, fitted, or scored.

The implementation cannot guarantee improved results. The fixed success rule
requires the stacked law to clear equal mixing and the recency specialist in
both folds with uncertainty and a 5 percent relative energy-score margin.

## Current verified status

Generated verification covers source/archive receipts, CSV and TXT parsing,
calendar continuity, specialist causality, score reconstruction, policy drift
and costs, revision alignment, raw artifact hashing, and future invariance.
The first actual host admission is retained at
`research/frem_s11_20261018/host_admission_attempt1.json`: it is blocked because
all three samples were below 512 MiB and pageouts rose. Consequently, zero
archives have been acquired, zero market rows parsed, and zero scientific
folds executed.

The latest exact-candidate retry is `host_admission_attempt2.json`; it is also
blocked and supersedes attempt 1 only as the current admission observation.

## Exact execution sequence

Run from the repository root with a supported Python 3.12 interpreter and the
package installed, or set `PYTHONPATH=successors/frem_s11/src`.

```bash
python -B -m frem_s11.cli admit-host \
  --output research/frem_s11_20261018/host_admission_attempt2.json --execute
```

Only after that receipt says `PASS`, acquire the three exact URLs in
`sources.json`; each command also requires `--admission` and writes an adjacent
`.receipt.json`. Then run `admit-data` for release `201608`, cutoff `201607`.
The one permitted development command requires that data receipt, a fresh host
receipt, a new output directory, and a new central `--consumption` path. It
retains every score, stack weight, allocation, wealth path, summary, provenance
identity, and file hash. Revision stress permits only the two P1 comparisons.

Do not weaken admission, substitute another archive, reuse a consumed protocol,
or inspect post-July-2016 numeric rows during development. A supported
development result still requires a separately frozen unused-data confirmation;
a valid non-supporting result closes P1.

