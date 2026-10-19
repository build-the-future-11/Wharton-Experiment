# Kaggle runtime repair

New delivery: `tools/kaggle/FREM_S10_KAGGLE_CPU.ipynb`; instructions in
`tools/kaggle/README.md`. It prepares isolated Python 3.12 and hash-pinned
NumPy 2.5.4, and explicitly uses that interpreter for every helper/model child.
The notebook kernel remains untouched. Setup requires Internet and retains
logs/failure receipts under its separate 300-second bound. The original deadline
remains 22:00 IST on 16 October; no scientific or numerical gate was relaxed.

28 stdlib tests passed under Python 3.12; the same 28 passed under Python 3.14
before the final installer-isolation flag change. Nine generated notebook code
cells parse; embedded helper, requirement bytes and source archive hashes match.
Scoped E9/F lint passed. Duplicate JSON keys are rejected; active invocation
exports are explicitly nonresumable; descendant cleanup errors are retained.
Prior source and Kaggle evidence directories are unchanged against HEAD.

Actual Linux dependency download/install and target numerical execution are
NOT_RUN. Local numerical admission failed (361.59 / 688.42 / 596.58 MiB; all
three must exceed 512 MiB). No full-suite or all-bugs-free claim is made.
Scientific status remains NOT_EVALUATED. Read
`research/frem_s10_kaggle_fix_20261016/RECEIPT.json` for exact checks and hashes.

Preserved failed checks: attempt1 had a stale copied test path; attempt3 exposed
macOS cleanup PermissionError after an exited leader. Both are retained alongside
the corrected passing checks. Network/install commands are covered by injected
runner tests; this does not establish package availability or Linux compatibility.

The scientific package remains the exact sealed candidate. The repair is in the
notebook delivery layer. Complete production qualification, cgroup compatibility,
market-data admission, precision, freezing and empirical evaluation remain open.
