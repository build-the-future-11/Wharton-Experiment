# Wharton amended model candidate

Real local implementation under amendment A-20261008. Current revision: **A-20261008-R2**. Read `SPEC.md` for exact mechanisms, policy alternatives, data contracts and limitations. The historical Wharton repository and spent lockbox are untouched. This candidate is not a validated final investment model.

## Layout

- `wharton_final/accounting.py`: deterministic cash, units, marked holdings, scheduled liabilities, arrears, fees and payment ledger.
- `wharton_final/client_contract.py`: source-linked contribution/obligation events and explicit-assumption reserve arithmetic.
- `wharton_final/data.py`: offline manifest admission, timestamps, causal contexts, compounded horizons and maturity gates.
- `wharton_final/model.py`: trainable JEPA, contractive reservoir, XGBoost heads, matured residual memory and checkpointing.
- `wharton_final/planning.py`: forecast-conditioned bootstrap scenarios, shared-action constrained planner and ledger rollout.
- `wharton_final/pipeline.py`: chronological candidate/baseline evaluation, inference and hashed receipts.
- `wharton_final/__main__.py`: evaluate, forecast/plan, and ledger commands.
- `tests/`: hand-verifiable accounting, chronological isolation, model and end-to-end correctness tests.
- `configs/full-model.json`: explicit development model parameters.
- `configs/ledger-correctness.json`: hand-verifiable accounting fixture, not client facts or market performance.
- `evidence/`: recovered source revisions, test logs, environment and completion records.

## Run

Use Python 3.10+ with dependencies from `pyproject.toml`. The available host Python has numpy/scipy/sklearn/torch; the local `.deps` directory supplies XGBoost when present. Commands from this directory:

```sh
PYTHONPATH=.deps:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -v
PYTHONPATH=.deps:. python3 -m wharton_final ledger --contract configs/ledger-correctness.json --output ledger-receipt.json
PYTHONPATH=.deps:. python3 -m wharton_final evaluate --manifest /absolute/path/to/admitted-manifest.json --config configs/full-model.json --first-test-row 600 --test-count 3 --stride 20 --output development-run
```

The evaluation command is runnable when its input is admitted. A manifest uses these fields: `csv` (relative to manifest), `sha256` (exact CSV SHA-256), `assets` (ordered unique ticker list), `source`, `license`, `lawful_use_attested: true`, `asset_order_verified: true`, `role: "development"`, `data_kind: "market"` (or explicitly `"synthetic_fixture"` for correctness), `return_kind: "total"`, `protected_rows: []`. Historical cache derivatives additionally require `upstream_sha256` and exact `upstream_rows`. Do not supply excluded/protected rows. The CSV header must be `decision_time,observed_at,<assets in exact order>`. Returns are finite simple total returns > -1. Timestamps include timezone; decisions and observations are strictly increasing, and the observation timestamp cannot precede its decision timestamp.

After evaluation, forecast from its trusted local checkpoint using the hash in `receipt.json`:

```sh
PYTHONPATH=.deps:. python3 -m wharton_final forecast --checkpoint development-run/model-600.pkl --checkpoint-sha256 HASH_FROM_RECEIPT --manifest /absolute/path/to/admitted-manifest.json --as-of 2026-10-09T10:00:00Z --max-context-age-seconds 86400 --output forecast.json
```

Optional `--plan-contract` supplies JSON with `assets`, `state` (`at`, `units`, `prices`, `cash`, optionally scheduled `liabilities` and `arrears`), all five `policy` fields, `scenario_count`, `seed`, `cash_rates` (explicit rate at each horizon step), `future_liability_policy` (`reserve_nominal_cash` or `report_only`), `obligations` (one list per horizon step), `max_weight`, `alpha`, and `terminal_cash_target`. Each obligation has `id`, integer `due_at`, and `amount`. The abstract integer ledger clock is separate from market timestamps and must be mapped deliberately by the contract provider. State asset order must match the forecast panel. Every ledger step must provide `cash_rate` and `return_kind`; optional `contributions` contain `{id, amount}` beginning-of-step deposits. The sample 86400-second context-age allowance is a demonstration assumption, not a market-calendar rule. Every input and output is local; there is no live execution.

All generated correctness data is explicitly synthetic. No real-market metric has been produced by this candidate. See `evidence/REVIEW_COMPLETION.json` for current verified scope and blockers. Initial `COMPLETION.json`, initial test logs and `correctness-fixture/` are preserved revision-1 history; their checkpoints intentionally fail the revised-source inference gate. Current synthetic CLI artifacts are in `review-correctness-fixture/`. The initial Library ZIP is preserved as version 0; the reviewed ZIP replaces that same Library item as a new version.
