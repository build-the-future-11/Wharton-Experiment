# FREM implementation contract v0.1

Status: prospective research artifact; no experiments, training, downloads of market datasets, backtests, simulations, runtime unit tests, or model inference are authorized in this authoring session. Static Python parsing and document rendering are permitted. Never report scientific performance or passing runtime tests.

## Scientific question

Under point-in-time information and equal training budgets, does bounded, delayed-feedback, utility-weighted spectral/engram memory improve predictive log score and net portfolio utility over the same JEPA-reservoir predictor without memory, especially in predefined distribution shifts and low-data settings?

The contribution is a candidate mechanism, not established novelty: a causal trailing spectral descriptor retrieves bounded episodic memories; memory writes and utility updates occur only when outcomes mature. It may fail when signal is absent, motifs do not recur, samples are too few, or costs dominate.

## Package and ownership

Python package: `src/frem`. Python >=3.11. No broker execution or order submission.

- Model owner: `model.py`, `training.py`, `memory.py`, `controller.py`, `baselines.py`, model-specific tests, `docs/MODEL_SPEC.md`.
- Data owner: `data.py`, `splits.py`, `metrics.py`, `statistics.py`, `evaluation.py`, `data_sources.py`, data/evaluation tests, `docs/DATA_GUIDE.md` and `docs/EVALUATION_SPEC.md`.
- Portfolio owner: `portfolio.py`, `risk.py`, `mandate.py`, portfolio tests, `configs/mandates`, `docs/LAURA_MANDATE.md`, `docs/MARKET_RULES.md`, `docs/PORTFOLIO_SPEC.md`.
- Root owner: package/config/CLI/governance/experiments/reporting, integration, paper, README, release packaging.

## Shared window representation

`WindowData` in data.py is a dataclass with `x: ndarray[N,L,F]`, `future_x: ndarray[N,H,F]`, `y: ndarray[N]` (cumulative base-currency log return over H executable intervals), `time: ndarray[N]` (int64 UTC ns decision timestamps), `label_time: ndarray[N]` (int64 UTC ns outcome-availability timestamps), `asset_id: ndarray[N]` (integer IDs), `feature_names: list[str]`, `asset_names: list[str]`. Add optional fields only with defaults. Provide `subset(indices)`.

Rows are sorted by `(time, asset_id)`. All splitting uses unique decision times, never random rows. Training requires `label_time < next_split_start`; embargo uses explicit unique-time bars. Same-time cross-asset rows cannot observe one another's future outcomes. Label starts at the next executable observation strictly after decision and ends H intervals after entry; realized outcome is available no earlier than exit publication. Rows missing an executable target are dropped from training, not silently zero-filled. Market snapshots have availability timestamps. Only known instruments eligible at decision time are used.

## Model public API

`ModelConfig` and `TrainingConfig` dataclasses in model.py and training.py respectively, constructed from config dictionaries. `FREMModel(ModelConfig)` is torch.nn.Module.

`Trainer(model_config, training_config)` with `fit(train: WindowData, validation: WindowData|None=None) -> dict`, `predict(data: WindowData, online_update: bool=False) -> pandas.DataFrame`, `save(path)`, and class method `load(path)`. Predictions include `time, label_time, asset_id, prediction, scale, actual` where prediction/actual are cumulative log returns and scale is strictly positive. Predict must not consume future_x or y to construct current predictions; use y only after maturity in explicitly enabled online mode. Default frozen evaluation. Save optimizer/controller/memory/scaler reference as applicable. A trainer used for one split must not silently fit another.

Baselines: `make_baseline(name, seed=0, **kwargs)` returns an object with `fit(train)` and `predict(data)` returning the same columns. Names should include zero, ridge, reservoir_ridge, xgboost, gru. Model ablations preserve input/split/information parity. XGBoost is a tree model, never a graph traversal algorithm.

## Data and evaluation public API

`load_panel(path, column_map=None) -> DataFrame`; `validate_panel(frame) -> report`; `build_windows(frame, context, horizon, ...) -> WindowData`. Point-in-time feature scaler fitted exclusively on training windows, reusable on validation/test, with serializable state. Root coordinates exact scaler API after implementation.

Use plain named dataclasses with JSON-serializable configurations and descriptive exceptions; no hidden network calls. Data providers are explicit opt-in commands only. Independent bond return paths use total-return/NAV or explicit coupons, not yields as prices. FX conversion must be time-safe.

Evaluation consumes standardized prediction frames, never trains implicitly. Statistics resample blocks of dates preserving cross-asset dependence; random seeds are not independent market histories. Artifacts record every failure. No fabricated charts/results.

## Portfolio public API

Research optimizer takes forecast vector, aligned covariance/scenarios, metadata and a validated mandate. Enforce eligible instruments, long-only/cash, market/sector/asset/turnover limits and liquidity. Fallback must obey the same restrictions or declare infeasible. Cash-flow planning and funding reserve must be separate from speculative return forecasts. Do not invent Laura case facts or encode an unverified Taiwan 10% legal ceiling. Rules carry provenance, effective dates and verification status. Explicitly missing material inputs fail closed for Laura-specific allocations.

## Integration

Agents should message root with exact signatures, dependency choices, sources and assumptions as soon as available. No editing others' owned files without coordination. Each module must be real executable source; no `pass`, `NotImplementedError`, fabricated evidence, TODO implementations or success claims. Template configuration fields for genuinely unknown external inputs are allowed, must be marked and validated.
