"""Fixed-budget S6 development cells; no implicit protected execution."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from successors.frem_portfolio_v1.engine import Forecast, backtest

from .check import write
from .model import MixtureForecast, ModelConfig, S6Forecaster


def planned_cells(input_dim, *, with_xgboost=False):
    cells = []
    for seed in (11, 29, 47):
        for context in ("full", "half", "zero"):
            for width in (8, 16):
                config = ModelConfig(input_dim, state_dim=width, latent_dim=width, rff_dim=width,
                                     seed=seed, context=context, use_xgboost=with_xgboost)
                cells.append((f"seed{seed}_{context}_width{width}", config))
        for control, kwargs in (("no_jepa", {"jepa_weight": 0.0}),
                                ("immediate", {"ripple_delay": 0, "ripple_steps": 1, "quantization_step": 0.0})):
            cells.append((f"seed{seed}_full_width8_{control}",
                          ModelConfig(input_dim, seed=seed, use_xgboost=with_xgboost, **kwargs)))
    return cells


def validate_protocol(protocol, data, *, with_xgboost):
    """Bind a declared development plan to exact input dates and model budgets.

    Source rights and whether a date has actually been inspected still require
    honest provenance. A supplied claim is not externally certified by this code.
    """
    expected = {name: asdict(config) for name, config in planned_cells(
        data["train"]["windows"].shape[-1], with_xgboost=with_xgboost)}
    if protocol.get("cell_configurations") != expected:
        raise ValueError("protocol must freeze every actual cell configuration")
    for part in ("train", "calibration", "evaluation"):
        dates = np.unique(data[part]["origin_times"]).tolist()
        if protocol.get("fold_dates", {}).get(part) != dates:
            raise ValueError("protocol date grid does not match data")
    precision = protocol.get("precision_analysis", {})
    required = precision.get("required_evaluation_dates") if isinstance(precision, dict) else None
    if type(required) is not int or required < 24 or required > len(np.unique(data["evaluation"]["origin_times"])):
        raise ValueError("precision-qualified evaluation dates required")
    if precision.get("block_dates") != [3, 6] or not precision.get("development_effect_analysis"):
        raise ValueError("dated-block precision analysis is incomplete")
    inventory = protocol.get("spent_date_inventory")
    if not isinstance(inventory, dict) or not inventory.get("source") or not isinstance(inventory.get("dates"), list):
        raise ValueError("explicit spent-date inventory required")
    if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in inventory["dates"]):
        raise ValueError("spent dates must be finite numeric UTC seconds")
    if set(inventory["dates"]) & set(np.unique(data["evaluation"]["origin_times"])):
        raise ValueError("new S6 outer evaluation cannot reopen spent dates")
    if protocol.get("primary_endpoint") != "mean_date_level_mixture_nll":
        raise ValueError("unsupported primary endpoint")
    if protocol.get("economic_policy") != "shared_fixed_80_20_signal":
        raise ValueError("economic policy mismatch")


def economics_from_records(records, currency):
    """Reconstruct economic comparisons from complete saved mixture forecasts."""
    periods = {}
    for record in records:
        width = len(record["mean"][0])
        dist = MixtureForecast(record["mean"], record["scale"], record["df"], record["weights"],
                               tuple(str(i) for i in range(width)))
        mean, variance = dist.log_return_moments()
        periods.setdefault(record["variant"], []).append({
            "time": record["date"], "end": record["end"], "currency": currency,
            "returns": dict(zip(record["assets"], record["simple_returns"], strict=True)),
            "forecasts": [Forecast(a, float(m), float(v)) for a, m, v in
                          zip(record["assets"], mean, variance, strict=True)]})
    results = {}
    for cost in (10, 20, 40):
        ledgers = {name: backtest(history, cost_bps=cost) for name, history in periods.items()}
        ledgers.update({name: backtest(periods["equal"], policy=name, cost_bps=cost)
                        for name in ("cash", "buy_hold", "equal_weight")})
        results[str(cost)] = ledgers
    return results


def run_cell(data, config, output):
    """One declared fit with all exact-distribution controls, costs and raw rows.

    Called by the admitted worker or tiny integrity tests. No selection occurs on
    evaluation outcomes. Stocks on the same date are one statistical episode.
    Fixed signal policy is shared by all forecast variants, separating forecast
    differences from policy differences. It is not a certified Laura mandate.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train, cal, evaluation = (data[name] for name in ("train", "calibration", "evaluation"))
    model = S6Forecaster(config)
    model.fit(train["windows"], train["future_windows"], train["y"], origin_times=train["origin_times"],
              label_available=train["available_times"], future_available=train["future_available"].max(axis=1),
              fit_cutoff=data["fit_cutoff"])
    model.calibrate(cal["windows"], cal["y"], origin_times=cal["origin_times"],
                    maturity_times=cal["maturity_times"], available_times=cal["available_times"])
    model.save(output / "fit_checkpoint.json")
    raw = []
    periods = {}
    names = ["memory", "equal", "recent", *model.names]
    for date in np.unique(evaluation["origin_times"]):
        idx = evaluation["origin_times"] == date
        assets = np.asarray(evaluation["assets"])[idx].tolist()
        forecasts = {router: model.forecast(evaluation["windows"][idx], issue_time=float(date),
                                            asset_ids=assets, router=router)
                     for router in ("memory", "equal", "recent")}
        main = forecasts["memory"]
        for column, name in enumerate(model.names):
            forecasts[name] = MixtureForecast(main.mean[:, column:column+1], main.scale[:, column:column+1],
                                              main.df[:, column:column+1], np.ones((len(assets), 1)), (name,))
        for name in names:
            forecast = forecasts[name]
            mu, var = forecast.log_return_moments()
            nll = forecast.nll(evaluation["y"][idx])
            record = {"date": float(date), "end": float(evaluation["maturity_times"][idx][0]),
                      "assets": assets, "y": evaluation["y"][idx].tolist(),
                      "simple_returns": evaluation["simple_returns"][idx].tolist(),
                      "variant": name, "mean": forecast.mean.tolist(), "scale": forecast.scale.tolist(),
                      "df": forecast.df.tolist(), "weights": forecast.weights.tolist(),
                      "nll": nll.tolist(), "date_nll": float(nll.mean())}
            raw.append(record)
            periods.setdefault(name, []).append({
                "time": record["date"], "end": record["end"], "currency": data["currency"],
                "forecasts": [Forecast(a, float(m), float(v)) for a, m, v in zip(assets, mu, var, strict=True)],
                "returns": dict(zip(assets, record["simple_returns"], strict=True)), "cash_return": 0.0})
        # Outcomes may be queued now; neither memory nor recent control may retrieve
        # them until the whole date's maturity and availability strictly precede issue.
        model.observe(issue_time=float(date), y=evaluation["y"][idx], asset_ids=assets,
                      maturity_times=evaluation["maturity_times"][idx],
                      available_times=evaluation["available_times"][idx])
    economics = {}
    for cost in (10, 20, 40):
        ledgers = {name: backtest(history, cost_bps=cost) for name, history in periods.items()}
        ledgers.update({name: backtest(periods["equal"], policy=name, cost_bps=cost)
                        for name in ("equal_weight", "buy_hold", "cash")})
        economics[str(cost)] = ledgers
    write(output / "config.json", asdict(config))
    write(output / "predictions.json", raw)
    write(output / "economics.json", economics)
    model.save(output / "online_checkpoint.json")
    scores = {name: float(np.mean([r["date_nll"] for r in raw if r["variant"] == name])) for name in names}
    receipt = {"status": "COMPLETE_DEVELOPMENT_CELL", "scores": scores,
               "evaluation_dates": len(np.unique(evaluation["origin_times"])),
               "training_history": model.history,
               "terminal_wealth_10bps": {name: item["terminal_wealth"] for name, item in economics["10"].items()},
               "scope": "Dated custom admitted input, retrospective research ledger; no actual fills or causal claim",
               "protected_access": False}
    write(output / "receipt.json", receipt)
    return raw, receipt


def ensemble_records(families):
    """Pool every seed distribution, never select the favorable initialization."""
    if not families or any(len(family) != len(families[0]) for family in families):
        raise ValueError("all complete seed families required")
    combined = []
    for rows in zip(*families, strict=True):
        first = rows[0]
        for row in rows[1:]:
            if any(row[k] != first[k] for k in ("date", "end", "assets", "y", "simple_returns", "variant")):
                raise ValueError("ensemble row grid or outcomes differ")
        arrays = [np.concatenate([np.asarray(row[field]) for row in rows], axis=1)
                  for field in ("mean", "scale", "df", "weights")]
        arrays[-1] /= len(rows)
        distribution = MixtureForecast(*arrays, tuple(str(i) for i in range(arrays[0].shape[1])))
        nll = distribution.nll(first["y"])
        combined.append({**first, **dict(zip(("mean", "scale", "df", "weights"),
                                              [v.tolist() for v in arrays], strict=True)),
                         "nll": nll.tolist(), "date_nll": float(nll.mean())})
    return combined


def paired_date_interval(differences, *, block_dates=6, repetitions=2000, seed=1901):
    """Prescribed circular date blocks; no shortening to manufacture precision."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not len(values):
        raise ValueError("finite date-level differences required")
    if type(block_dates) is not int or block_dates < 1 or type(repetitions) is not int or repetitions < 100:
        raise ValueError("invalid bootstrap parameters")
    if len(values) < 4 * block_dates:
        return {"status": "INSUFFICIENT_DATES", "dates": len(values), "block_dates": block_dates,
                "mean": float(values.mean())}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    offsets = np.arange(block_dates)
    for j in range(repetitions):
        starts = rng.integers(len(values), size=int(np.ceil(len(values) / block_dates)))
        indices = ((starts[:, None] + offsets) % len(values)).ravel()[:len(values)]
        draws[j] = values[indices].mean()
    return {"status": "DESCRIPTIVE_DEPENDENT_DATE_INTERVAL", "mean": float(values.mean()),
            "interval95": np.quantile(draws, [.025, .975]).tolist(), "block_dates": block_dates,
            "scope": "Conditional development comparison, not confirmatory significance or a causal effect"}
