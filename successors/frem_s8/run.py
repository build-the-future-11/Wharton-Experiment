"""One frozen three-seed run; no search or outcome-driven retries."""

import argparse
import json
import resource
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from successors.frem_portfolio_v1.engine import backtest
from successors.frem_s6.experiment import ensemble_records, paired_date_interval
from successors.frem_s6.model import MixtureForecast, ModelConfig, S6Forecaster, student_logpdf
from successors.frem_s6_real.prepare import sha, stamp, write
from successors.frem_s7.data import validate_dataset
from successors.frem_s7.router import MixtureMemory, select_strength
from successors.frem_s7.run import cost_aware, period

from .policy import holding_policy
from .router import SoftGate

SEEDS = (11, 29, 47)
COSTS = (0, 10, 20, 40)
BLOCKS = (6, 12)


def validate_scope(payload):
    """Inspect only timing metadata before converting any outcome or feature."""
    for name, part in payload["partitions"].items():
        low, high = (
            (stamp("2020-01-01"), stamp("2023-01-01"))
            if name == "evaluation"
            else (stamp("2005-01-01"), stamp("2019-01-01"))
        )
        for key in (
            "origin_times",
            "maturity_times",
            "available_times",
            "window_times",
            "window_available",
            "future_times",
            "future_available",
        ):
            if key in part:
                values = np.asarray(part[key], float)
                if not np.isfinite(values).all() or np.any(values < low) or np.any(values >= high):
                    raise ValueError(f"{name}.{key}: 2019 or out-of-scope metadata")
    return validate_dataset(payload)


def record(part, idx, name, forecast):
    nll = forecast.nll(part["y"][idx])
    return {
        "date": float(part["origin_times"][idx][0]),
        "end": float(part["maturity_times"][idx][0]),
        "available": float(part["available_times"][idx].max()),
        "feature_available_max": float(part["window_available"][idx].max()),
        "assets": np.asarray(part["assets"])[idx].tolist(),
        "y": part["y"][idx].tolist(),
        "simple_returns": part["simple_returns"][idx].tolist(),
        "variant": name,
        "mean": forecast.mean.tolist(),
        "scale": forecast.scale.tolist(),
        "df": forecast.df.tolist(),
        "weights": forecast.weights.tolist(),
        "nll": nll.tolist(),
        "date_nll": float(nll.mean()),
    }


def fit_seed(data, output, seed):
    folder = output / f"seed{seed}"
    folder.mkdir()
    started = time.monotonic()
    model = S6Forecaster(ModelConfig(1, seed=seed, use_xgboost=True))
    train, cal, evaluation = (data["partitions"][p] for p in ("train", "calibration", "evaluation"))
    model.fit(
        train["windows"],
        train["future_windows"],
        train["y"],
        origin_times=train["origin_times"],
        label_available=train["available_times"],
        future_available=train["future_available"].max(1),
        fit_cutoff=data["metadata"]["fit_cutoff"],
    )
    split = stamp("2018-01-01")
    early = cal["available_times"] < split
    model.calibrate(
        cal["windows"][early],
        cal["y"][early],
        origin_times=cal["origin_times"][early],
        maturity_times=cal["maturity_times"][early],
        available_times=cal["available_times"][early],
    )
    memory = MixtureMemory(len(model.names))
    local_gate, global_gate = SoftGate(), SoftGate()
    for date in np.unique(cal["origin_times"][early]):
        idx = (cal["origin_times"] == date) & early
        keys, means, scales = model._experts(cal["windows"][idx])
        memory.add(
            keys.mean(0),
            student_logpdf(cal["y"][idx, None], means, scales, model.config.df),
            cal["available_times"][idx].max(),
        )
    gains = {a: [] for a in (0.25, 0.5, 1.0)}
    trace, alpha = [], 0.0
    evaluation_rows = None
    for stage, part in [("selection", cal), ("evaluation", evaluation)]:
        rows = []
        if stage == "evaluation":
            alpha = select_strength(gains)
            write(
                folder / "selection.json",
                {
                    "s7_alpha": alpha,
                    "gains_2018": gains,
                    "s8_selection": "Fixed algorithm; no discrete parameter selected.",
                },
            )
            model.save(folder / "fit_checkpoint.json")
            write(
                folder / "gates_before_evaluation.json",
                {"local": local_gate.state_dict(), "global": global_gate.state_dict()},
            )
        dates = np.unique(part["origin_times"])
        if stage == "selection":
            dates = dates[dates > max(split, model.calibration_cutoff)]
        for date in dates:
            idx = part["origin_times"] == date
            assets = np.asarray(part["assets"])[idx].tolist()
            old = {
                r: model.forecast(part["windows"][idx], issue_time=float(date), asset_ids=assets, router=r)
                for r in ("memory", "equal", "recent")
            }
            base = old["equal"]
            keys, _, _ = model._experts(part["windows"][idx])
            key = keys.mean(0)
            local, global_w = memory.weights(key, date), memory.weights(key, date, contextual=False)
            q, qg = local_gate.weights(date), global_gate.weights(date)

            def dist(w, base=base):
                return MixtureForecast(
                    base.mean, base.scale, base.df, np.broadcast_to(w, base.mean.shape).copy(), base.names
                )

            uniform = np.ones(len(local)) / len(local)
            forecasts = {
                **old,
                "s8": dist(q[0] * uniform + q[1] * local),
                "s8_global": dist(qg[0] * uniform + qg[1] * global_w),
                "fixed_half": dist(0.5 * uniform + 0.5 * local),
                "s7": dist((1 - alpha) * uniform + alpha * local),
                "mixture_ungated": dist(local),
                "mixture_global": dist(global_w),
            }
            for j, name in enumerate(model.names):
                forecasts[name] = MixtureForecast(
                    base.mean[:, j : j + 1],
                    base.scale[:, j : j + 1],
                    base.df[:, j : j + 1],
                    np.ones((len(assets), 1)),
                    (name,),
                )
            for name, f in forecasts.items():
                rows.append(record(part, idx, name, f))
            loss = {name: float(f.nll(part["y"][idx]).mean()) for name, f in forecasts.items()}
            if stage == "selection":
                for a, values in gains.items():
                    values.append(
                        float(loss["equal"] - dist((1 - a) * uniform + a * local).nll(part["y"][idx]).mean())
                    )
            available = float(part["available_times"][idx].max())
            trace.append(
                {
                    "stage": stage,
                    "date": float(date),
                    "available": available,
                    "local_gate": q.tolist(),
                    "global_gate": qg.tolist(),
                    "local_expert_weights": local.tolist(),
                    "global_expert_weights": global_w.tolist(),
                }
            )
            local_gate.observe(date, available, [loss["equal"], loss["mixture_ungated"]])
            global_gate.observe(date, available, [loss["equal"], loss["mixture_global"]])
            memory.add(key, student_logpdf(part["y"][idx, None], base.mean, base.scale, base.df), available)
            model.observe(
                issue_time=float(date),
                y=part["y"][idx],
                asset_ids=assets,
                maturity_times=part["maturity_times"][idx],
                available_times=part["available_times"][idx],
            )
        write(folder / f"{stage}_predictions.json", rows)
        if stage == "evaluation":
            evaluation_rows = rows
    model.save(folder / "online_checkpoint.json")
    write(folder / "gate_state.json", {"local": local_gate.state_dict(), "global": global_gate.state_dict()})
    write(folder / "trace.json", trace)
    write(
        folder / "mixture_state.json",
        {
            "records": [
                {"available": t, "key": k.tolist(), "density": v.tolist()} for t, k, v in memory.records
            ]
        },
    )
    write(
        folder / "receipt.json",
        {
            "status": "COMPLETE",
            "seed": seed,
            "s7_alpha": alpha,
            "rows": len(evaluation_rows),
            "config": asdict(model.config),
            "seconds": time.monotonic() - started,
        },
    )
    return evaluation_rows


def run(data, output):
    families = [fit_seed(data, output, seed) for seed in SEEDS]
    combined = ensemble_records(families)
    write(output / "ensembles.json", combined)
    variants = sorted({r["variant"] for r in combined})
    selected = np.array([r["date_nll"] for r in combined if r["variant"] == "s8"])
    comparisons, economics, scores = {}, {}, {}
    for name in variants:
        rows = [r for r in combined if r["variant"] == name]
        baseline = np.array([r["date_nll"] for r in rows])
        comparisons[name] = {str(b): paired_date_interval(baseline - selected, block_dates=b) for b in BLOCKS}
        scores[name] = {
            "ensemble_nll": float(baseline.mean()),
            "seed_nll": {
                str(seed): float(np.mean([r["date_nll"] for r in family if r["variant"] == name]))
                for seed, family in zip(SEEDS, families, strict=True)
            },
            "years": {
                str(y): float(
                    np.mean([r["date_nll"] for r in rows if datetime.fromtimestamp(r["date"], UTC).year == y])
                )
                for y in (2020, 2021, 2022)
            },
        }
        periods = [period(r) for r in rows]
        economics[name] = {
            str(c): {
                "signal": backtest(periods, cost_bps=c),
                "cost_aware": cost_aware(periods, c),
                "s8_h4": holding_policy(periods, c, horizon=4),
                "s8_h1": holding_policy(periods, c, horizon=1),
            }
            for c in COSTS
        }
    periods = [period(r) for r in combined if r["variant"] == "equal"]
    benchmarks = {
        v: {str(c): backtest(periods, policy=v, cost_bps=c) for c in COSTS}
        for v in ("cash", "buy_hold", "equal_weight")
    }
    for name, value in [
        ("scores", scores),
        ("comparisons", comparisons),
        ("economics", economics),
        ("benchmarks", benchmarks),
    ]:
        write(output / f"{name}.json", value)
    write(
        output / "completion.json",
        {
            "status": "COMPLETE",
            "base_fits": 3,
            "seeds": list(SEEDS),
            "variants": variants,
            "evaluation_dates": len(selected),
            "costs": list(COSTS),
            "protected_access": False,
            "2019_used": False,
            "scope": "Frozen development on 2020-2022 retrospective ETF histories.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        frozen = json.loads(args.freeze.read_text())
        for name, digest in frozen["sources"].items():
            if sha(name) != digest:
                raise ValueError("source drift: " + name)
        identities = {str(p): sha(p) for p in sorted(args.root.glob("*.json"))}
        write(args.output / "data_identity.json", {"files": identities, "freeze_sha256": sha(args.freeze)})
        # Permanent study receipt prevents an unchanged completed evaluation repeat.
        write(
            args.freeze.parent / "evaluation_consumed.json",
            {
                "started_at": datetime.now(UTC).isoformat(),
                "output": str(args.output),
                "freeze_sha256": sha(args.freeze),
                "data_files": identities,
            },
        )
        data = validate_scope(json.loads((args.root / "windows.json").read_text()))
        run(data, args.output)
        for name, digest in frozen["sources"].items():
            if sha(name) != digest:
                raise ValueError("source drift during run: " + name)
    except BaseException:
        write(args.output / "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        write(
            args.output / "resources.json",
            {
                "seconds": time.monotonic() - started,
                "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        )


if __name__ == "__main__":
    main()
