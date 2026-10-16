"""Descriptive diagnosis of a retained synthetic run; never fits or tunes a model.

The eight parameter swaps below are arithmetic on already saved distributions,
not new forecasts, causal learning experiments, or candidates for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_engineering_evidence import audit, balanced_mean, digest, nll, require

FIELDS = ("prediction", "scale", "student_df")
DAY_NS = 86_400_000_000_000


def weighted(rows: list[dict], values: list[float]) -> float:
    require(len(rows) == len(values), "Metric length differs from observations")
    return balanced_mean(
        [
            (str(int(row["time"]) // DAY_NS), row["asset"], value)
            for row, value in zip(rows, values, strict=True)
        ]
    )


def pair_rows(candidate: list[dict], control: list[dict]) -> list[dict]:
    def key(row):
        return int(row["time"]), row["asset"]

    lookup = {key(row): row for row in control}
    require(len(lookup) == len(control), "Duplicate control observation")
    require(len({key(row) for row in candidate}) == len(candidate), "Duplicate candidate observation")
    require(set(lookup) == {key(row) for row in candidate}, "Unpaired observation grid")
    aligned = [lookup[key(row)] for row in candidate]
    for a, b in zip(candidate, aligned, strict=True):
        require(
            all(a[k] == b[k] for k in ("actual", "entry_time", "exit_time", "label_time", "asset_id")),
            "Paired outcomes or timing differ",
        )
    return aligned


def distribution_score(rows: list[dict], locations=None) -> float:
    locations = [float(r["prediction"]) for r in rows] if locations is None else locations
    return weighted(
        rows,
        [
            nll(float(row["actual"]), mu, float(row["scale"]), float(row["student_df"]))
            for row, mu in zip(rows, locations, strict=True)
        ],
    )


def describe(rows: list[dict]) -> dict:
    y = [float(r["actual"]) for r in rows]
    mu = [float(r["prediction"]) for r in rows]
    errors = [p - a for p, a in zip(mu, y, strict=True)]
    bias = weighted(rows, errors)
    mse = weighted(rows, [e * e for e in errors])
    return {
        "rows": len(rows),
        "dates": len({int(r["time"]) // DAY_NS for r in rows}),
        "mean_actual": weighted(rows, y),
        "mean_prediction": weighted(rows, mu),
        "mean_error_prediction_minus_actual": bias,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias_squared_share_of_mse": bias * bias / mse if mse else None,
        "centered_error_variance": mse - bias * bias,
        "mae": weighted(rows, [abs(e) for e in errors]),
        "nll": distribution_score(rows),
        "positive_prediction_fraction": weighted(rows, [float(p > 0) for p in mu]),
        "positive_outcome_fraction": weighted(rows, [float(a > 0) for a in y]),
        "mean_scale": weighted(rows, [float(r["scale"]) for r in rows]),
        "mean_df": weighted(rows, [float(r["student_df"]) for r in rows]),
    }


def shapley_reductions(scores: dict[int, float], dimensions: int) -> list[float]:
    """Average each field's score reduction across all replacement orders."""
    require(set(scores) == set(range(1 << dimensions)), "Incomplete swap grid")
    contributions = []
    for i in range(dimensions):
        terms = []
        for mask in range(1 << dimensions):
            if mask & (1 << i):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(dimensions - size - 1) / math.factorial(dimensions)
            terms.append(weight * (scores[mask] - scores[mask | (1 << i)]))
        contributions.append(math.fsum(terms))
    require(
        math.isclose(math.fsum(contributions), scores[0] - scores[(1 << dimensions) - 1], abs_tol=1e-12),
        "Decomposition does not reconstruct gap",
    )
    return contributions


def diagnose(root: Path) -> dict:
    verified = audit(root, root / "research/all_admissible_runs_2026-10-11.json")
    receipt = json.loads((root / "research/all_admissible_runs_2026-10-11.json").read_text())
    run = root / receipt["frem"]["run_directory"]
    sources = dict(verified["source_sha256"])

    def read_json(path):
        sources[str(path.relative_to(root))] = digest(path)
        return json.loads(path.read_text())

    all_rows = {}
    metrics, fits = {}, {}
    for cell in verified["cells"]:
        name = cell["model"]
        directory = run / name / f"seed_{cell['seed']}"
        with (directory / "predictions.csv").open(newline="") as stream:
            all_rows[name] = list(csv.DictReader(stream))
        metrics[name] = read_json(directory / "metrics.json")
        fits[name] = read_json(directory / "fit.json")
    f = all_rows["frem"]
    c = pair_rows(f, all_rows["no_memory"])
    base = distribution_score(f, [float(r["base_prediction"]) for r in f])
    swaps = {}
    for mask in range(8):
        losses = []
        for a, b in zip(f, c, strict=True):
            params = [float((b if mask & (1 << i) else a)[key]) for i, key in enumerate(FIELDS)]
            losses.append(nll(float(a["actual"]), *params))
        swaps[mask] = weighted(f, losses)
    contributions = shapley_reductions(swaps, 3)
    correction = [float(r["prediction"]) - float(r["base_prediction"]) for r in f]
    gates = [float(r["memory_gate"]) for r in f]
    states = {name: read_json(run / name / "seed_11/checkpoint/state.json") for name in ("frem", "no_memory")}
    records = states["frem"]["memory"]["records"]
    times = [int(r["time"]) for r in f]
    memory = {
        "retained_records": len(records),
        "records_per_asset": {
            asset: sum(r["asset_id"] == i for r in records)
            for i, asset in enumerate(states["frem"]["asset_names"])
        },
        "retained_records_never_credited": sum(r["credited_uses"] == 0 for r in records),
        "retained_record_credit_count": sum(r["credited_uses"] for r in records),
        "credit_count_scope": "Retained records only, not lifetime credits or independent observations",
        "mean_stored_residual": statistics.mean(r["residual"] for r in records),
        "mean_historical_base_prediction": statistics.mean(r["outcome"] - r["residual"] for r in records),
        "minimum_historical_base_prediction": min(r["outcome"] - r["residual"] for r in records),
        "maximum_historical_base_prediction": max(r["outcome"] - r["residual"] for r in records),
    }
    for label, timestamp in (("first_test", min(times)), ("last_test", max(times))):
        ages = [(timestamp - r["label_time"]) / DAY_NS for r in records]
        memory[label + "_all_retained_record_age_days"] = {
            "min": min(ages),
            "mean": statistics.mean(ages),
            "max": max(ages),
        }
    for relative in (
        "scripts/generate_yahoo_calibrated_engineering_panel.py",
        "scripts/diagnose_frem_loss.py",
        "scripts/audit_engineering_evidence.py",
    ):
        sources[relative] = digest(root / relative)
    return {
        "status": "COMPLETE_DESCRIPTIVE_DIAGNOSIS_OF_RETAINED_SYNTHETIC_RUN",
        "created_utc": datetime.now(UTC).isoformat(),
        "scientific_status": "NOT_EVALUATED",
        "new_model_fits": 0,
        "protected_outcomes_accessed": False,
        "run_directory": str(run.relative_to(root)),
        "scope": "One synthetic path, four assets, seed 11; saved development predictions only",
        "models": {name: describe(rows) for name, rows in all_rows.items()},
        "by_asset": {
            asset: {
                name: describe([r for r in rows if r["asset"] == asset]) for name, rows in all_rows.items()
            }
            for asset in sorted({r["asset"] for r in f})
        },
        "frem_fixed_checkpoint_memory": {
            "base_nll": base,
            "deployed_nll": swaps[0],
            "nll_reduction_from_correction": base - swaps[0],
            "mean_correction": weighted(f, correction),
            "mean_absolute_correction": weighted(f, [abs(x) for x in correction]),
            "min_correction": min(correction),
            "max_correction": max(correction),
            "mean_gate": weighted(f, gates),
            "min_gate": min(gates),
            "max_gate": max(gates),
            "rows_with_four_neighbors": sum(int(r["memory_neighbors"]) == 4 for r in f),
            "selected_controller_weight": 1 / (1 + math.exp(-states["frem"]["controller"]["memory_logit"])),
        },
        "distribution_swap_diagnostic": {
            "warning": "Post-hoc arithmetic, not causal training attribution, deployable calibration, or selection",
            "control_minus_candidate_original_nll": swaps[7] - swaps[0],
            "all_eight_swaps": [
                {
                    "fields_from_no_memory": [field for i, field in enumerate(FIELDS) if mask & (1 << i)],
                    "nll": value,
                }
                for mask, value in swaps.items()
            ],
            "order_averaged_nll_reductions": dict(zip(FIELDS, contributions, strict=True)),
            "order_averaged_gap_fractions": {
                key: value / (swaps[0] - swaps[7]) for key, value in zip(FIELDS, contributions, strict=True)
            },
        },
        "checkpoint_model_configuration_differences": {
            k: {name: states[name]["model_config"][k] for name in states}
            for k, v in states["frem"]["model_config"].items()
            if v != states["no_memory"]["model_config"][k]
        },
        "checkpoint_scalers_equal": states["frem"]["scaler"] == states["no_memory"]["scaler"],
        "memory": memory,
        "retained_fit_histories": fits,
        "retained_metrics": metrics,
        "retained_inference_unchanged": verified["retained_inference"],
        "source_sha256": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output exists; preserve the prior diagnosis")
    root = Path(__file__).resolve().parents[1]
    report = diagnose(root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "memory": report["frem_fixed_checkpoint_memory"],
                "decomposition": report["distribution_swap_diagnostic"]["order_averaged_gap_fractions"],
            }
        )
    )


if __name__ == "__main__":
    main()
