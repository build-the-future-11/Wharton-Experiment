"""Recompute retained synthetic scores without importing the research stack.

This audit neither fits a model nor reads the protected market panel. It refuses
non-synthetic inputs and writes a new evidence directory, never a prior result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nll(actual: float, prediction: float, scale: float, df: float) -> float:
    require(all(math.isfinite(x) for x in (actual, prediction, scale, df)), "Nonfinite forecast")
    require(scale > 0 and df > 0, "Invalid Student-t parameters")
    return (
        math.lgamma(df / 2) - math.lgamma((df + 1) / 2)
        + 0.5 * math.log(df * math.pi) + math.log(scale)
        + (df + 1) / 2 * math.log1p(((actual - prediction) / scale) ** 2 / df)
    )


def balanced_mean(values: list[tuple[str, str, float]]) -> float:
    """Equal dates; equal assets per date; repeated rows averaged per asset."""
    by_asset: dict[tuple[str, str], list[float]] = defaultdict(list)
    for date, asset, value in values:
        by_asset[date, asset].append(value)
    by_date: dict[str, list[float]] = defaultdict(list)
    for (date, _), rows in by_asset.items():
        by_date[date].append(math.fsum(rows) / len(rows))
    require(bool(by_date), "Empty prediction table")
    return math.fsum(math.fsum(v) / len(v) for v in by_date.values()) / len(by_date)


def score_rows(rows: list[dict], model: str, seed: int, cutoff: int, protected: int) -> dict:
    seen = set()
    scores: dict[str, list] = defaultdict(list)
    outcomes = {}
    for row in rows:
        require(row["model"] == model and int(row["seed"]) == seed, "Wrong model/seed")
        t, entry, end, mature = (int(row[k]) for k in ("time", "entry_time", "exit_time", "label_time"))
        require(cutoff < t < entry < end <= mature < protected, "Timing or protected-boundary violation")
        key = (t, row["asset"])
        require(key not in seen, "Duplicate observation")
        seen.add(key)
        y, mu, scale, df = (float(row[k]) for k in ("actual", "prediction", "scale", "student_df"))
        loss = nll(y, mu, scale, df)
        date = datetime.fromtimestamp(t / 1e9, UTC).date().isoformat()
        outcomes[key] = (y, entry, end, mature, row["asset_id"])
        for metric, value in (
            ("mean_nll", loss), ("mae", abs(y - mu)),
            ("direction_accuracy", float((mu > 0) == (y > 0))),
        ):
            scores[metric].append((date, row["asset"], value))
    return {
        "metrics": {name: balanced_mean(values) for name, values in scores.items()},
        "outcomes": outcomes, "rows": len(rows),
        "dates": len({v[0] for v in scores["mean_nll"]}),
    }


def verify_source_archive(root: Path, source_root: Path, expected: dict[str, str]) -> dict[str, str]:
    """Use exact archived bytes for old runs, never replace the saved identity."""
    root, source_root = root.resolve(), source_root.resolve()
    require(source_root.is_relative_to(root), "Source archive escapes repository")
    verified = {}
    for relative, expected_hash in expected.items():
        path = (source_root / relative).resolve()
        require(path.is_relative_to(source_root), "Source path escapes archive")
        actual = digest(path)
        require(actual == expected_hash, f"Hash mismatch: {path}")
        verified[str(path.relative_to(root))] = actual
    return verified


def audit(root: Path, receipt_path: Path, *, source_root: Path | None = None) -> dict:
    root = root.resolve()
    source_root = root if source_root is None else source_root.resolve()
    sources = {}

    def checked(path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        require(path.is_relative_to(root), "Evidence path escapes repository")
        actual = digest(path)
        require(expected is None or actual == expected, f"Hash mismatch: {path}")
        sources[str(path.relative_to(root))] = actual
        return path

    def read(path: Path, expected: str | None = None):
        return json.loads(checked(path, expected).read_text())

    receipt = read(receipt_path)
    require(receipt["scientific_claims_allowed"] is False, "Receipt must prohibit scientific claims")
    data = receipt["data"]
    manifest = read(root / data["manifest"], data["manifest_sha256"])
    require(manifest["dataset_synthetic"] is True, "Only declared synthetic data admitted")
    require(manifest["scientific_claims_allowed"] is False, "Synthetic claim boundary missing")
    checked(root / data["path"], data["panel_sha256"])  # Opaque bytes only; no panel deserialization.
    frem = receipt["frem"]
    run = root / frem["run_directory"]
    run_manifest = read(run / "manifest.json", frem["manifest_sha256"])
    summary = read(run / "summary.json", frem["summary_sha256"])
    analysis = read(root / frem["primary_comparison"]["path"], frem["primary_comparison"]["sha256"])
    require(run_manifest["status"] == "COMPLETE", "Incomplete matrix")
    require(run_manifest["phase"] == analysis["phase"] == "development", "Unexpected phase")
    require(run_manifest["identity"]["data"]["panel"] == data["panel_sha256"], "Data identity mismatch")
    sources.update(verify_source_archive(root, source_root, run_manifest["identity"]["code"]["files"]))
    for relative, expected in analysis["source_prediction_sha256"].items():
        checked(run / relative, expected)
    config = run_manifest["configuration"]
    expected_cells = {(model, seed) for model in config["models"] for seed in config["seeds"]}
    actual_cells = [(cell["model"], cell["seed"]) for cell in summary["cells"]]
    require(len(actual_cells) == len(set(actual_cells)) and set(actual_cells) == expected_cells,
            "Incomplete or duplicated configured cell family")
    require(len(config["seeds"]) == 1, "This bounded audit supports one retained engineering seed")
    split = read(run / "split.json")
    ranges = split["metadata"]["ranges"]
    require(ranges["train"]["latest_label_ns"] < ranges["validation"]["decision_start_ns"],
            "Training labels overlap validation")
    cutoff = ranges["validation"]["latest_label_ns"]
    protected = split["metadata"]["protected_start_ns"]
    common_outcomes = None
    audited = []
    for cell in summary["cells"]:
        model, seed = cell["model"], cell["seed"]
        path = checked(run / model / f"seed_{seed}" / "predictions.csv")
        with path.open(newline="") as stream:
            result = score_rows(list(csv.DictReader(stream)), model, seed, cutoff, protected)
        if common_outcomes is None:
            common_outcomes = result["outcomes"]
        require(result["outcomes"] == common_outcomes, "Unpaired outcomes or timing across models")
        require(result["rows"] == ranges["test"]["rows"], "Missing evaluation observations")
        for metric, value in result["metrics"].items():
            require(math.isclose(value, cell["metrics"][metric], rel_tol=1e-7, abs_tol=1e-7),
                    f"Score mismatch: {model}/{metric}")
        audited.append({"model": model, "seed": seed, "rows": result["rows"],
                        "dates": result["dates"], **result["metrics"],
                        "training_wall_seconds": cell["metrics"]["training_wall_seconds"]})
    by_model = {c["model"]: c for c in audited}
    effect = by_model["no_memory"]["mean_nll"] - by_model["frem"]["mean_nll"]
    require(math.isclose(effect, analysis["inference"]["estimate"], abs_tol=1e-7), "Paired effect mismatch")
    for family, key in (("rejepa_v0_3", "receipt"), ("reng_jepa_v0_1", "summary")):
        read(root / receipt[family][key], receipt[family][key + "_sha256"])
    return {
        "schema_version": 1, "status": "PASS_RETAINED_ENGINEERING_EVIDENCE_AUDIT",
        "scientific_status": "NOT_EVALUATED", "new_model_fits": 0,
        "protected_outcomes_accessed": False, "cells": audited,
        "paired_effect_recomputed": effect, "retained_inference": analysis["inference"],
        "source_sha256": sources,
        "verified_source_root": source_root.relative_to(root).as_posix(),
        "audit_source_sha256": digest(Path(__file__)),
        "checks": ["source/data/receipt hashes", "complete configured cells", "exact paired outcomes",
                   "purged split metadata", "prediction execution and maturity ordering",
                   "independent stdlib Student-t score and equal-date/asset aggregation"],
        "limits": ["No refitting or independent scientific reproduction",
                   "Bootstrap interval copied unchanged from hashed original, not recomputed",
                   "Successor top-level receipt hashes checked; successor numbers not recomputed",
                   "Synthetic input; no market, profitability or confirmatory conclusion"],
    }


def table_tex(report: dict) -> str:
    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{lrrrr}",
             r"\toprule Model & NLL & MAE & Direction accuracy & Fit seconds \\", r"\midrule"]
    for cell in report["cells"]:
        label = cell["model"].replace("_", r"\_")
        lines.append(f"{label} & {cell['mean_nll']:.4f} & {cell['mae']:.4f} & "
                     f"{cell['direction_accuracy']:.4f} & {cell['training_wall_seconds']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              (r"\caption{Retained synthetic integration run only: four generated assets, 92 dates, "
              r"368 paired predictions per model, seed 11, two neural epochs. Lower NLL and MAE "
              r"are better. Timings are the original single-run measurements, not efficiency evidence.}"),
              r"\label{tab:engineering}", r"\end{table}"]
    inf = report["retained_inference"]
    lines += [(f"The recomputed control-minus-FREM NLL difference is {report['paired_effect_recomputed']:.6f} "
              f"nats. The retained 95\\% interval is [{inf['ci_lower']:.6f}, {inf['ci_upper']:.6f}], "
              f"with centered-bootstrap two-sided $p={inf['p_two_sided']:.6f}$. "
              f"The original analysis used {inf['n_resamples']} resamples and "
              f"{inf['block_length']}-date blocks; it is unchanged by this audit.")]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path,
                        help="Explicit local archive of original source; every original hash must match")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(not args.output.exists(), "Output exists; preserve the prior audit")
    report = audit(root, root / "research/all_admissible_runs_2026-10-11.json", source_root=args.source_root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "engineering_results.tex").write_text(table_tex(report))
    print(json.dumps({"status": report["status"], "cells": len(report["cells"]),
                      "effect": report["paired_effect_recomputed"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
